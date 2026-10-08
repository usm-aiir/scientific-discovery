"""Rank original table cells with pretrained BGE-M3 embeddings."""

from table_retrieval.data import COORDINATES
from table_retrieval.stage2.structure import rows_digest


DEFAULT_MODEL = 'BAAI/bge-m3'
DEFAULT_REVISION = '5617a9f61b028005a4858fdac845db406aefb181'
DEFAULT_THRESHOLD = 0.5
DEFAULT_BATCH_SIZE = 32
MAX_LENGTH = 512
MODEL_CACHE = None


def _text(value):
    return '' if value is None else str(value).strip()


def _source_header_paths(table):
    """Read exact header spans when scraper cell metadata is available."""
    cells = table.get('cells')
    width = table.get('n_cols')
    if not isinstance(cells, list) or type(width) is not int or width < 1:
        return None
    if any(not isinstance(cell, dict) for cell in cells):
        return None

    body_rows = [cell.get('row') for cell in cells if cell.get('is_header') is False]
    if not body_rows or any(type(row) is not int for row in body_rows):
        return None
    body_start = min(body_rows)
    headers = [cell for cell in cells if cell.get('is_header') is True and
               type(cell.get('row')) is int and cell['row'] < body_start]
    if not headers:
        return None

    paths = {}
    for column in range(width):
        values = []
        for cell in sorted(headers, key=lambda item: (item['row'], item.get('col', -1))):
            start = cell.get('col')
            span = cell.get('colspan', 1)
            if type(start) is not int or type(span) is not int or span < 1:
                return None
            value = _text(cell.get('text'))
            if start <= column < start + span and value and value not in values:
                values.append(value)
        paths[column] = values or [f'Column {column}']
    return dict(columns=paths, cells={}, source='source_structure')


def _row_header_paths(rows):
    """Recover only header alignments that are unambiguous in the raw rows."""
    width = max(map(len, rows))
    if width < 1:
        return None
    lengths = [len(row) for row in rows]
    if len(set(lengths)) == 1:
        paths = {column: ([_text(rows[0][column])] if _text(rows[0][column])
                          else [f'Column {column}']) for column in range(width)}
        return dict(columns=paths, cells={}, source='flat_first_row')
    if len(rows[0]) == width and all(_text(value) for value in rows[0]):
        paths = {column: [_text(rows[0][column])] for column in range(width)}
        return dict(columns=paths, cells={}, source='flat_first_row')

    # A shorter multi-row header followed by full-width rows can reveal one
    # omitted leading stub column. Other ragged patterns remain ambiguous.
    body_start = next((index for index, row in enumerate(rows) if len(row) == width), None)
    if body_start is None or body_start < 2 or any(len(row) != width for row in rows[body_start:]):
        return None
    header_rows = rows[:body_start]
    lower_lengths = {len(row) for row in header_rows[1:]}
    if lower_lengths != {width - 1} or width < 2:
        return None

    top_stub = _text(header_rows[0][0]) if header_rows[0] else ''
    top_groups = [_text(value) for value in header_rows[0][1:] if _text(value)]
    if not top_stub or len(top_groups) != 1:
        return None

    aligned = [[top_stub] + [top_groups[0]] * (width - 1)]
    aligned.extend([[''] + [_text(value) for value in row] for row in header_rows[1:]])
    paths = {}
    for column in range(width):
        values = []
        for row in aligned:
            value = row[column]
            if value and value not in values:
                values.append(value)
        paths[column] = values or [f'Column {column}']
    return dict(columns=paths, cells={}, source='inferred_hierarchy')


def _layout_header_paths(rows, layout):
    """Use reviewed flattened headers only when automatic recovery failed."""
    if not layout or layout.get('rows_sha256') != rows_digest(rows):
        return None
    headers = layout.get('headers')
    coordinates = layout.get('body_coordinates')
    if not isinstance(headers, list) or not isinstance(coordinates, list):
        return None
    paths = {}
    ambiguous = set()
    for row in coordinates:
        if not isinstance(row, list) or len(row) != len(headers):
            return None
        for column, coordinate in enumerate(row):
            if not isinstance(coordinate, list) or len(coordinate) != 2:
                return None
            key = tuple(coordinate)
            path = [_text(headers[column])] if _text(headers[column]) else []
            if key in paths and paths[key] != path:
                ambiguous.add(key)
            else:
                paths[key] = path
    for key in ambiguous:
        paths.pop(key, None)
    return dict(columns={}, cells=paths, source='reviewed_layout')


def header_context(table, layout=None):
    """Return trustworthy header paths, or conservative column fallbacks."""
    rows = table.get('rows')
    if not isinstance(rows, list) or not rows or any(not isinstance(row, list) for row in rows):
        raise ValueError('Table rows must be a nonempty list of lists.')
    automatic = _source_header_paths(table) or _row_header_paths(rows)
    if automatic is not None:
        return automatic
    reviewed = _layout_header_paths(rows, layout)
    if reviewed is not None:
        return reviewed
    width = max(map(len, rows), default=0)
    return dict(columns={column: [f'Column {column}'] for column in range(width)},
                cells={}, source='column_fallback')


def make_cells(table_id, table, layout=None):
    """Create one searchable record for every nonempty original table cell."""
    rows = table.get('rows')
    if not isinstance(rows, list) or not rows:
        raise ValueError('Table rows must be a nonempty list.')
    if any(not isinstance(row, list) for row in rows):
        raise ValueError('Each table row must be a list.')

    context = header_context(table, layout)
    cells = []
    for row_index, row in enumerate(rows):
        for column_index, raw_value in enumerate(row):
            value = _text(raw_value)
            if not value:
                continue

            # Scientific tables normally keep the row name in their first
            # nonempty cell and the column name in the first row.
            row_label = next((_text(item) for item in row[:column_index + 1]
                              if _text(item)), value)
            header_path = context['cells'].get(
                (row_index, column_index),
                context['columns'].get(column_index, [f'Column {column_index}']))
            representation = f'{row_label} | {" > ".join(header_path)} | {value}'
            cells.append({
                'table_id': table_id,
                'cell_id': f'r{row_index}:c{column_index}',
                'row': row_index,
                'column': column_index,
                'value': raw_value,
                'row_label': row_label,
                'header_path': header_path,
                'header_source': context['source'],
                'text': representation,
            })
    if not cells:
        raise ValueError('Table has no nonempty cells.')
    return cells


class BGECellSelector:
    """Load BGE-M3 once and rank cells by cosine similarity to a question."""

    def __init__(self, checkpoint=DEFAULT_MODEL, device='auto', batch_size=DEFAULT_BATCH_SIZE):
        try:
            import torch
            from transformers import AutoModel, AutoTokenizer
        except ImportError as error:
            raise RuntimeError('Install torch and transformers to run BGE cell retrieval.') from error

        if batch_size < 1:
            raise ValueError('BGE batch size must be positive.')
        self.torch = torch
        if not isinstance(device, str) or not device.strip():
            raise ValueError('BGE device must be a nonempty string or "auto".')
        requested = ('cuda:0' if torch.cuda.is_available() else 'cpu') if device == 'auto' else device
        try:
            parsed_device = torch.device(requested)
        except (RuntimeError, TypeError) as error:
            raise ValueError(f'Invalid BGE device: {device!r}') from error
        if parsed_device.type == 'cuda':
            if not torch.cuda.is_available():
                raise RuntimeError(f'CUDA device requested but CUDA is unavailable: {requested}')
            if parsed_device.index is not None and parsed_device.index >= torch.cuda.device_count():
                raise RuntimeError(f'CUDA device does not exist: {requested}')
        self.device = str(parsed_device)
        self.checkpoint = checkpoint
        self.batch_size = batch_size
        revision = DEFAULT_REVISION if checkpoint == DEFAULT_MODEL else None
        options = dict(revision=revision, cache_dir=MODEL_CACHE)
        dtype = torch.float16 if self.device.startswith('cuda') else torch.float32
        self.tokenizer = AutoTokenizer.from_pretrained(checkpoint, **options)
        self.model = AutoModel.from_pretrained(
            checkpoint, torch_dtype=dtype, attn_implementation='eager', **options
        ).to(self.device).eval()
        self.revision = getattr(self.model.config, '_commit_hash', revision)

    def encode(self, texts):
        """Return normalized BGE CLS embeddings in input order."""
        vectors = []
        with self.torch.inference_mode():
            for start in range(0, len(texts), self.batch_size):
                batch = texts[start:start + self.batch_size]
                inputs = self.tokenizer(
                    batch, padding=True, truncation=True, max_length=MAX_LENGTH,
                    return_tensors='pt')
                output = self.model(**{key: value.to(self.device)
                                      for key, value in inputs.items()})
                vector = self.torch.nn.functional.normalize(
                    output.last_hidden_state[:, 0].float(), dim=-1)
                vectors.append(vector.cpu())
        result = self.torch.cat(vectors)
        if not self.torch.isfinite(result).all():
            raise ValueError('BGE produced non-finite cell embeddings.')
        return result

    def rank_prepared_cells(self, query, cells):
        """Rank a prepared cross-table cell pool with one query encoding."""
        if not isinstance(query, str) or not query.strip():
            raise ValueError('Enter a nonempty question.')
        if not cells:
            return []
        if any(not isinstance(cell, dict) or not isinstance(cell.get('text'), str)
               for cell in cells):
            raise ValueError('Prepared cells must contain text representations.')

        query_vector = self.encode([query.strip()])[0]
        cell_vectors = self.encode([cell['text'] for cell in cells])
        scores = cell_vectors @ query_vector
        order = self.torch.argsort(scores, descending=True, stable=True).tolist()

        ranked = []
        for global_rank, index in enumerate(order, 1):
            cell = dict(cells[index])
            cell.update(stage2_bge_score=float(scores[index]), global_rank=global_rank)
            ranked.append(cell)
        return ranked

    def rank_cells(self, query, table_id, table, layout=None):
        """Rank every nonempty cell and retain original table coordinates."""
        cells = make_cells(table_id, table, layout)
        ranked = self.rank_prepared_cells(query, cells)
        for cell in ranked:
            cell['score'] = cell.pop('stage2_bge_score')
            cell['rank'] = cell.pop('global_rank')
        return ranked

    def select(self, query, table_id, table, layout=None, threshold=DEFAULT_THRESHOLD):
        """Return cells above a cosine threshold using the shared Stage 2 shape."""
        if not -1 <= threshold <= 1:
            raise ValueError('BGE cosine threshold must be between -1 and 1.')
        ranked = self.rank_cells(query, table_id, table, layout)
        return {
            'table_id': table_id,
            'query': query.strip(),
            'model': self.checkpoint,
            'revision': self.revision,
            'threshold': threshold,
            'coordinate_system': COORDINATES,
            'cells': [cell for cell in ranked if cell['score'] >= threshold],
            'warnings': ['BGE cell scores are cosine similarities from an unfine-tuned baseline.'],
        }
