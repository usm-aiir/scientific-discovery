#turn a table into individual cell records with structural context.
#that structural text is what Stage 2 BGE eventually scores.

# Converts a cell value into clean text.
def text_value(value):
    return '' if value is None else str(value).strip()


_text = text_value

# Extracts column header paths from the original table structure when available.
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

# Tries to infer column headers from raw table rows when the structure is clear.
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

# Finds the best available header information for each table column.
def header_context(table):
    """Return trustworthy header paths, or conservative column fallbacks."""
    rows = table.get('rows')
    if not isinstance(rows, list) or not rows or any(not isinstance(row, list) for row in rows):
        raise ValueError('Table rows must be a nonempty list of lists.')
    automatic = _source_header_paths(table) or _row_header_paths(rows)
    if automatic is not None:
        return automatic
    width = max(map(len, rows), default=0)
    return dict(columns={column: [f'Column {column}'] for column in range(width)},
                cells={}, source='column_fallback')

# Converts every non-empty table cell into a structured cell record for Stage 2.
def make_cells(table_id, table):
    """Create nonempty cells using source one-based rows, columns and rNcM IDs."""
    rows = table.get('rows')
    if not isinstance(rows, list) or not rows:
        raise ValueError('Table rows must be a nonempty list.')
    if any(not isinstance(row, list) for row in rows):
        raise ValueError('Each table row must be a list.')

    context = header_context(table)
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
                'cell_id': f'r{row_index + 1}c{column_index + 1}',
                'row': row_index + 1,
                'column': column_index + 1,
                'value': raw_value,
                'row_label': row_label,
                'header_path': header_path,
                'header_source': context['source'],
                'text': representation,
            })
    if not cells:
        raise ValueError('Table has no nonempty cells.')
    return cells
