"""Pool and rank cells in supplied candidate tables without running retrieval."""
from table_retrieval.data import load_json
from .bge.model import make_cells


DEFAULT_CELL_TOP_K = 3


def load_layouts(path=None):
    """Only structural layouts belong here, never example queries or labels."""
    if path is None:
        return {}
    layouts = load_json(path)
    if not isinstance(layouts, dict) or any(not isinstance(v, dict) or
            set(v) != {'rows_sha256', 'headers', 'body_coordinates', 'note'}
            for v in layouts.values()):
        raise ValueError('Layouts must be an object keyed by table ID with reviewed structural layouts.')
    return layouts


class BGECellRetrievalPipeline:
    """Pool cells from Stage 1 candidates and select one global BGE top-k."""
    def __init__(self, selector, layouts=None, provenance=None):
        self.selector = selector
        self.layouts = layouts or {}
        self.provenance = provenance or {}

    def rank(self, query, candidates, cell_top_k=DEFAULT_CELL_TOP_K):
        if not isinstance(query, str) or not query.strip():
            raise ValueError('Enter a nonempty query.')
        if type(cell_top_k) is not int or cell_top_k < 1:
            raise ValueError('cell_top_k must be a positive integer.')

        query = query.strip()
        pooled = []
        tables = []
        errors = []
        for candidate in candidates:
            summary = {
                'table_id': candidate.table_id,
                'retrieval_rank': candidate.retrieval_rank,
                'retrieval_score': candidate.retrieval_score,
                'status': 'scored',
                'candidate_cell_count': 0,
            }
            try:
                if candidate.table is None:
                    raise KeyError(f'Table ID missing from corpus: {candidate.table_id}')
                cells = make_cells(
                    candidate.table_id, candidate.table,
                    self.layouts.get(candidate.table_id))
                for cell in cells:
                    cell.update(
                        stage1_table_rank=candidate.retrieval_rank,
                        stage1_table_score=candidate.retrieval_score,
                    )
                pooled.extend(cells)
                summary['candidate_cell_count'] = len(cells)
            except (KeyError, ValueError, RuntimeError, OSError) as error:
                detail = {
                    'table_id': candidate.table_id,
                    'error': str(error),
                    'error_type': type(error).__name__,
                }
                summary.update(status='error', **detail)
                errors.append(detail)
            tables.append(summary)

        if not pooled:
            if not errors:
                errors.append({
                    'error': 'No candidate cells were available for ranking.',
                    'error_type': 'ValueError',
                })
            return {'status': 'error', 'tables': tables, 'cells': [], 'errors': errors}

        ranked = self.selector.rank_prepared_cells(query, pooled)
        selected = []
        for cell in ranked[:cell_top_k]:
            value = {key: cell[key] for key in (
                'global_rank', 'table_id', 'cell_id', 'row', 'column', 'value',
                'row_label', 'header_path', 'header_source', 'stage1_table_rank',
                'stage1_table_score', 'stage2_bge_score')}
            selected.append(value)
        return {
            'status': 'partial' if errors else 'complete',
            'tables': tables,
            'cells': selected,
            'errors': errors,
        }
