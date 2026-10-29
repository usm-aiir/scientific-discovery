# Builds structural cells from Stage 1 tables and selects the global top Stage 2 cells.
import logging
from pathlib import Path
from .representation import build_structural_cells, REPRESENTATION_VERSION
from .source import load_table_sources


logger = logging.getLogger(__name__)

# Handles Stage 2 structural cell preparation and global BGE ranking.

# Stores the BGE selector, provenance, and optional original table source directory.
class BGECellRetrievalPipeline:
    """Rank pooled corpus cells, optionally enhancing structure with source HTML.

    ``table_source_dir=None`` is the normal setup and needs no HTML files.
    An explicitly supplied directory must exist, but individual HTML files
    are optional; unavailable or unusable source HTML uses corpus structure.
    """
    def __init__(self, selector, provenance=None, *, table_source_dir=None):
        self.selector = selector
        self.provenance = dict(provenance or {}, cell_representation=REPRESENTATION_VERSION)
        self.table_source_dir = (Path(table_source_dir).expanduser().resolve()
                                 if table_source_dir is not None else None)
        if self.table_source_dir is not None and not self.table_source_dir.is_dir():
            raise FileNotFoundError(f'Table source HTML directory not found: {self.table_source_dir}')

# Pools cells from all Stage 1 tables, ranks them with BGE, and returns the global top-k.
    def rank(self, query, candidates, cell_top_k):
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
                sources, html_path, warnings = self._source_context(candidate.table_id)
                cells, audit = build_structural_cells(
                    candidate.table_id, candidate.table, sources, html_path)
                for cell in cells:
                    cell.update(
                        stage1_table_rank=candidate.retrieval_rank,
                        stage1_table_score=candidate.retrieval_score,
                    )
                pooled.extend(cells)
                summary['candidate_cell_count'] = len(cells)
                summary['structure'] = audit
                if warnings:
                    summary['warnings'] = warnings
            except (KeyError, ValueError, RuntimeError, OSError) as error:
                # One malformed table must not discard evidence from valid candidates.
                detail = {
                    'table_id': candidate.table_id,
                    'error': str(error),
                    'error_type': type(error).__name__,
                }
                summary.update(status='error', **detail)
                errors.append(detail)
                logger.warning('Skipping Stage 2 table %s: %s', candidate.table_id, error)
            tables.append(summary)

        if not pooled:
            if not errors:
                errors.append({
                    'error': 'No candidate cells were available for ranking.',
                    'error_type': 'ValueError',
                })
            return {'status': 'error', 'tables': tables, 'cells': [], 'errors': errors}

        ranked = self.selector.rank_prepared_cells(query, pooled)
        logger.debug('Stage 2 ranked %d cells globally from %d tables.', len(pooled), len(tables))
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

    def _source_context(self, table_id):
        """Return optional HTML context, falling back to corpus-based structure.

        No source directory or matching HTML yields an empty context normally.
        HTML read/parse failures add a warning and the same empty context;
        this optional enhancement must never stop retrieval of valid corpus cells.
        """
        try:
            sources, path = load_table_sources(table_id, self.table_source_dir)
            return sources, path, []
        except (OSError, ValueError) as error:
            logger.warning('Source HTML unavailable for %s: %s; using corpus structure.', table_id, error)
            return (), None, [f'Source HTML unavailable; using corpus structure: {error}']
