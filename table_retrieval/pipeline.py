"""Connect table retrieval to cell selection and package the results."""
from datetime import datetime, timezone

from .data import SCHEMA_VERSION
from .stage1.pipeline import RetrievalPipeline
from .stage2.bge.pipeline import create_pipeline
from .stage2.pipeline import DEFAULT_CELL_TOP_K


class EvidencePipeline:
    """Connect two loaded stages without owning their search or model logic."""
    def __init__(self, retrieval, cell_retrieval):
        self.retrieval = retrieval
        self.cell_retrieval = cell_retrieval

    @classmethod
    def from_settings(cls, settings_path=None, experiment=None, retriever='fusion',
                      checkpoint=None, stage1_device='auto', cell_device='auto', layouts_path=None,
                      stage1_index=None):
        if checkpoint is None:
            raise ValueError('A local fine-tuned BGE checkpoint is required.')
        if stage1_index is None and experiment is None:
            raise ValueError('A Stage 1 index path or named experiment is required.')
        retrieval = RetrievalPipeline.from_settings(
            settings_path, experiment, retriever, index_path=stage1_index,
            device_name=stage1_device)
        cell_retrieval = create_pipeline(checkpoint, cell_device, layouts_path)
        return cls(retrieval, cell_retrieval)

    def configuration(self, table_top_k, cell_top_k):
        return dict(self.retrieval.provenance, **self.cell_retrieval.provenance,
                    retriever=self.retrieval.retriever, table_top_k=table_top_k,
                    cell_top_k=cell_top_k, cell_selection='global_top_k')

    def search(self, query, table_top_k=5, cell_top_k=DEFAULT_CELL_TOP_K, query_id=None):
        candidates = self.retrieval.search(query, table_top_k)
        evidence = self.cell_retrieval.rank(query, candidates, cell_top_k)
        return dict(schema_version=SCHEMA_VERSION, query_id=str(query_id) if query_id is not None else None,
                    query=query.strip(), created_at_utc=datetime.now(timezone.utc).isoformat(),
                    configuration=self.configuration(table_top_k, cell_top_k),
                    status=evidence['status'], tables=evidence['tables'],
                    cells=evidence['cells'], errors=evidence['errors'])


def predict_queries(engine, queries, table_top_k=5, cell_top_k=DEFAULT_CELL_TOP_K):
    """Run a query batch and retain failed queries in the output."""
    for item in queries:
        try:
            yield engine.search(item['query'], table_top_k, cell_top_k, item['query_id'])
        except (OSError, ValueError, RuntimeError, KeyError) as error:
            yield dict(schema_version=SCHEMA_VERSION, query_id=str(item['query_id']), query=item['query'],
                       status='error', configuration=engine.configuration(table_top_k, cell_top_k),
                       error=str(error), error_type=type(error).__name__, tables=[], cells=[],
                       errors=[{'error': str(error), 'error_type': type(error).__name__}])


def main(argv=None):
    """Run both stages from the command line."""
    import argparse
    from pathlib import Path
    from .data import load_queries, write_predictions
    parser = argparse.ArgumentParser(description=__doc__)
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument('--query')
    source.add_argument('--queries', type=Path)
    parser.add_argument('--query-id', default='interactive')
    parser.add_argument('--retriever', choices=['fusion', 'bm25', 'bge'], default='fusion')
    parser.add_argument('--table-top-k', type=int, default=5)
    parser.add_argument('--cell-top-k', type=int, default=DEFAULT_CELL_TOP_K)
    parser.add_argument('--checkpoint', required=True, type=Path,
                        help='Local fine-tuned BGE checkpoint directory')
    parser.add_argument('--cell-device', default='auto')
    parser.add_argument('--stage1-device', default='auto')
    parser.add_argument('--layouts', type=Path)
    parser.add_argument('--output', required=True, type=Path)
    parser.add_argument('--settings', type=Path)
    stage1_source = parser.add_mutually_exclusive_group(required=True)
    stage1_source.add_argument('--stage1-index', type=Path,
                               help='Completed Stage 1 validation index directory')
    stage1_source.add_argument('--experiment',
                               help='Existing named Stage 1 experiment under output_root/indexes')
    args = parser.parse_args(argv)
    if args.table_top_k < 1 or args.cell_top_k < 1:
        parser.error('Require positive table-top-k and cell-top-k')
    try:
        queries = load_queries(args.queries) if args.queries else [dict(query_id=args.query_id, query=args.query)]
        if not all(q['query'].strip() for q in queries):
            raise ValueError('Queries must not be empty.')
        if args.output.exists() or args.output.with_name(args.output.name + '.partial').exists():
            raise ValueError('Output or partial output already exists; choose a new output path.')
        engine = EvidencePipeline.from_settings(
            settings_path=args.settings, experiment=args.experiment, retriever=args.retriever,
            checkpoint=args.checkpoint, stage1_device=args.stage1_device,
            cell_device=args.cell_device, layouts_path=args.layouts,
            stage1_index=args.stage1_index)
        count, failed = write_predictions(args.output, predict_queries(
            engine, queries, args.table_top_k, args.cell_top_k))
    except (OSError, ValueError, RuntimeError, KeyError, ImportError) as error:
        parser.exit(1, f'Pipeline failed: {error}\n')
    print(f'Saved {count} query results to {args.output}; {failed} with errors.')
    if failed:
        raise SystemExit(2)


if __name__ == '__main__':
    main()
