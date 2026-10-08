"""Globally rank cells in supplied Stage 1 candidate tables with BGE."""
import argparse
from datetime import datetime, timezone
from pathlib import Path

from table_retrieval.data import Candidate, SCHEMA_VERSION, load_json, write_predictions
from .bge.pipeline import create_pipeline
from .pipeline import DEFAULT_CELL_TOP_K


def add_model_arguments(parser):
    """Add the canonical local BGE checkpoint and output options."""
    parser.add_argument('--checkpoint', required=True, type=Path,
                        help='Local fine-tuned BGE checkpoint directory')
    parser.add_argument('--cell-device', default='auto')
    parser.add_argument('--layouts', type=Path)
    parser.add_argument('--cell-top-k', type=int, default=DEFAULT_CELL_TOP_K)
    parser.add_argument('--output', required=True, type=Path)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--query', required=True)
    parser.add_argument('--query-id', default='interactive')
    parser.add_argument('--candidates', required=True, type=Path,
                        help='JSON list with table_id, table, and optional retrieval_rank/retrieval_score')
    add_model_arguments(parser)
    args = parser.parse_args(argv)
    if not args.query.strip() or args.cell_top_k < 1:
        parser.error('Require a nonempty query and positive cell-top-k')
    try:
        values = load_json(args.candidates)
        if not isinstance(values, list):
            raise ValueError('Candidates must be a JSON list.')
        candidates = [Candidate.from_dict(value) for value in values]
        if args.output.exists() or args.output.with_name(args.output.name + '.partial').exists():
            raise ValueError('Output or partial output already exists; choose a new output path.')
        engine = create_pipeline(args.checkpoint, args.cell_device, args.layouts)
        evidence = engine.rank(args.query, candidates, args.cell_top_k)
        configuration = dict(
            engine.provenance, retriever='provided_candidates', table_top_k=len(candidates),
            cell_top_k=args.cell_top_k, cell_selection='global_top_k')
        result = dict(
            schema_version=SCHEMA_VERSION, query_id=args.query_id, query=args.query.strip(),
            created_at_utc=datetime.now(timezone.utc).isoformat(), configuration=configuration,
            status=evidence['status'], tables=evidence['tables'], cells=evidence['cells'],
            errors=evidence['errors'])
        _, failed = write_predictions(args.output, [result])
    except (OSError, ValueError, RuntimeError, KeyError, ImportError) as error:
        parser.exit(1, f'Cell selection failed: {error}\n')
    print(f'Saved results to {args.output}.')
    if failed:
        raise SystemExit(2)


if __name__ == '__main__':
    main()
