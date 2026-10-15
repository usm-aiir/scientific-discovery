# Standalone test CLI for Stage 2 cell retrieval.
# Not used by the live retrieval pipeline.
import argparse
from datetime import datetime, timezone
from pathlib import Path

from table_retrieval.data import Candidate, SCHEMA_VERSION, load_json, write_predictions
from table_retrieval.search import DEFAULT_CONFIG, load_config
from .bge.pipeline import create_pipeline


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--query', required=True)
    parser.add_argument('--query-id', default='interactive')
    parser.add_argument('--candidates', required=True, type=Path,
                        help='JSON list with table_id, table, and optional retrieval_rank/retrieval_score')
    parser.add_argument('--config', type=Path, default=DEFAULT_CONFIG,
                        help='Stage 2 runtime config (default: table_retrieval/config.json)')
    parser.add_argument('--output', required=True, type=Path)
    args = parser.parse_args(argv)
    if not args.query.strip():
        parser.error('Require a nonempty query')
    try:
        values = load_json(args.candidates)
        if not isinstance(values, list):
            raise ValueError('Candidates must be a JSON list.')
        candidates = [Candidate.from_dict(value) for value in values]
        if args.output.exists() or args.output.with_name(args.output.name + '.partial').exists():
            raise ValueError('Output or partial output already exists; choose a new output path.')
        config = load_config(args.config)
        engine = create_pipeline(config['stage2_checkpoint'], config['device'],
                                 max_length=config['stage2_max_length'],
                                 table_source_dir=config['table_source_dir'])
        evidence = engine.rank(args.query, candidates, config['cell_top_k'])
        configuration = dict(
            engine.provenance, retriever='provided_candidates', table_top_k=len(candidates),
            cell_top_k=config['cell_top_k'], cell_selection='global_top_k')
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
