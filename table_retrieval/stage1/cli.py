"""
Command-line interface for running table retrieval experiments.

Lets you choose the retrieval model and dataset split, then runs the
corresponding experiment using the project settings.
"""
import argparse
from pathlib import Path
import sys


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, epilog='Examples:\n'
        '  python -m table_retrieval.stage1.cli --model bge --split val\n'
        '  python -m table_retrieval.stage1.cli --model fusion --split test',
        formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--model', choices=['bm25', 'bge', 'fusion'], help='Retriever to evaluate')
    parser.add_argument('--split', choices=['val', 'test'], default='val')
    parser.add_argument('--qrels', type=Path,
                        help='Judgments file for the selected split; defaults to its standard dataset qrels')
    parser.add_argument('--experiment', help='New experiment name; keeps previous outputs intact')
    parser.add_argument('--settings', type=Path, default=Path(__file__).resolve().parents[1] / 'settings.json')
    argv = sys.argv[1:] if argv is None else argv
    if not argv:
        parser.print_help()
        return
    args = parser.parse_args(argv)
    if not args.model:
        parser.error('--model is required for retrieval')
    from .pipeline import run
    try:
        run(args.settings.resolve(), args.model, args.split, args.experiment, qrels=args.qrels)
    except (ValueError, FileNotFoundError) as error:
        parser.exit(1, f'{error}\n')


if __name__ == '__main__':
    main()
