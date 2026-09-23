"""Run a complete table retrieval experiment."""
import argparse
from pathlib import Path
import sys


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, epilog='Examples:\n'
        '  python -m table_retrieval --model bge --split val\n'
        '  python -m table_retrieval --model fusion --split test',
        formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--model', choices=['bm25', 'bge', 'tapas', 'fusion'],
                        help='Retriever; tapas is optional for reproducing the recorded comparison')
    parser.add_argument('--split', choices=['val', 'test'], default='val')
    parser.add_argument('--experiment', help='New experiment name; keeps previous outputs intact')
    parser.add_argument('--settings', type=Path, default=Path(__file__).with_name('settings.json'))
    argv = sys.argv[1:] if argv is None else argv
    if not argv:
        parser.print_help()
        return
    args = parser.parse_args(argv)
    if not args.model:
        parser.error('--model is required for retrieval')
    from .pipeline import run
    try:
        run(args.settings.resolve(), args.model, args.split, args.experiment)
    except (ValueError, FileNotFoundError) as error:
        parser.exit(1, f'{error}\n')


if __name__ == '__main__':
    main()
