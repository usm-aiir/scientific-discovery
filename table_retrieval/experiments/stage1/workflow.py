"""Builds indexes, writes saved runs, and evaluates them outside live retrieval."""
import os
from pathlib import Path

from ...data import load_json, write_json
from .data import write_run
from .config import ROOT, arguments, check_settings
from .indexing import prepare, prepare_test, check_frozen, reject_partial


def run(settings_path, model, split='val', experiment=None, qrels=None):
    """Run one offline retrieval evaluation from its dedicated settings file."""
    settings = load_json(settings_path)
    if qrels is not None:
        path = Path(qrels).resolve()
        if not path.is_file():
            raise FileNotFoundError(f'Qrels file not found: {path}')
        settings[f'{split}_qrels'] = str(path)
    os.chdir(ROOT)
    os.environ.setdefault('OMP_NUM_THREADS', '4')
    os.environ.setdefault('TOKENIZERS_PARALLELISM', 'false')
    if experiment:
        if Path(experiment).name != experiment or experiment in ('.', '..'):
            raise ValueError('Experiment must be a simple directory name')
        settings['experiment'] = experiment
    settings['retriever'] = model
    workflow(settings, test=split == 'test')

def show_results(args, names):
    for name in names:
        report = load_json(args.results / f'{name}.metrics.json')
        score = report['recall@10']
        print(f'{name}: Recall@10 = {100 * score:.2f}%' if score is not None else f'{name}: unscored')


def methods(retriever):
    return {'bm25': ['bm25_full', 'bm25_matched'], 'bge': ['bge'],
            'fusion': ['bm25_full', 'bm25_matched', 'bge', 'fusion']}[retriever]


# Record the exact code and index used before a test run.
def retrieve_bge(args):
    from .retrievers.bge import search
    rankings, clipped = search(args)
    write_run(getattr(args, 'rankings', args.output) / 'bge.run', rankings, 'bge_m3_dense')
    write_json(args.output / 'truncated_queries.json', clipped)


def retrieve_lexical(args):
    from .retrievers.bm25 import search
    for name, rankings in search(args).items():
        write_run(getattr(args, 'rankings', args.output) / f'{name}.run', rankings, name)


def retrieve_fusion(args):
    from .fusion import combine
    write_run(getattr(args, 'rankings', args.output) / 'fusion.run', combine(args), 'rrf60')


def workflow(settings, test=False):
    """Evaluate matching saved runs, or generate only their missing rankings."""
    from . import metrics
    args = arguments(settings, test)
    check_settings(settings, args)
    needed = methods(settings['retriever'])
    for directory in {args.output, args.rankings, args.results}:
        reject_partial(directory)
    missing = [name for name in needed if not (args.rankings / f'{name}.run').exists()]
    if not missing:
        # Recalculate from saved rankings: existing metrics must agree exactly.
        print('Evaluating saved rankings; retrieval was not rerun.', flush=True)
        args.results.mkdir(parents=True, exist_ok=True)
        metrics.assess(args)
        show_results(args, needed)
        return
    if not (args.output / 'prepared.json').exists():
        if test and settings['retriever'] == 'fusion':
            source_args = arguments(settings)
            if not (source_args.results / 'comparison.json').exists():
                raise ValueError('Complete fusion validation first: --model fusion --split val')
            check_settings(settings, source_args)
            prepare_test(args)
        elif test and args.source and (args.source / 'index.json').exists():
            check_settings(settings, arguments(settings))
            prepare_test(args)
        else:
            prepare(args)
    for directory in {args.rankings, args.results}:
        directory.mkdir(parents=True, exist_ok=True)
    if (args.output / 'frozen.json').exists():
        check_frozen(args.output)
    print(f'Generating missing rankings: {", ".join(missing)}.', flush=True)
    if any(name in missing for name in ('bm25_full', 'bm25_matched')):
        from .retrievers.bm25 import index_lexical
        lexical_args = arguments(settings) if (args.output / 'frozen.json').exists() else args
        reject_partial(lexical_args.output)
        index_lexical(lexical_args)
        retrieve_lexical(args)
    if 'bge' in missing:
        from .retrievers.bge import index_bge
        if not (args.output / 'index.json').exists():
            # The index stores a stable link to corpus embeddings outside it.
            if args.embeddings != args.output / 'embeddings.npy':
                if args.embeddings.exists() or (args.output / 'embeddings.npy').is_symlink():
                    raise ValueError('Partial or incompatible embeddings; choose a new --experiment name')
                args.embeddings.parent.mkdir(parents=True, exist_ok=True)
            index_bge(args)
        if not (args.rankings / 'bge.run').exists():
            retrieve_bge(args)
    if settings['retriever'] == 'fusion' and not (args.rankings / 'fusion.run').exists():
        retrieve_fusion(args)
    metrics.assess(args)
    show_results(args, needed)
