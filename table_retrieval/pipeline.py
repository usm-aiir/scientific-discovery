"""Coordinate preparation, indexing, ranking, and evaluation."""
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import shutil
from types import SimpleNamespace
from .data import (load_json, load_corpus, load_queries, check_id,
                   digest, write_json, write_run, read_run)
from . import evaluation

ROOT = Path(__file__).resolve().parents[1]

def device(settings):
    if settings['device'] != 'auto':
        return settings['device']
    import torch
    return 'cuda:1' if torch.cuda.device_count() >= 2 else 'cuda:0' if torch.cuda.is_available() else 'cpu'


def arguments(settings, test=False):
    data = Path(settings['data'])
    split = 'test' if test else 'val'
    experiment = settings.get('experiment')
    legacy = Path(settings['test_output'] if test else settings['validation_output'])
    # Existing recorded experiments keep every original path. New named runs
    # use the organized output tree, without moving historical artifacts.
    if experiment is None and legacy.exists():
        output = rankings = results = legacy
        source = Path(settings['validation_output']) if test else None
        embeddings = output / 'embeddings.npy'
    else:
        root = Path(settings.get('output_root', 'outputs'))
        name = experiment or 'default'
        output = root / 'indexes' / name / split
        rankings = root / 'rankings' / split / name
        results = root / 'results' / name / split
        source = root / 'indexes' / name / 'val' if test else None
        embeddings = root / 'embeddings' / name / 'corpus.npy'
    return SimpleNamespace(
        corpus=data / 'Corpus.json', queries=data / ('Test.json' if test else 'Val.json'),
        qrels=data / ('Test_table_qrels.INSTRUCTOR_ONLY.tsv' if test else 'Val_table_qrels.tsv'),
        output=output, rankings=rankings, results=results, source=source,
        embeddings=embeddings, model=settings['model'], revision=settings['revision'],
        max_length=settings['max_length'], depth=settings['candidate_depth'],
        top_k=settings.get('top_k', 10), rrf_constant=settings.get('fusion', {}).get('constant', 60),
        batch_size=settings['batch_size'], device=device(settings))


def check_settings(settings, args):
    if not 4 <= args.max_length <= 8192 or args.batch_size < 1 or args.depth < max(10, args.top_k) or args.top_k < 10:
        raise ValueError('Require max_length 4..8192, positive batch_size, and candidate_depth >= top_k >= 10')
    if args.rrf_constant < 1:
        raise ValueError('Fusion constant must be positive')
    manifest = args.output / 'prepared.json'
    if manifest.exists():
        saved = load_json(manifest)
        expected = dict(model=args.model, revision=args.revision, max_length=args.max_length,
                        candidate_depth=args.depth, top_k=args.top_k, rrf_constant=args.rrf_constant)
        for key, value in expected.items():
            if saved.get(key, {'top_k': 10}.get(key)) != value:
                raise ValueError(f'{key} differs from the saved experiment; choose a new --experiment name')
        for path, key in [(args.corpus, 'corpus_sha256'), (args.queries, 'queries_sha256'),
                          (args.qrels, 'qrels_sha256'),
                          (args.output / 'table_inputs.json', 'inputs_sha256'),
                          (args.output / 'table_ids.json', 'ids_sha256'),
                          (args.output / 'queries.json', 'saved_queries_sha256')]:
            if digest(path) != saved[key]:
                raise ValueError(f'{path} differs from the saved experiment; choose a new --experiment name')


def show_results(args, names):
    for name in names:
        report = load_json(args.results / f'{name}.metrics.json')
        score = report['recall@10']
        print(f'{name}: Recall@10 = {100 * score:.2f}%' if score is not None else f'{name}: unscored')


def methods(retriever):
    return {'bm25': ['bm25_full', 'bm25_matched'], 'bge': ['bge'],
            'fusion': ['bm25_full', 'bm25_matched', 'bge', 'fusion']}[retriever]

def prepare(args):
    from transformers import AutoConfig, AutoTokenizer
    from .retrievers.bge import CACHE
    if args.output.exists() and any(args.output.iterdir()):
        raise ValueError('prepare needs a new or empty output directory; inspect partial files or choose a new experiment')
    corpus, queries = load_corpus(args.corpus), load_queries(args.queries)
    config = AutoConfig.from_pretrained(args.model, revision=args.revision, cache_dir=CACHE)
    revision = getattr(config, '_commit_hash', None) or args.revision
    tokenizer = AutoTokenizer.from_pretrained(args.model, revision=revision, cache_dir=CACHE)
    args.output.mkdir(parents=True, exist_ok=True)
    tokenizer.save_pretrained(args.output / 'tokenizer')
    from .data import prepare_text_inputs
    ids, encoded, clipped = prepare_text_inputs(corpus, tokenizer, args.max_length)
    write_json(args.output / 'table_ids.json', ids)
    write_json(args.output / 'table_inputs.json', encoded)
    write_json(args.output / 'queries.json', queries)
    write_json(args.output / 'truncated_tables.json', clipped)
    manifest = {'model': args.model, 'revision': revision, 'max_length': args.max_length,
                'fields': 'paper_title + caption + first row (headers) + all subsequent rows',
                'table_count': len(ids), 'query_count': len(queries), 'truncated_tables': len(clipped),
                'candidate_depth': args.depth, 'top_k': getattr(args, 'top_k', 10),
                'rrf_constant': getattr(args, 'rrf_constant', 60), 'rrf_weights': [1, 1],
                'bm25_k1': 1.2, 'bm25_b': 0.75,
                'corpus': str(args.corpus), 'corpus_sha256': digest(args.corpus),
                'queries_sha256': digest(args.queries), 'qrels': str(args.qrels),
                'qrels_sha256': digest(args.qrels), 'inputs_sha256': digest(args.output / 'table_inputs.json'),
                'ids_sha256': digest(args.output / 'table_ids.json'),
                'saved_queries_sha256': digest(args.output / 'queries.json'),
                'pooling': 'L2-normalized CLS; no query instruction', 'finetuned': False}
    write_json(args.output / 'prepared.json', manifest)
    print(f'Prepared {len(ids):,} tables; {len(clipped):,} exceeded {args.max_length} tokens.', flush=True)


def freeze(source, output):
    if output.exists() and any(output.iterdir()):
        raise ValueError('Test output must be new or empty; existing results will not be overwritten')
    manifest = load_json(source / 'prepared.json')
    metadata = load_json(source / 'index.json')
    if metadata['prepared_sha256'] != digest(source / 'prepared.json'):
        raise ValueError('Source index manifest mismatch')
    for filename, key in [('table_ids.json', 'ids_sha256'), ('table_inputs.json', 'inputs_sha256')]:
        if digest(source / filename) != manifest[key]:
            raise ValueError(f'Source {filename} changed')
    if digest(manifest['corpus']) != manifest['corpus_sha256']:
        raise ValueError('Source corpus changed')
    frozen = {'frozen_at_utc': datetime.now(timezone.utc).isoformat(),
              'source_directory': str(source.resolve()), 'configuration': manifest,
              'source_index': metadata,
              'embeddings_sha256': digest(source / 'embeddings.npy'),
              'primary_metric': 'fusion Recall@10, grades >= 1, entire corpus',
              'test_tuning': False,
              'code_sha256': {name: digest(Path(__file__).parent / name)
                              for name in ['data.py', 'pipeline.py', 'evaluation.py', 'fusion.py',
                                           'retrievers/bm25.py', 'retrievers/bge.py', 'retrievers/tapas.py']}}
    output.mkdir(parents=True, exist_ok=True)
    write_json(output / 'frozen.json', frozen)
    return manifest, metadata


def prepare_test(args):
    # Freeze settings before reading the test query file or judgments.
    manifest, metadata = freeze(args.source, args.output)
    queries = load_queries(args.queries)
    qids = [check_id(query['query_id']) for query in queries]
    validation = load_json(args.source / 'queries.json')
    if set(qids) & {check_id(q['query_id']) for q in validation}:
        raise ValueError('Test and validation query IDs overlap')
    if {q['query'].strip() for q in queries} & {q['query'].strip() for q in validation}:
        raise ValueError('Test and validation query text overlaps')
    for filename in ['table_ids.json', 'table_inputs.json', 'embeddings.npy']:
        (args.output / filename).symlink_to(os.path.relpath((args.source / filename).resolve(), args.output.resolve()))
    shutil.copytree(args.source / 'tokenizer', args.output / 'tokenizer')
    write_json(args.output / 'queries.json', queries)
    manifest = dict(manifest, query_count=len(queries), queries_sha256=digest(args.queries),
                    saved_queries_sha256=digest(args.output / 'queries.json'),
                    qrels=str(args.qrels), qrels_sha256=digest(args.qrels), split='test')
    write_json(args.output / 'prepared.json', manifest)
    metadata = dict(metadata, prepared_sha256=digest(args.output / 'prepared.json'))
    write_json(args.output / 'index.json', metadata)
    print('Test preparation complete; reused corpus embeddings without re-encoding.', flush=True)


def check_frozen(output):
    frozen = load_json(output / 'frozen.json')
    for name, expected in frozen['code_sha256'].items():
        path = Path(__file__).parent / name
        if not path.exists() or digest(path) != expected:
            raise ValueError(f'Frozen evaluation code changed: {name}; use a new experiment directory')
    if digest(output / 'embeddings.npy') != frozen['embeddings_sha256']:
        raise ValueError('Frozen corpus embeddings changed')


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


def retrieve_tapas(args):
    from .retrievers.tapas import search
    rankings, metadata, similarity = search(args)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    write_run(args.output, rankings, 'tapas_dtr')
    settings = {key: str(value) if isinstance(value, Path) else value for key, value in vars(args).items()}
    settings.update(query_count=len(rankings), table_count=metadata['table_count'],
                    model_fingerprint=metadata['model_fingerprint'], similarity=similarity)
    write_json(args.output.with_suffix('.settings.json'), settings)


def reject_partial(directory):
    if directory.exists():
        partial = list(directory.glob('*.tmp'))
        if partial:
            raise ValueError(f'Partial files in {directory}; inspect them or choose a new --experiment name')


def workflow(settings, test=False):
    """Run one complete experiment, reusing only matching completed artifacts."""
    if settings['retriever'] == 'tapas':
        return tapas(settings, test)
    args = arguments(settings, test)
    check_settings(settings, args)
    needed = methods(settings['retriever'])
    for directory in {args.output, args.rankings, args.results}:
        reject_partial(directory)
    if all((args.rankings / f'{name}.run').exists() for name in needed):
        # Recalculate from saved rankings: existing metrics must agree exactly.
        args.results.mkdir(parents=True, exist_ok=True)
        evaluation.assess(args)
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
    if settings['retriever'] in ('bm25', 'fusion'):
        from .retrievers.bm25 import index_lexical
        lexical_args = arguments(settings) if (args.output / 'frozen.json').exists() else args
        reject_partial(lexical_args.output)
        index_lexical(lexical_args)
        retrieve_lexical(args)
    if settings['retriever'] in ('bge', 'fusion'):
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
    evaluation.assess(args)
    show_results(args, needed)


def tapas(settings, test=False):
    from .retrievers.tapas import index_tapas, checkpoint_fingerprint
    config = settings['tapas']
    data = Path(settings['data'])
    split = 'test' if test else 'val'
    root = Path(settings.get('output_root', 'outputs'))
    name = settings.get('experiment') or 'default'
    legacy = settings.get('experiment') is None and Path(config['index']).exists()
    index = Path(config['index']) if legacy else root / 'indexes' / name / 'tapas'
    run = (Path(config['run']) if not test else Path(config['run']).with_name('test.run')) if legacy else root / 'rankings' / split / name / 'tapas.run'
    result = run.with_suffix('.metrics.json') if legacy else root / 'results' / name / split / 'tapas.metrics.json'
    queries = data / ('Test.json' if test else 'Val.json')
    qrels = data / ('Test_table_qrels.INSTRUCTOR_ONLY.tsv' if test else 'Val_table_qrels.tsv')
    signature = dict(model_fingerprint=checkpoint_fingerprint(config['checkpoint']),
                     corpus_sha256=digest(data / 'Corpus.json'), queries_sha256=digest(queries),
                     qrels_sha256=digest(qrels), top_k=settings.get('top_k', 10))
    if signature['top_k'] < 10:
        raise ValueError('top_k must be >= 10 for Recall@10')
    for directory in {index, run.parent, result.parent}:
        reject_partial(directory)
    guard = run.with_suffix('.experiment.json')
    if guard.exists() and load_json(guard) != signature:
        raise ValueError('TAPAS configuration differs; choose a new --experiment name')
    if run.exists() and not guard.exists():
        saved = load_json(run.with_suffix('.settings.json'))
        if (saved['model_fingerprint'] != signature['model_fingerprint'] or
                saved['top_k'] != signature['top_k'] or Path(saved['queries']) != queries):
            raise ValueError('TAPAS saved settings differ; choose a new --experiment name')
    args = SimpleNamespace(model=Path(config['checkpoint']), output=index,
                           corpus=data / 'Corpus.json', device=device(settings), batch_size=config['eval_batch_size'])
    if (index / 'index.json').exists():
        metadata = load_json(index / 'index.json')
        if metadata['model_fingerprint'] != signature['model_fingerprint']:
            raise ValueError('TAPAS index checkpoint differs; choose a new --experiment name')
        if metadata.get('corpus_sha256', signature['corpus_sha256']) != signature['corpus_sha256']:
            raise ValueError('TAPAS index corpus differs; choose a new --experiment name')
    elif not run.exists():
        if not legacy:
            args.embeddings = root / 'embeddings' / name / 'tapas.npy'
        index_tapas(args)
    if not run.exists():
        run.parent.mkdir(parents=True, exist_ok=True)
        write_json(guard, signature)
        args.index, args.output, args.queries, args.top_k = index, run, queries, signature['top_k']
        retrieve_tapas(args)
    report = evaluation.evaluate(run, qrels, [q['query_id'] for q in load_queries(queries)])
    result.parent.mkdir(parents=True, exist_ok=True)
    write_json(result, report)
    score = report['recall@10']
    print(f'TAPAS: Recall@10 = {100 * score:.2f}%' if score is not None else 'TAPAS: unscored')


def run(settings_path, model, split='val', experiment=None):
    settings = load_json(settings_path)
    os.chdir(ROOT)
    os.environ.setdefault('OMP_NUM_THREADS', '4')
    os.environ.setdefault('TOKENIZERS_PARALLELISM', 'false')
    if experiment:
        if Path(experiment).name != experiment or experiment in ('.', '..'):
            raise ValueError('Experiment must be a simple directory name')
        settings['experiment'] = experiment
        if model == 'tapas':
            settings['tapas']['checkpoint'] = str(Path(settings.get('output_root', 'outputs')) / 'checkpoints' / experiment / 'best')
    settings['retriever'] = model
    workflow(settings, test=split == 'test')
