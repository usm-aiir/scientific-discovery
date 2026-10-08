"""Run the complete table retrieval pipeline.

Prepares data, builds indexes, runs BM25, BGE, or fusion retrieval,
evaluates results, manages experiment outputs, and supports interactive search."""
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import shutil
from types import SimpleNamespace
from ..data import (load_json, load_corpus, load_queries, check_id,
                   digest, write_json, write_run, read_run, Candidate)

CODE_ROOT = Path(__file__).resolve().parents[1]
ROOT = CODE_ROOT.parent


def resolve_device(requested):
    """Resolve and validate an inference device before loading a model."""
    import torch
    if not isinstance(requested, str) or not requested.strip():
        raise ValueError('Stage 1 device must be a nonempty string or "auto".')
    if requested == 'auto':
        return 'cuda:1' if torch.cuda.device_count() >= 2 else 'cuda:0' if torch.cuda.is_available() else 'cpu'
    try:
        parsed = torch.device(requested)
    except (RuntimeError, TypeError) as error:
        raise ValueError(f'Invalid Stage 1 device: {requested!r}') from error
    if parsed.type == 'cuda':
        if not torch.cuda.is_available():
            raise RuntimeError(f'CUDA device requested but CUDA is unavailable: {requested}')
        if parsed.index is not None and parsed.index >= torch.cuda.device_count():
            raise RuntimeError(f'CUDA device does not exist: {requested}')
    return str(parsed)


def device(settings):
    return resolve_device(settings['device'])


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
        qrels=Path(settings.get(f'{split}_qrels', data / (
            'Test_table_qrels.INSTRUCTOR_ONLY.tsv' if test else 'Val_table_qrels.tsv'))),
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


# Record the exact code and index used before a test run.
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
              'code_sha256': {name: digest(CODE_ROOT / name)
                              for name in CODE_FILES}}
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


# Historical code hashes record provenance; they are not a reuse requirement.
def check_frozen(output):
    """Protect reused test embeddings without requiring historical source paths."""
    frozen = load_json(output / 'frozen.json')
    if digest(output / 'embeddings.npy') != frozen['embeddings_sha256']:
        raise ValueError('Frozen corpus embeddings changed; choose a new --experiment name')


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


def reject_partial(directory):
    if directory.exists():
        partial = list(directory.glob('*.tmp'))
        if partial:
            raise ValueError(f'Partial files in {directory}; inspect them or choose a new --experiment name')


def workflow(settings, test=False):
    """Evaluate matching saved runs, or generate only their missing rankings."""
    # Evaluation is an experiment-only dependency; live retrieval does not need it.
    from . import evaluation
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
    evaluation.assess(args)
    show_results(args, needed)


def run(settings_path, model, split='val', experiment=None, qrels=None):
    if model not in BACKENDS:
        raise ValueError(f'Unknown retriever: {model}')
    settings = load_json(settings_path)
    if qrels is not None:
        # Resolve relative CLI paths before switching to the repository root.
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


def interactive_config(settings_path=None, experiment=None, index_path=None, device_name=None):
    """Resolve existing validation artifacts for an interactive, read-only search."""
    settings_file = (Path(settings_path).expanduser().resolve() if settings_path is not None
                     else CODE_ROOT / 'settings.json')
    settings = load_json(settings_file)
    requested_device = settings['device'] if device_name is None else device_name
    if index_path is not None:
        if experiment:
            raise ValueError('Choose either a Stage 1 index path or an experiment, not both.')
        index = Path(index_path).expanduser().resolve()
        if not index.is_dir():
            raise FileNotFoundError(
                f'Stage 1 index directory not found: {index}. Build it separately or provide '
                '--experiment for an existing named experiment.')
        manifest = index / 'prepared.json'
        if not manifest.is_file():
            raise FileNotFoundError(
                f'Stage 1 index is incomplete: missing {manifest}. Live inference does not '
                'build indexes automatically.')
        return dict(index=str(index), device=resolve_device(requested_device))
    if experiment:
        if Path(experiment).name != experiment or experiment in ('.', '..'):
            raise ValueError('Experiment must be a simple directory name')
        settings['experiment'] = experiment
    if settings_path is not None:
        for key in ('output_root', 'validation_output', 'test_output'):
            if key in settings and not Path(settings[key]).is_absolute():
                settings[key] = str((settings_file.parent / settings[key]).resolve())
    args = arguments(settings)
    index = args.output
    if not index.is_dir():
        raise FileNotFoundError(
            f'Stage 1 experiment index directory not found: {index}. Run the Stage 1 '
            'experiment separately before live inference.')
    manifest = index / 'prepared.json'
    if not manifest.is_file():
        raise FileNotFoundError(
            f'Stage 1 experiment index is incomplete: missing {manifest}. Live inference '
            'does not build indexes automatically.')
    return dict(index=str(index.resolve()), device=resolve_device(requested_device))


def load_search_corpus(index):
    """Load and validate the source corpus referenced by a completed index."""
    index = Path(index).expanduser().resolve()
    metadata = load_json(index / ('prepared.json' if (index / 'prepared.json').exists() else 'index.json'))
    recorded = Path(metadata['corpus']).expanduser()
    if recorded.is_absolute() and recorded.is_file():
        corpus_path = recorded.resolve()
    else:
        relative = Path(recorded.name) if recorded.is_absolute() else recorded
        candidates = [index / relative]
        candidates.extend(parent / relative for parent in index.parents)
        if recorded.is_absolute():
            candidates.insert(0, index / recorded.name)
        corpus_path = next((path.resolve() for path in candidates if path.is_file()), None)
        if corpus_path is None:
            searched = ', '.join(str(path) for path in candidates)
            raise FileNotFoundError(
                f'Stage 1 corpus not found. Index metadata records {recorded!s}; searched: {searched}')
    if metadata.get('corpus_sha256') and digest(corpus_path) != metadata['corpus_sha256']:
        raise ValueError(f'Corpus changed since indexing: {corpus_path}')
    return load_corpus(corpus_path)

CODE_FILES = [
    'data.py', 'stage1/text.py', 'stage1/pipeline.py',
    'stage1/evaluation.py', 'stage1/fusion.py',
    'stage1/retrievers/bm25.py', 'stage1/retrievers/bge.py',
    'stage1/retrievers/ranking.py',
]


def prepare_tables(args):
    from transformers import AutoConfig, AutoTokenizer
    from .retrievers.bge import CACHE
    if args.output.exists() and any(args.output.iterdir()):
        raise ValueError('prepare needs a new or empty output directory; inspect partial files or choose a new experiment')
    corpus = load_corpus(args.corpus)
    config = AutoConfig.from_pretrained(args.model, revision=args.revision, cache_dir=CACHE)
    revision = getattr(config, '_commit_hash', None) or args.revision
    tokenizer = AutoTokenizer.from_pretrained(args.model, revision=revision, cache_dir=CACHE)
    args.output.mkdir(parents=True, exist_ok=True)
    tokenizer.save_pretrained(args.output / 'tokenizer')
    from .text import prepare_text_inputs
    ids, encoded, clipped = prepare_text_inputs(corpus, tokenizer, args.max_length)
    write_json(args.output / 'table_ids.json', ids)
    write_json(args.output / 'table_inputs.json', encoded)
    write_json(args.output / 'truncated_tables.json', clipped)
    manifest = {'model': args.model, 'revision': revision, 'max_length': args.max_length,
                'fields': 'paper_title + caption + first row (headers) + all subsequent rows',
                'table_count': len(ids), 'truncated_tables': len(clipped),
                'candidate_depth': args.depth, 'top_k': getattr(args, 'top_k', 10),
                'rrf_constant': getattr(args, 'rrf_constant', 60), 'rrf_weights': [1, 1],
                'bm25_k1': 1.2, 'bm25_b': 0.75,
                'corpus': str(args.corpus), 'corpus_sha256': digest(args.corpus),
                'inputs_sha256': digest(args.output / 'table_inputs.json'),
                'ids_sha256': digest(args.output / 'table_ids.json'),
                'pooling': 'L2-normalized CLS; no query instruction', 'finetuned': False}
    print(f'Prepared {len(ids):,} tables; {len(clipped):,} exceeded {args.max_length} tokens.', flush=True)
    return manifest


def prepare(args):
    queries = load_queries(args.queries)
    manifest = prepare_tables(args)
    write_json(args.output / 'queries.json', queries)
    manifest.update(query_count=len(queries), queries_sha256=digest(args.queries),
                    qrels=str(args.qrels), qrels_sha256=digest(args.qrels),
                    saved_queries_sha256=digest(args.output / 'queries.json'))
    write_json(args.output / 'prepared.json', manifest)


def load_search_backend(name, index, device='cpu'):
    """Load an existing backend once; never build or modify experiment artifacts."""
    index = Path(index)
    if name in ('bm25_full', 'bm25_matched'):
        from .retrievers.bm25 import load_lexical
        manifest = load_json(index / 'prepared.json')
        model = load_lexical(index / f'{name}.npz', manifest)
        return model, manifest
    if name == 'bge':
        from .retrievers.bge import load_search_index
        return load_search_index(index, device)
    raise ValueError(f'Unknown search backend: {name}')


def search_query(query, model, resources, top_k=5):
    """Rank a live query with already-loaded backends; return (table ID, score)."""
    if not query.strip() or top_k < 1:
        raise ValueError('Enter a nonempty query and a positive top-k')
    if model == 'bm25':
        return resources['bm25_full'][0].search(query, top_k)
    if model in ('bge', 'fusion'):
        from .retrievers.bge import search_texts
        if model == 'bge':
            return search_texts(resources['bge'], [query], top_k)[0][0]
        from .fusion import fuse
        lexical, manifest = resources['bm25_matched']
        depth = manifest['candidate_depth']
        dense = search_texts(resources['bge'], [query], depth)[0][0]
        return fuse(lexical.search(query, depth), dense, depth, manifest['rrf_constant'])[:top_k]
    raise ValueError(f'Unknown retrieval model: {model}')


BACKENDS = {'fusion': ['bm25_matched', 'bge'], 'bm25': ['bm25_full'], 'bge': ['bge']}


class RetrievalPipeline:
    """Load existing retrieval resources once and return ranked candidate tables."""
    def __init__(self, retriever, resources, corpus, provenance=None):
        if retriever not in BACKENDS:
            raise ValueError(f'Unknown retriever: {retriever}')
        self.retriever = retriever
        self.resources = resources
        self.corpus = corpus
        self.provenance = provenance or {}

    @classmethod
    def from_settings(cls, settings_path=None, experiment=None, retriever='fusion', index_path=None,
                      device_name=None):
        if retriever not in BACKENDS:
            raise ValueError(f'Unknown retriever: {retriever}')
        config = interactive_config(settings_path, experiment, index_path, device_name)
        index = config['index']
        corpus = load_search_corpus(index)
        resources = {name: load_search_backend(name, index, config['device']) for name in BACKENDS[retriever]}
        manifest_path = Path(index) / 'prepared.json'
        manifest = load_json(manifest_path)
        provenance = dict(index=str(Path(index).resolve()), index_manifest_sha256=digest(manifest_path),
                          corpus_sha256=manifest.get('corpus_sha256'), retrieval_device=config['device'])
        return cls(retriever, resources, corpus, provenance)

    def search(self, query, top_k=5):
        if not isinstance(query, str) or not query.strip():
            raise ValueError('Enter a nonempty query.')
        if type(top_k) is not int or top_k < 1:
            raise ValueError('top_k must be a positive integer.')
        hits = search_query(query.strip(), self.retriever, self.resources, top_k)
        return [Candidate(uid, self.corpus.get(uid), rank, float(score))
                for rank, (uid, score) in enumerate(hits, 1)]
