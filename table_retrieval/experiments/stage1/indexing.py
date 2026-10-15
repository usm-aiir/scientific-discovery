"""Offline preparation and frozen-index reuse; never called by live search."""
from datetime import datetime, timezone
import os
import shutil
from ...data import load_json, load_corpus, check_id, digest, write_json
from .data import load_queries
from .config import CODE_ROOT

CODE_FILES = [
    'data.py', 'stage1/text.py', 'stage1/pipeline.py',
    'stage1/fusion.py', 'experiments/stage1/metrics.py',
    'stage1/retrievers/bm25.py', 'stage1/retrievers/bge.py',
    'stage1/retrievers/ranking.py',
    'stage1/artifacts.py', 'stage1/retrieval.py',
    'experiments/stage1/indexing.py', 'experiments/stage1/workflow.py', 'experiments/stage1/data.py',
]


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


def reject_partial(directory):
    if directory.exists():
        partial = list(directory.glob('*.tmp'))
        if partial:
            raise ValueError(f'Partial files in {directory}; inspect them or choose a new --experiment name')


def prepare_tables(args):
    from transformers import AutoConfig, AutoTokenizer
    from ...stage1.retrievers.bge import CACHE
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
