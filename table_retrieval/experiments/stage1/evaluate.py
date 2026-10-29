# Retrieves tables from a completed index and scores validation or approved test judgments.
import argparse
from collections import defaultdict
import json
from pathlib import Path

from tqdm.auto import tqdm

from table_retrieval.data import check_id, digest, load_json, write_json
from table_retrieval.live.stage1.artifacts import resolve_device
from table_retrieval.live.stage1.fusion import fuse
from table_retrieval.live.stage1.retrievers.bge import load_search_index, search_texts
from table_retrieval.live.stage1.retrievers.bm25 import load_lexical
from .metrics import evaluate

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]


def load_queries(path):
    queries = load_json(path)
    if not isinstance(queries, list) or any(not isinstance(q, dict) or
            'query_id' not in q or 'query' not in q for q in queries):
        raise ValueError('Queries must be a list of objects with query_id and query fields')
    ids = [check_id(q['query_id']) for q in queries]
    if not queries or len(ids) != len(set(ids)):
        raise ValueError('Queries must be nonempty and have unique IDs')
    if any(not isinstance(q['query'], str) or not q['query'].strip() for q in queries):
        raise ValueError('Queries must contain nonempty text')
    return queries


# Preserve the existing TREC format, score precision and partial-file protection.
def write_run(path, rankings, name):
    path = Path(path)
    temporary = path.with_suffix(path.suffix + '.tmp')
    if path.exists():
        raise ValueError(f'Existing rankings {path}; choose a new experiment/output path')
    with temporary.open('x') as stream:
        for qid, hits in rankings.items():
            for rank, (uid, score) in enumerate(hits, 1):
                stream.write(f'{qid} Q0 {uid} {rank} {score:.12g} {name}\n')
    temporary.replace(path)


def read_run(path):
    result = defaultdict(list)
    for line in Path(path).read_text().splitlines():
        qid, _, uid, rank, score, _ = line.split()
        result[qid].append((int(rank), uid, float(score)))
    return {qid: [(uid, score) for _, uid, score in sorted(hits)] for qid, hits in result.items()}


# Batch queries exactly as the original Stage 1 evaluation did.
def retrieve_bge(index, queries, device, batch_size, depth):
    resource = load_search_index(index, device)
    rankings, query_clipped = {}, []
    for start in tqdm(range(0, len(queries), batch_size), desc='Retrieving queries'):
        batch = queries[start:start + batch_size]
        hits, clipped = search_texts(resource, [query['query'] for query in batch], depth)
        query_clipped.extend(str(batch[position]['query_id']) for position in clipped)
        rankings.update((check_id(query['query_id']), row) for query, row in zip(batch, hits))
    return rankings, query_clipped


# A fixed index is shared by validation and test; evaluation never rebuilds it.
def run(index, split='val', *, qrels=None, queries=None, output=None,
        model='fusion', device='auto', batch_size=2, k=10, rankings=None):
    if split not in ('val', 'test') or model not in ('bm25', 'bge', 'fusion'):
        raise ValueError('Choose split val/test and model bm25/bge/fusion.')
    if k < 1 or batch_size < 1:
        raise ValueError('Require positive k and batch_size.')
    if split == 'test' and qrels is None:
        raise ValueError('Test judgments are not distributed. Supply an approved --qrels file.')
    queries_path = Path(queries) if queries is not None else HERE / 'data' / f'{split}.json'
    qrels = Path(qrels) if qrels is not None else HERE / 'data' / 'val_qrels.tsv'
    if not qrels.is_file():
        raise FileNotFoundError(f'Judgments file not found: {qrels}')
    query_data = load_queries(queries_path)
    output = Path(output) if output is not None else ROOT / 'results' / 'stage1' / split
    needed = {'bm25': ['bm25_full', 'bm25_matched'], 'bge': ['bge'],
              'fusion': ['bm25_full', 'bm25_matched', 'bge', 'fusion']}[model]
    source = Path(rankings) if rankings is not None else output
    provenance = {'split': split, 'queries_sha256': digest(queries_path),
                  'qrels_sha256': digest(qrels), 'model': model, 'k': k,
                  'rankings_source': str(source.resolve())}
    if rankings is None:
        index = Path(index).resolve()
        manifest = load_json(index / 'prepared.json')
        if load_json(index / 'index.json')['prepared_sha256'] != digest(index / 'prepared.json'):
            raise ValueError('Index manifest changed.')
        if k > manifest['candidate_depth']:
            raise ValueError('k exceeds the saved index candidate depth.')
        provenance.update(index=str(index), prepared_sha256=digest(index / 'prepared.json'),
                          batch_size=batch_size, device=resolve_device(device))
    else:
        missing = [name for name in needed if not (source / f'{name}.run').is_file()]
        if missing:
            raise FileNotFoundError(f'Missing saved runs: {", ".join(missing)}')
        provenance['run_sha256'] = {name: digest(source / f'{name}.run') for name in needed}
    output.mkdir(parents=True, exist_ok=True)
    if list(output.glob('*.tmp')):
        raise ValueError('Partial files found; inspect them or choose a new --output.')
    # Refuse to reuse an output folder with different inputs or evaluation settings.
    write_json(output / 'experiment.json', provenance)
    if rankings is None:
        depth = manifest['candidate_depth']
        for name in ('bm25_full', 'bm25_matched'):
            if name not in needed or (source / f'{name}.run').exists():
                continue
            lexical = load_lexical(index / f'{name}.npz', manifest)
            hits = {check_id(q['query_id']): lexical.search(q['query'], depth) for q in query_data}
            write_run(source / f'{name}.run', hits, name)
        if 'bge' in needed and not (source / 'bge.run').exists():
            hits, clipped = retrieve_bge(index, query_data, provenance['device'], batch_size, depth)
            write_run(source / 'bge.run', hits, 'bge_m3_dense')
            write_json(output / 'truncated_queries.json', clipped)
        if model == 'fusion' and not (source / 'fusion.run').exists():
            lexical, dense = read_run(source / 'bm25_matched.run'), read_run(source / 'bge.run')
            hits = {check_id(q['query_id']): fuse(lexical.get(str(q['query_id']), []),
                    dense.get(str(q['query_id']), []), depth, manifest['rrf_constant'])[:k]
                    for q in query_data}
            write_run(source / 'fusion.run', hits, 'rrf60')
    reports = {}
    for name in needed:
        reports[name] = evaluate(source / f'{name}.run', qrels,
                                 [q['query_id'] for q in query_data], k=k)
        write_json(output / f'{name}.metrics.json', reports[name])
    return reports


def main(argv=None):
    parser = argparse.ArgumentParser(description='Evaluate a completed Stage 1 index or saved TREC runs.')
    parser.add_argument('--index', type=Path, default=ROOT / 'results/stage1/index')
    parser.add_argument('--split', choices=['val', 'test'], default='val')
    parser.add_argument('--qrels', type=Path, help='Required for test: supply approved external judgments')
    parser.add_argument('--queries', type=Path, help='Optional query-file override')
    parser.add_argument('--output', type=Path)
    parser.add_argument('--rankings', type=Path, help='Rescore existing *.run files without loading models')
    parser.add_argument('--model', choices=['bm25', 'bge', 'fusion'], default='fusion')
    parser.add_argument('--device', default='auto')
    parser.add_argument('--batch-size', type=int, default=2)
    parser.add_argument('--k', type=int, default=10)
    args = parser.parse_args(argv)
    try:
        print(json.dumps(run(**vars(args)), indent=2))
    except (OSError, ValueError, RuntimeError) as error:
        parser.exit(1, f'Evaluation failed: {error}\n')


if __name__ == '__main__':
    main()
