"""Find tables for each question using BM25, BGE, or both, and evaluate the results.

Previously saved search results can also be evaluated without running the search again.
"""
import argparse
from collections import defaultdict
import json
from pathlib import Path

from table_retrieval.data import DEFAULT_DATA_DIR, RESULTS_DIR, check_id, digest, load_json, table_split_paths, write_json
from table_retrieval.live.stage1.artifacts import resolve_device
from table_retrieval.live.stage1.fusion import fuse
from .metrics import evaluate

# Choose which rankings to generate and evaluate for each search method.
RUNS = {
    'bm25': ('bm25_matched',),
    'bge': ('bge',),
    'fusion': ('bm25_matched', 'bge', 'fusion'),
}


def load_queries(path):
    """Load questions and check that each has a unique ID and nonempty text."""
    queries = load_json(path)
    if not isinstance(queries, list) or not queries:
        raise ValueError('Queries must be a nonempty list')

    ids = set()
    for query in queries:
        if not isinstance(query, dict) or 'query_id' not in query or 'query' not in query:
            raise ValueError('Each query needs query_id and query fields')

        # Repeated IDs would mix results from different questions.
        qid = check_id(query['query_id'])
        if qid in ids:
            raise ValueError('Queries must have unique IDs')
        ids.add(qid)

        if not isinstance(query['query'], str) or not query['query'].strip():
            raise ValueError('Queries must contain nonempty text')
    return queries


def write_run(path, rankings, name):
    """Save ranked tables and scores without replacing an existing results file."""
    path = Path(path)
    temporary = path.with_suffix(path.suffix + '.tmp')
    if path.exists():
        raise ValueError(f'Existing rankings {path}; choose a new experiment/output path')

    # Write to a temporary file, then rename it once saving is complete.
    with temporary.open('x') as stream:
        for qid, hits in rankings.items():
            for rank, (uid, score) in enumerate(hits, 1):
                # TREC format: question ID, placeholder, table ID, rank, score, run name.
                stream.write(f'{qid} Q0 {uid} {rank} {score:.12g} {name}\n')
    temporary.replace(path)


def read_run(path):
    """Load saved results and put each question's tables in rank order."""
    result = defaultdict(list)
    for line in Path(path).read_text().splitlines():
        qid, _, uid, rank, score, _ = line.split()
        result[qid].append((int(rank), uid, float(score)))
    return {qid: [(uid, score) for _, uid, score in sorted(hits)] for qid, hits in result.items()}


def retrieve_bge(index, queries, device, batch_size, depth):
    """Search with BGE and record questions shortened to fit the model's limit."""
    from tqdm.auto import tqdm
    from table_retrieval.live.stage1.retrievers.bge import load_search_index, search_texts

    resource = load_search_index(index, device)
    rankings, query_clipped = {}, []

    # Search a few questions at a time to limit memory use.
    for start in tqdm(range(0, len(queries), batch_size), desc='Retrieving queries'):
        batch = queries[start:start + batch_size]
        hits, clipped = search_texts(resource, [query['query'] for query in batch], depth)

        # Save the IDs of shortened questions and match results to each question.
        query_clipped.extend(str(batch[position]['query_id']) for position in clipped)
        rankings.update((check_id(query['query_id']), row) for query, row in zip(batch, hits))
    return rankings, query_clipped


def generate_runs(index, queries, output, *, model, device, batch_size, k):
    """Generate and save rankings for the selected search method."""
    from table_retrieval.live.stage1.retrievers.bm25 import load_lexical

    manifest = load_json(index / 'prepared.json')
    depth = manifest['candidate_depth']

    for name in RUNS[model]:
        path = output / f'{name}.run'

        # Skip rankings that have already been saved.
        if path.exists():
            continue

        if name == 'bm25_matched':
            # Search by keyword using the table text included in the BGE inputs.
            lexical = load_lexical(index / f'{name}.npz', manifest)
            hits = {check_id(q['query_id']): lexical.search(q['query'], depth) for q in queries}
            write_run(path, hits, name)

        elif name == 'bge':
            # Search by similarity between question and table vectors.
            hits, clipped = retrieve_bge(index, queries, device, batch_size, depth)
            write_run(path, hits, 'bge_m3_dense')
            write_json(output / 'truncated_queries.json', clipped)

        else:
            lexical = read_run(output / 'bm25_matched.run')
            dense = read_run(output / 'bge.run')
            hits = {}

            # Fusion combines bm25 and BGE results using their rank positions.
            # Keep the first k tables from the combined ranking.
            for query in queries:
                qid = check_id(query['query_id'])
                hits[qid] = fuse(lexical.get(qid, []), dense.get(qid, []),
                                 depth, manifest['rrf_constant'])[:k]
            write_run(path, hits, 'rrf60')


def run(index, split='val', *, qrels=None, queries=None, output=None,
        model='fusion', device='auto', batch_size=2, k=10, rankings=None, data_dir=DEFAULT_DATA_DIR):
    """Generate or load rankings, then measure retrieval quality."""
    if split not in ('val', 'test') or model not in RUNS:
        raise ValueError('Choose split val/test and model bm25/bge/fusion.')
    if k < 1 or batch_size < 1:
        raise ValueError('Require positive k and batch_size.')
    paths = table_split_paths(data_dir, split)
    queries_path = Path(queries) if queries is not None else paths['queries']
    qrels = Path(qrels) if qrels is not None else paths['qrels']
    if not qrels.is_file():
        raise FileNotFoundError(f'Judgments file not found: {qrels}')

    query_data = load_queries(queries_path)
    query_ids = [q['query_id'] for q in query_data]
    output = Path(output) if output is not None else RESULTS_DIR / 'stage1' / split
    source = Path(rankings) if rankings is not None else output

    # Record the experiment settings and file fingerprints (hashes).
    # These help identify the exact inputs used for this evaluation.
    provenance = {
        'split': split, 'queries_sha256': digest(queries_path),
        'qrels_sha256': digest(qrels), 'model': model, 'k': k,
        'rankings_source': str(source.resolve()),
    }

    if rankings is None:
        # Before searching, check that the index settings have not changed.
        index = Path(index).resolve()
        manifest_hash = digest(index / 'prepared.json')
        if load_json(index / 'index.json')['prepared_sha256'] != manifest_hash:
            raise ValueError('Index manifest changed.')
        if k > load_json(index / 'prepared.json')['candidate_depth']:
            raise ValueError('k exceeds the saved index candidate depth.')

        device = resolve_device(device)
        provenance.update(index=str(index), prepared_sha256=manifest_hash,
                          batch_size=batch_size, device=device)
    else:
        # For saved rankings, check that all required results files exist.
        missing = [name for name in RUNS[model] if not (source / f'{name}.run').is_file()]
        if missing:
            raise FileNotFoundError(f'Missing saved runs: {", ".join(missing)}')
        provenance['run_sha256'] = {name: digest(source / f'{name}.run') for name in RUNS[model]}

    output.mkdir(parents=True, exist_ok=True)

    # Temporary files may indicate that an earlier save did not finish.
    if list(output.glob('*.tmp')):
        raise ValueError('Partial files found; inspect them or choose a new --output.')

    write_json(output / 'experiment.json', provenance)
    if rankings is None:
        generate_runs(index, query_data, output, model=model, device=device, batch_size=batch_size, k=k)

    # Compare the first k results with the relevance labels and save the metrics.
    reports = {}
    for name in RUNS[model]:
        reports[name] = evaluate(source / f'{name}.run', qrels, query_ids, k=k)
    write_json(output / 'metrics.json', reports)
    return reports


def main(argv=None):
    """Read command-line options and start retrieval or evaluation."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--index', type=Path, default=RESULTS_DIR / 'stage1/index')
    parser.add_argument('--split', choices=['val', 'test'], default='val')
    parser.add_argument('--data-dir', type=Path, default=DEFAULT_DATA_DIR, help='Dataset directory (default: Behrooz v2)')
    parser.add_argument('--qrels', type=Path, help='Optional override for the split\'s table judgments')
    parser.add_argument('--queries', type=Path, help='Optional query-file override')
    parser.add_argument('--output', type=Path)
    parser.add_argument('--rankings', type=Path, help='Rescore existing *.run files without loading models')
    parser.add_argument('--model', choices=list(RUNS), default='fusion')
    parser.add_argument('--device', default='auto')
    parser.add_argument('--batch-size', type=int, default=2)
    parser.add_argument('--k', type=int, default=10)
    args = parser.parse_args(argv)

    # Print the metrics, or show a clear error message if evaluation fails.
    try:
        print(json.dumps(run(**vars(args)), indent=2))
    except (OSError, ValueError, RuntimeError) as error:
        parser.exit(1, f'Evaluation failed: {error}\n')

if __name__ == '__main__':
    main()
