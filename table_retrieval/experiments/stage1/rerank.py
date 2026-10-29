"""Use a reranking model to reorder the first 10 tables from fusion search.

Compare the first five results before and after reranking
to see whether retrieval quality improves.
"""
import argparse
import json
from pathlib import Path

from table_retrieval.data import check_id, digest, load_json, write_json
from table_retrieval.live.stage1.artifacts import load_search_corpus
from table_retrieval.live.stage1.reranker import CANDIDATE_COUNT, TableReranker
from .evaluate import load_queries, read_run, write_run
from .metrics import evaluate


def run(rankings, queries, qrels, output, *, device='auto'):
    """Rerank saved fusion results and evaluate them against the original results."""
    source, output = Path(rankings).resolve(), Path(output).resolve()
    queries, qrels = Path(queries).resolve(), Path(qrels).resolve()

    # Save reranking results in a new folder.
    if output.exists():
        raise ValueError('Choose a new output directory for reranking.')

    # Check that the questions and relevance labels match the original experiment.
    # File fingerprints (hashes) detect changes in file contents.
    experiment = load_json(source / 'experiment.json')
    if (experiment['queries_sha256'] != digest(queries) or
            experiment['qrels_sha256'] != digest(qrels)):
        raise ValueError('Queries or judgments differ from the saved Stage 1 run.')
    if experiment['model'] != 'fusion' or experiment['k'] < CANDIDATE_COUNT:
        raise ValueError('Supply an original fusion run saved with --k 10 or greater.')

    # Check that the saved index settings still match the original experiment.
    index = Path(experiment['index'])
    prepared_hash = digest(index / 'prepared.json')
    if (experiment['prepared_sha256'] != prepared_hash or
            load_json(index / 'index.json')['prepared_sha256'] != prepared_hash):
        raise ValueError('Stage 1 index manifest changed.')

    corpus = load_search_corpus(index)
    query_data = load_queries(queries)
    query_ids = [check_id(query['query_id']) for query in query_data]
    source_run = source / 'fusion.run'

    # Score the saved ranking, with cutoff metrics at Top-5.
    baseline = evaluate(source_run, qrels, query_ids, k=5)
    hits = read_run(source_run)

    # Check that every question has enough candidates and their tables exist.
    expected = min(CANDIDATE_COUNT, len(corpus))
    if set(hits) != set(query_ids) or any(len(hits[qid]) < expected for qid in query_ids):
        raise ValueError('Saved fusion rankings do not cover every query and Top-10 candidate.')
    if any(uid not in corpus for qid in query_ids for uid, _ in hits[qid][:CANDIDATE_COUNT]):
        raise ValueError('A saved candidate is missing from the indexed corpus.')

    reranker = TableReranker(device)
    output.mkdir(parents=True)

    # Record the inputs, model details, and device used for this experiment.
    provenance = {
        'split': experiment['split'], 'model': 'fusion_reranked', 'k': 5,
        'index': str(index), 'prepared_sha256': prepared_hash,
        'queries_sha256': digest(queries), 'qrels_sha256': digest(qrels),
        'rankings_source': str(source), 'source_run_sha256': digest(source_run),
        'source_experiment_sha256': digest(source / 'experiment.json'),
        'table_reranker': reranker.provenance, 'device': reranker.device,
    }
    write_json(output / 'experiment.json', provenance)

    # Reorder the retrieved candidates for each question using the reranking model.
    reranked = {}
    for position, query in enumerate(query_data, 1):
        qid = check_id(query['query_id'])
        reranked[qid] = reranker.rank(query['query'], hits[qid], corpus)
        print(f'Reranked {position}/{len(query_data)} questions', flush=True)
    write_run(output / 'reranked.run', reranked, 'bge_reranker_v2_m3')

    # Compare both saved rankings using the same labels and cutoff.
    reports = {'fusion': baseline,
               'reranked': evaluate(output / 'reranked.run', qrels, query_ids, k=5)}
    write_json(output / 'metrics.json', reports)
    return reports


def main(argv=None):
    """Read command-line options and start the reranking experiment."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--rankings', type=Path, required=True,
                        help='Completed Stage 1 directory containing fusion.run and experiment.json')
    parser.add_argument('--queries', type=Path, required=True)
    parser.add_argument('--qrels', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True, help='New output directory')
    parser.add_argument('--device', default='auto')
    args = parser.parse_args(argv)

    # Print the metrics, or show a clear error message if reranking fails.
    try:
        print(json.dumps(run(**vars(args)), indent=2))
    except (OSError, ValueError, RuntimeError, KeyError, ImportError) as error:
        parser.exit(1, f'Table reranking failed: {error}\n')


if __name__ == '__main__':
    main()
