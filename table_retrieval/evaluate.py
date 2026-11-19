# Scores a finished run_batch folder against the validation or test labels.
import argparse
import json
import math
from pathlib import Path

from table_retrieval.data import DEFAULT_DATA_DIR, RESULTS_DIR, digest, load_corpus, load_json, table_split_paths, write_json
from table_retrieval.experiments.stage2_bge.dataset import load_split
from table_retrieval.experiments.stage2_bge.metrics import score_queries

CUTOFF = 5


# Averages table precision, recall and binary nDCG per question, as in Stage 1 evaluation.
def score_tables(records):
    precision = recall = ndcg = 0.0
    for record in records:
        gold = {table for table, _ in record['gold']}
        tables = record['candidate_table_ids']
        precision += len(set(tables) & gold) / CUTOFF
        recall += len(set(tables) & gold) / len(gold)
        dcg = sum(1 / math.log2(rank + 1) for rank, table in enumerate(tables, 1) if table in gold)
        ndcg += dcg / sum(1 / math.log2(rank + 1) for rank in range(1, min(CUTOFF, len(gold)) + 1))
    count = len(records)
    return {'precision@5': precision / count, 'recall@5': recall / count, 'ndcg@5': ndcg / count}


# Finds the newest completed run_batch folder.
def latest_run(results_dir=RESULTS_DIR):
    runs = sorted(path for path in Path(results_dir).glob('table_retrieval_*')
                  if (path / 'predictions.jsonl').is_file() and (path / 'summary.json').is_file())
    if not runs:
        raise FileNotFoundError(f'No run_batch results found in {results_dir}. Run table_retrieval.run_batch first.')
    return runs[-1]


# Finds which labelled split the run's question file is, by comparing file checksums.
def find_split(protocol, data_dir):
    for split in ('val', 'test'):
        if digest(table_split_paths(data_dir, split)['queries']) == protocol['sha256']['input']:
            return split
    raise ValueError('The run used a question file without labels; only Val.json and Test.json can be scored.')


# Scores the saved Top-5 tables and cells for every question in the split.
def evaluate(run_dir, data_dir=DEFAULT_DATA_DIR):
    run_dir, data_dir = Path(run_dir).resolve(), Path(data_dir).resolve()
    output = run_dir / 'metrics.json'
    if output.exists():
        raise ValueError(f'Already evaluated: {output}')
    if load_json(run_dir / 'summary.json').get('status') != 'completed':
        raise ValueError('The run did not complete; rerun it before evaluating.')
    protocol = load_json(run_dir / 'protocol.json')
    split = find_split(protocol, data_dir)
    corpus = load_corpus(data_dir / 'Corpus.json')
    queries, labels, _ = load_split(data_dir, split, corpus)
    predictions = [json.loads(line) for line in (run_dir / 'predictions.jsonl').read_text().splitlines()]
    by_id = {row['query_id']: row for row in predictions}
    if len(by_id) != len(predictions) or set(by_id) != set(labels):
        raise ValueError(f'The run does not cover exactly the {len(labels)} {split} questions.')
    records = [{
        'query_id': qid,
        'candidate_table_ids': [table['table_id'] for table in by_id[qid]['tables']],
        'ranking': [(cell['table_id'], cell['cell_id']) for cell in by_id[qid]['cells']],
        'gold': sorted(labels[qid]),
    } for qid in (str(query['query_id']) for query in queries)]
    if any(len(record['candidate_table_ids']) != CUTOFF or len(record['ranking']) != CUTOFF
           for record in records):
        raise ValueError(f'Every question needs {CUTOFF} tables and {CUTOFF} cells.')
    cells = score_queries(records, k=CUTOFF)
    metrics = {
        'split': split, 'questions': cells['queries'], 'gold_cells': cells['gold_cells'],
        'cells': {
            'precision@5': cells['precision@5'], 'recall@5': cells['recall@5'],
            'f1@5': cells['f1@5'], 'ndcg@5': cells['ndcg@5'],
            # Only the Top-5 cells are saved, so reciprocal rank stops at 5.
            'mrr@5': cells['mrr'],
            'correct_cells': cells['correct_cells'], 'returned_cells': cells['returned_cells'],
        },
        'tables': score_tables(records),
        'labels_sha256': {name: digest(path) for name, path in table_split_paths(data_dir, split).items()},
        'corpus_sha256': digest(data_dir / 'Corpus.json'),
        'predictions_sha256': digest(run_dir / 'predictions.jsonl'),
    }
    write_json(output, metrics)
    return output, metrics


# Prints the main scores after writing metrics.json next to the predictions.
def main(argv=None):
    parser = argparse.ArgumentParser(description='Score a run_batch results folder.')
    parser.add_argument('run', nargs='?', type=Path, help='run_batch results folder (default: newest)')
    parser.add_argument('--data-dir', type=Path, default=DEFAULT_DATA_DIR)
    args = parser.parse_args(argv)
    try:
        output, metrics = evaluate(args.run or latest_run(), args.data_dir)
    except (OSError, ValueError, KeyError) as error:
        parser.exit(1, f'{error}\n')
    print(f"{metrics['split']}: {metrics['questions']} questions")
    for group in ('tables', 'cells'):
        print(group + ': ' + ', '.join(f'{name} {value:.4f}' for name, value in metrics[group].items()
                                       if isinstance(value, float)))
    print(f'Saved {output}')


if __name__ == '__main__':
    main()
