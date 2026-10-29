"""Load v2 questions and source cell labels, and prepare explicit training negatives."""
from collections import defaultdict
import csv
from functools import lru_cache
import hashlib
import json
from pathlib import Path
import random
import re

from table_retrieval.data import COORDINATES, digest, load_corpus, load_json, table_split_paths as split_paths
from table_retrieval.experiments.stage1.evaluate import load_queries, read_run
from table_retrieval.experiments.stage1.metrics import evaluate as validate_run
from table_retrieval.live.stage2.cells import text_value
from table_retrieval.live.stage2.representation import build_structural_cells, REPRESENTATION_VERSION

HERE = Path(__file__).resolve().parent


def load_table_grades(path, query_ids, corpus):
    grades = {}
    for line in path.read_text().splitlines():
        qid, placeholder, table_id, grade = line.split()
        key = qid, table_id
        if (qid not in query_ids or table_id not in corpus or placeholder != '0' or
                grade not in ('0', '1', '2') or key in grades):
            raise ValueError(f'Invalid or duplicate v2 table judgment: {line}')
        grades[key] = int(grade)
    return grades


def load_cell_labels(path, query_ids, grades, corpus):
    labels = {qid: set() for qid in query_ids}
    with path.open(encoding='utf-8', newline='') as stream:
        # Quotes are literal source cell content, not TSV quoting syntax.
        reader = csv.DictReader(stream, delimiter='\t', quoting=csv.QUOTE_NONE)
        for cell in reader:
            qid, table_id, ref = cell['query_id'], cell['table_id'], cell['cell_ref']
            match = re.fullmatch(r'r([1-9]\d*)c([1-9]\d*)', ref)
            grade = grades.get((qid, table_id), 0)
            if match is None or grade < 1 or cell['relevance_score'] != str(grade):
                raise ValueError(f'Invalid source cell label: {qid}, {table_id}, {ref}')
            row, column = map(int, match.groups())
            table = corpus[table_id]
            rows = table['rows']
            if row > len(rows) or column > len(rows[row - 1]):
                raise ValueError(f'Cell outside corpus table: {table_id}, {ref}')
            value = ' '.join(text_value(rows[row - 1][column - 1]).split())
            if not value or value != cell['cell_content'] or cell['paper_id'] != table['paper_id']:
                raise ValueError(f'Source cell value or paper differs: {table_id}, {ref}')
            key = table_id, ref
            if key in labels[qid]:
                raise ValueError(f'Duplicate source cell: {qid}, {table_id}, {ref}')
            labels[qid].add(key)
    return labels


def load_split(data_dir, split, corpus):
    """Keep each source split and verify its table grades and positive cell labels."""
    paths = split_paths(data_dir, split)
    queries = load_queries(paths['queries'])
    query_ids = {str(query['query_id']) for query in queries}
    grades = load_table_grades(paths['qrels'], query_ids, corpus)
    labels = load_cell_labels(paths['cells'], query_ids, grades, corpus)
    if any(not gold for gold in labels.values()):
        raise ValueError('Every question must have positive cell labels.')
    positive_tables = {key for key, grade in grades.items() if grade >= 1}
    labeled_tables = {(qid, table) for qid, gold in labels.items() for table, _ in gold}
    if positive_tables != labeled_tables:
        raise ValueError('Positive table judgments and cell-labeled tables differ.')
    return queries, labels, grades


def check_stage1_run(stage1_run, split):
    """Require saved rankings and their metadata before loading models or data."""
    stage1_run = Path(stage1_run).resolve()
    if not stage1_run.is_file():
        raise FileNotFoundError(
            f'Stage 1 rankings not found: {stage1_run}. Run '
            f'python -m table_retrieval.experiments.stage1.evaluate --split {split} first, '
            'or supply another Stage 1 run.')
    metadata = stage1_run.parent / 'experiment.json'
    if not metadata.is_file():
        raise FileNotFoundError(f'Stage 1 metadata not found: {metadata}. Use a completed Stage 1 output folder.')
    return stage1_run


def load_inputs(data_dir, split, stage1_run):
    """Load source labels and check that Stage 1 used the same split and corpus."""
    data_dir, stage1_run = Path(data_dir).resolve(), check_stage1_run(stage1_run, split)
    paths = {
        'corpus': data_dir / 'Corpus.json', **split_paths(data_dir, split),
        'stage1_run': stage1_run, 'stage1_experiment': stage1_run.parent / 'experiment.json',
    }
    hashes = {name: digest(path) for name, path in paths.items()}
    experiment = load_json(paths['stage1_experiment'])
    if experiment['split'] != split or experiment['queries_sha256'] != hashes['queries']:
        raise ValueError('Stage 1 split or query file differs from the requested v2 data.')
    if experiment['qrels_sha256'] != hashes['qrels']:
        raise ValueError('Stage 1 judgments differ from the requested v2 data.')
    index = Path(experiment['index'])
    prepared_hash = digest(index / 'prepared.json')
    if (prepared_hash != experiment['prepared_sha256'] or
            load_json(index / 'index.json')['prepared_sha256'] != prepared_hash):
        raise ValueError('Stage 1 index provenance changed.')
    if load_json(index / 'prepared.json')['corpus_sha256'] != hashes['corpus']:
        raise ValueError('Stage 1 indexed a different corpus.')

    corpus = load_corpus(paths['corpus'])
    queries, labels, _ = load_split(data_dir, split, corpus)
    query_ids = {str(query['query_id']) for query in queries}
    validate_run(stage1_run, paths['qrels'], query_ids, k=5)
    rankings = read_run(stage1_run)
    if set(rankings) != query_ids:
        raise ValueError('Stage 1 rankings must cover every evaluation question.')
    candidates = {}
    for qid in query_ids:
        tables = [table_id for table_id, _ in rankings[qid][:5]]
        if len(tables) != 5 or any(table_id not in corpus for table_id in tables):
            raise ValueError(f'Expected five corpus tables from Stage 1: {qid}')
        candidates[qid] = tables
    provenance = {
        'split': split, 'paths': {name: str(path) for name, path in paths.items()},
        'sha256': hashes, 'stage1_prepared_sha256': prepared_hash,
        'table_top_k': 5, 'cell_coordinates': COORDINATES,
        'representation': REPRESENTATION_VERSION, 'source_html': None,
        'relevance': 'binary matched cells from table grades >= 1; all labeled gold retained',
    }
    return corpus, queries, labels, candidates, provenance


def training_pairs(queries, labels, grades, corpus, config):
    """Sample negatives from this question's explicitly grade-0 training tables."""
    negatives = defaultdict(list)
    for (qid, table_id), grade in grades.items():
        if grade == 0:
            negatives[qid].append(table_id)

    @lru_cache(maxsize=256)
    def texts(table_id):
        cells, _ = build_structural_cells(table_id, corpus[table_id])
        return {cell['cell_id']: cell['text'] for cell in cells}

    rng = random.Random(config['seed'])
    pairs, excluded, included = [], [], []
    pair_hash = hashlib.sha256()
    count = config['negatives_per_positive']
    for query in queries:
        qid = str(query['query_id'])
        if not negatives[qid]:
            excluded.append({'query_id': qid, 'positive_pairs': len(labels[qid]),
                             'reason': 'no explicitly grade-0 tables'})
            continue
        pool = [(table_id, ref, text) for table_id in sorted(negatives[qid])
                for ref, text in texts(table_id).items()]
        if len(pool) < count:
            raise ValueError(f'Not enough explicitly non-relevant cells for query {qid}.')
        included.append(qid)
        for table_id, ref in sorted(labels[qid]):
            selected = rng.sample(pool, count)
            pair = {
                'query_id': qid, 'query': query['query'].strip(),
                'positive_key': (table_id, ref), 'positive': texts(table_id)[ref],
                'negative_texts': [text for _, _, text in selected],
                'negative_keys': [(table, cell) for table, cell, _ in selected],
            }
            pairs.append(pair)
            pair_hash.update((json.dumps(pair, ensure_ascii=False, sort_keys=True) + '\n').encode('utf-8'))
    if not pairs:
        raise ValueError('No training pairs have explicitly non-relevant negative tables.')
    counts = {'explicit_grade0': len(pairs) * count}
    summary = {
        'total_queries': len(queries), 'included_queries': len(included),
        'included_query_ids': included, 'positive_pairs': len(pairs), 'excluded_queries': excluded,
        'excluded_positive_pairs': sum(item['positive_pairs'] for item in excluded),
        'negative_assignments': counts, 'pairs_sha256': pair_hash.hexdigest(),
        'negative_policy': 'query-specific explicit grade-0 tables only; no unjudged-cell fallback',
    }
    return pairs, counts, summary


def checkpoint_hashes(checkpoint):
    return {path.name: digest(path) for path in checkpoint.iterdir() if path.is_file()}
