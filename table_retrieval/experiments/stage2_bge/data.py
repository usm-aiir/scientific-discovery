# Loads the fixed experiment data and prepares the original training inputs.
from collections import Counter
from pathlib import Path
import random

from table_retrieval.data import digest, load_json
from table_retrieval.stage2.cells import make_cells
from table_retrieval.stage2.representation import structural_bge_text

HERE = Path(__file__).resolve().parent
DATA = HERE / 'data'


# Check the committed file hash before using a split or candidate snapshot.
def read_data(name):
    path = DATA / name
    expected = load_json(DATA / 'manifest.json')['files'][name]['sha256']
    if digest(path) != expected:
        raise ValueError(f'Experiment data changed: {name}')
    return load_json(path)


# A cell is identified by both its table and original row/column coordinates.
def cell_key(cell):
    return cell['table_id'], cell['cell_id']


# Training used table-aware text; recovered structural context was added at inference.
def training_cells(example, representation):
    cells = make_cells(example['table_id'], example['table'])
    if representation == 'table_aware':
        for cell in cells:
            context = {'header_path': cell['header_path'], 'row_context': []}
            cell['text'] = structural_bge_text(cell, example['table'], context)
    elif representation != 'legacy':
        raise ValueError(f'Unknown training representation: {representation}')
    by_id = {cell['cell_id']: cell for cell in cells}
    for saved in example['evidence_cells']:
        cell = by_id.get(saved['cell_id'])
        if cell is None or any(cell[key] != saved[key] for key in ('row', 'column', 'value')):
            raise ValueError(f'Evidence coordinates or value changed: {example["sample_id"]}')
    if not example['evidence_cells']:
        raise ValueError(f'Example has no evidence: {example["sample_id"]}')
    return cells


# Match the historical negative sampling order, including its cross-table fallback.
def training_pairs(examples, config):
    prepared, all_gold = [], set()
    for example in examples:
        cells = training_cells(example, config['training_representation'])
        gold = {saved['cell_id'] for saved in example['evidence_cells']}
        by_id = {cell['cell_id']: cell for cell in cells}
        positives = [by_id[saved['cell_id']] for saved in example['evidence_cells']]
        prepared.append((example, cells, positives, gold))
        all_gold.update(cell_key(cell) for cell in positives)
    never_gold = [cell for _, cells, _, _ in prepared for cell in cells
                  if cell_key(cell) not in all_gold]
    rng = random.Random(config['seed'])
    count = config['negatives_per_positive']
    pairs, counts = [], Counter()
    for example, cells, positives, gold in prepared:
        same_table = [cell for cell in cells if cell['cell_id'] not in gold
                      and cell_key(cell) not in all_gold]
        for positive in positives:
            rng.shuffle(same_table)
            negatives = same_table[:count]
            counts['same_table'] += len(negatives)
            needed = count - len(negatives)
            if needed:
                fallback = [cell for cell in never_gold if cell['table_id'] != example['table_id']]
                negatives += rng.sample(fallback, needed)
                counts['cross_table_fallback'] += needed
            pairs.append({'query': example['question'], 'positive': positive['text'],
                          'negative_texts': [cell['text'] for cell in negatives],
                          'negative_keys': [cell_key(cell) for cell in negatives]})
    return pairs, dict(counts)


# Group all annotated tables for a question; missing retrieved evidence stays in gold.
def query_labels(split):
    queries = {}
    for example in read_data(f'{split}.json')['annotations']:
        qid = str(example['query_id'])
        query = queries.setdefault(qid, {'question': example['question'], 'gold': set(), 'tables': set()})
        if query['question'] != example['question']:
            raise ValueError(f'Conflicting question text: {qid}')
        query['tables'].add(example['table_id'])
        query['gold'].update((example['table_id'], cell['cell_id']) for cell in example['evidence_cells'])
    return queries


# Use the live formatter with the source-recovered context frozen for the benchmark.
def candidate_cells(query, snapshot):
    cells = []
    for table_id in query['candidate_table_ids']:
        table = snapshot['tables'][table_id]
        for cell in make_cells(table_id, table):
            context = snapshot['context'][table_id][cell['cell_id']]
            cell['text'] = structural_bge_text(cell, table, context)
            cells.append(cell)
    return cells
