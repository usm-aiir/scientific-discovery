# Verifies the committed splits, evidence coordinates, and frozen candidate pools.
from itertools import combinations

from table_retrieval.data import digest, load_json
from .data import DATA, candidate_cells, query_labels, read_data, training_cells, training_pairs, HERE


# This command validates the existing splits; it never reshuffles them.
def validate_data():
    manifest = load_json(DATA / 'manifest.json')
    for name, saved in manifest['files'].items():
        if digest(DATA / name) != saved['sha256']:
            raise ValueError(f'Experiment data changed: {name}')
    splits = {name: read_data(f'{name}.json')['annotations'] for name in ('train', 'val', 'test')}
    runtime = read_data('runtime_train.json')['annotations']
    for train in (splits['train'], runtime):
        sets = [train, splits['val'], splits['test']]
        for left, right in combinations(sets, 2):
            for field in ('sample_id', 'query_id', 'paper_id'):
                if {str(item[field]) for item in left} & {str(item[field]) for item in right}:
                    raise ValueError(f'Split overlap in {field}')
    for name, examples in {**splits, 'runtime_train': runtime}.items():
        if len({item['sample_id'] for item in examples}) != len(examples):
            raise ValueError(f'Duplicate samples: {name}')
        for example in examples:
            training_cells(example, 'legacy')
    for split in ('val', 'test'):
        expected = {(str(item['query_id']), item['table_id'], cell['cell_id'])
                    for item in splits[split] for cell in item['evidence_cells']}
        lines = (DATA / f'{split}_cell_qrels.tsv').read_text().splitlines()
        if lines[0] != 'query_id\ttable_id\tcell_id\trelevance':
            raise ValueError(f'Invalid qrels header: {split}')
        actual = {tuple(line.split('\t')[:3]) for line in lines[1:] if int(line.split('\t')[3]) > 0}
        if actual != expected:
            raise ValueError(f'Qrels differ from evidence: {split}')
        snapshot = read_data(f'{split}_candidates.json')
        queries = query_labels(split)
        if len(snapshot['queries']) != len(queries) or {q['query_id'] for q in snapshot['queries']} != set(queries):
            raise ValueError(f'Frozen query IDs differ: {split}')
        for query in snapshot['queries']:
            if len(query['candidate_table_ids']) != 5 or len(set(query['candidate_table_ids'])) != 5:
                raise ValueError('Expected five unique Stage 1 candidates.')
            candidate_cells(query, snapshot)
    for config_name in ('config.json', 'runtime.json'):
        config = load_json(HERE / config_name)
        pairs, counts = training_pairs(read_data(config['train_split'])['annotations'], config)
        expected = manifest['training'][config_name]
        if len(pairs) != expected['positive_pairs'] or counts != expected['negative_assignments']:
            raise ValueError(f'Training pairs differ: {config_name}')
    return {name: len(examples) for name, examples in {**splits, 'runtime_train': runtime}.items()}


if __name__ == '__main__':
    print('Verified fixed data:', validate_data())
