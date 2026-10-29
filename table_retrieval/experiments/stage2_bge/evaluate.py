# Ranks cells in the fixed Stage 1 pool or rescores a supplied predictions file.
import argparse
import json
from pathlib import Path

from table_retrieval.data import digest, load_json, write_json
from table_retrieval.live.stage1.artifacts import resolve_device
from table_retrieval.live.stage2.bge.model import BGECellSelector
from .dataset import HERE, candidate_cells, cell_key, query_labels, read_data
from .metrics import score_queries


# Use float32 on both CPU and GPU, as in the published Structural BGE experiment.
def load_selector(checkpoint, device):
    import torch
    selector = BGECellSelector(str(checkpoint), device='cpu', batch_size=32, max_length=512)
    if device.startswith('cuda'):
        torch.backends.cuda.matmul.allow_tf32 = False
        torch.backends.cudnn.allow_tf32 = False
    selector.model.to(device)
    selector.device = device
    return selector


# Require every query and cell to belong to the committed benchmark pool.
def validate_rankings(records, snapshot, labels):
    expected = {query['query_id']: query for query in snapshot['queries']}
    if len(records) != len(expected) or {row['query_id'] for row in records} != set(expected):
        raise ValueError('Predictions must cover every benchmark query exactly once.')
    for record in records:
        qid = record['query_id']
        query = expected[qid]
        if record['candidate_table_ids'] != query['candidate_table_ids']:
            raise ValueError(f'Stage 1 candidates differ: {qid}')
        ranking = [tuple(item) for item in record['ranking']]
        pool = {cell_key(cell) for cell in candidate_cells(query, snapshot)}
        if len(ranking) != len(pool) or set(ranking) != pool:
            raise ValueError(f'Ranked cell pool differs: {qid}')
        if {tuple(item) for item in record['gold']} != labels[qid]['gold']:
            raise ValueError(f'Gold labels differ: {qid}')


def main(argv=None):
    parser = argparse.ArgumentParser(description='Evaluate Structural BGE on the fixed Stage 1 candidates.')
    parser.add_argument('--split', choices=['val', 'test'], default='test')
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument('--checkpoint', type=Path, help='A trained checkpoint, or its run folder containing training.json')
    mode.add_argument('--predictions', type=Path, help='Rescore an existing predictions JSON without loading BGE')
    parser.add_argument('--device', default='auto')
    parser.add_argument('--output', type=Path, help='New folder for predictions and metrics')
    args = parser.parse_args(argv)
    from .dataset import validate_data
    validate_data()
    if args.output and args.output.exists():
        parser.error('Output already exists; choose a new --output.')
    snapshot = read_data(f'{args.split}_candidates.json')
    labels = query_labels(args.split)
    if args.predictions:
        predictions = load_json(args.predictions)
        if predictions.get('split') != args.split:
            parser.error('Predictions split differs from --split.')
    else:
        if args.output is None:
            parser.error('--checkpoint requires --output to preserve the experiment results.')
        checkpoint = args.checkpoint
        if (checkpoint / 'training.json').is_file():
            checkpoint = checkpoint / load_json(checkpoint / 'training.json')['checkpoint']
        if not checkpoint.is_dir():
            parser.error(f'Checkpoint directory not found: {checkpoint}')
        selector = load_selector(checkpoint, resolve_device(args.device))
        records = []
        for index, query in enumerate(snapshot['queries'], 1):
            qid = query['query_id']
            ranked = selector.rank_prepared_cells(labels[qid]['question'], candidate_cells(query, snapshot))
            records.append({'query_id': qid, 'candidate_table_ids': query['candidate_table_ids'],
                            'ranking': [cell_key(cell) for cell in ranked], 'gold': sorted(labels[qid]['gold'])})
            print(f'{args.split} {index}/{len(snapshot["queries"])}', flush=True)
        predictions = {'split': args.split, 'representation': 'structural_v1',
                       'candidate_data_sha256': digest(HERE / 'data' / f'{args.split}_candidates.json'),
                       'device': selector.device, 'dtype': str(next(selector.model.parameters()).dtype),
                       'checkpoint_sha256': {path.name: digest(path) for path in checkpoint.iterdir() if path.is_file()},
                       'records': records}
    validate_rankings(predictions['records'], snapshot, labels)
    metrics = score_queries(predictions['records'])
    if args.output:
        args.output.mkdir(parents=True)
        write_json(args.output / 'predictions.json', predictions)
        write_json(args.output / 'metrics.json', metrics)
    print(json.dumps(metrics, indent=2))


if __name__ == '__main__':
    main()
