"""Train ModernBERT to route questions to sufficient modalities."""

import argparse
import importlib.metadata
import json
import math
import random
from datetime import datetime, timezone
from pathlib import Path

import torch
from torch.utils.data import DataLoader

from . import common
from .data import (ROUTES, ROUTING_TASK, collate_reviewed, describe_records,
                   file_hash, load_records)

MODEL_NAME = 'answerdotai/ModernBERT-base'
REVISION = '8949b909ec900327062f0ebf497f51aef5e6f0c8'


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--train-file', type=Path)
    parser.add_argument('--val-file', type=Path, help='Validation data for checkpoint selection; default: data/Val.json. Never Test.')
    parser.add_argument('--check-data', action='store_true', help='Validate inputs without loading weights or training.')
    parser.add_argument('--model', default=MODEL_NAME)
    parser.add_argument('--revision', default=REVISION)
    parser.add_argument('--epochs', type=int, default=3)
    parser.add_argument('--batch-size', type=int, default=8)
    parser.add_argument('--accumulation-steps', type=int, default=1)
    parser.add_argument('--learning-rate', type=float, default=2e-5)
    parser.add_argument('--max-length', type=int, default=256)
    parser.add_argument('--seed', type=int, default=42)
    args = parser.parse_args(argv)

    try:
        if any(getattr(args, name) <= 0 for name in ('epochs', 'batch_size', 'accumulation_steps', 'max_length')):
            raise ValueError('Epochs, batch sizes, accumulation steps and maximum length must be positive')
        if not math.isfinite(args.learning_rate) or args.learning_rate <= 0:
            raise ValueError('Learning rate must be finite and positive')
        files = {'Train': (args.train_file or common.DATA_DIR / 'Train.json').resolve(),
                 'Val': (args.val_file or common.DATA_DIR / 'Val.json').resolve()}
        hashes = {split: file_hash(path) for split, path in files.items()}
        training = load_records(files['Train'], 'Train')
        validation = load_records(files['Val'], 'Val')
        if {r['router_id'] for r in training} & {r['router_id'] for r in validation}:
            raise ValueError('Train and Val contain overlapping IDs')
        if {' '.join(r['query'].split()) for r in training} & {' '.join(r['query'].split()) for r in validation}:
            raise ValueError('Train and Val contain overlapping questions; preserve original splits')
        if hashes != {split: file_hash(path) for split, path in files.items()}:
            raise ValueError('Data changed while loading')
    except (ValueError, OSError, KeyError) as error:
        parser.error(str(error))

    check = {'task': ROUTING_TASK, 'train': describe_records(training),
             'validation': describe_records(validation),
             'data_files': {k: str(v) for k, v in files.items()}, 'data_sha256': hashes,
             'selection': 'highest Val routing accuracy; earliest epoch on ties',
             'model_input_fields': ['query']}
    print(json.dumps(check, indent=2), flush=True)
    if args.check_data:
        return check

    from transformers import get_linear_schedule_with_warmup

    random.seed(args.seed)
    torch.manual_seed(args.seed)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model, tokenizer = common.create_classifier(args.model, args.revision, args.max_length)
    model.to(device)
    loader = DataLoader(training, batch_size=args.batch_size, shuffle=True, collate_fn=collate_reviewed,
                        generator=torch.Generator().manual_seed(args.seed))
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.learning_rate, weight_decay=0.01)
    steps = math.ceil(len(loader) / args.accumulation_steps) * args.epochs
    scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=int(0.1 * steps), num_training_steps=steps)
    run_id = datetime.now(timezone.utc).strftime('run_%Y%m%d_%H%M%S_%f')
    checkpoint, results = common.MODEL_DIR / run_id, common.RESULTS_DIR / run_id
    checkpoint.mkdir(parents=True, exist_ok=False)
    results.mkdir(parents=True, exist_ok=False)
    settings = {**check, 'model': args.model, 'revision': args.revision, 'epochs': args.epochs,
                'batch_size': args.batch_size, 'accumulation_steps': args.accumulation_steps,
                'learning_rate': args.learning_rate, 'max_length': args.max_length, 'seed': args.seed,
                'device': str(device), 'route_order': ROUTES, 'checkpoint': str(checkpoint),
                'versions': {name: importlib.metadata.version(name) for name in ('torch', 'transformers')}}
    common.write_json(results / 'settings.json', settings)

    history, best_score = [], -1.0
    selection = None
    for epoch in range(1, args.epochs + 1):
        loss = common.train_epoch(model, tokenizer, loader, optimizer, scheduler, args.accumulation_steps, device)
        if hashes != {split: file_hash(path) for split, path in files.items()}:
            raise ValueError('Input data changed during training; checkpoint refused')
        metrics, outputs = common.evaluate(model, tokenizer, validation, args.batch_size, device)
        entry = {'epoch': epoch, 'train_loss': loss, 'validation': metrics}
        history.append(entry)
        score = metrics['known_acceptable_route_match_rate']
        if score > best_score:
            best_score = score
            model.save_pretrained(checkpoint)
            tokenizer.save_pretrained(checkpoint)
            selection = {'policy': 'validation', 'validation_used': True,
                         'selected_epoch': epoch, 'task': ROUTING_TASK,
                         'selection_metric': 'known_acceptable_route_match_rate', 'selection_score': score}
            common.write_json(results / 'val_metrics.json', metrics)
            common.write_json(results / 'val_predictions.json', outputs)
            common.write_json(checkpoint / 'training.json', {'settings': settings, 'selection': selection, 'history': history, 'status': 'running'})
        common.write_json(results / 'history.json', history)
        common.write_json(results / 'selection.json', selection)
        common.write_json(results / 'progress.json', {'completed_epochs': epoch, 'planned_epochs': args.epochs,
                          'status': 'complete' if epoch == args.epochs else 'running'})
        print(json.dumps(entry), flush=True)
    common.write_json(checkpoint / 'training.json', {'settings': settings, 'selection': selection,
                      'history': history, 'status': 'complete'})
    print('Checkpoint: ' + str(checkpoint))
    print('Results: ' + str(results))
    return checkpoint


if __name__ == '__main__':
    main()
