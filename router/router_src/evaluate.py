"""Check how well a saved router chooses tables, figures and text."""

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path

import torch

from . import common
from .data import (check_test_coverage, evaluation_coverage, file_hash,
                   load_records, write_json)


def main(argv=None):
    """Load the saved router, check its predictions and save the scores."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--split', choices=['Val', 'Test'], default='Val')
    parser.add_argument('--data-file', type=Path)
    parser.add_argument('--checkpoint', type=Path)
    parser.add_argument('--batch-size', type=int, default=1)
    parser.add_argument('--output', type=Path, help='New output directory; refuses to overwrite existing results.')
    parser.add_argument('--allow-partial-test', action='store_true', help='Explicitly permit scoring an incomplete accepted Test subset.')
    args = parser.parse_args(argv)
    data_file = (args.data_file or common.DATA_DIR / (args.split + '.json')).resolve()
    try:
        if args.batch_size <= 0:
            raise ValueError('Batch size must be positive')
        if args.output and args.output.exists():
            raise ValueError('Output directory already exists; choose a fresh path')
        coverage, manifest_hash = evaluation_coverage(data_file, args.split)
        if args.split == 'Test':
            check_test_coverage(data_file, args.allow_partial_test)
        data_hash = file_hash(data_file)
        records = load_records(data_file, args.split)
        if file_hash(data_file) != data_hash:
            raise ValueError('Evaluation data changed while loading')
        checkpoint = (args.checkpoint or common.latest_checkpoint()).resolve()
        common.checkpoint_file_task(checkpoint)
    except (OSError, ValueError, KeyError) as error:
        parser.error(str(error))
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model, tokenizer = common.load_checkpoint(checkpoint, device)
    metrics, predictions = common.evaluate(model, tokenizer, records, args.batch_size, device)
    if file_hash(data_file) != data_hash:
        raise ValueError('Evaluation data changed during inference; results refused')
    if manifest_hash and file_hash(data_file.parent / 'manifest.json') != manifest_hash:
        raise ValueError('Evaluation manifest changed during inference; results refused')
    run_id = args.split.lower() + '_' + datetime.now(timezone.utc).strftime('%Y%m%d_%H%M%S_%f')
    output = args.output or common.RESULTS_DIR / run_id
    output.mkdir(parents=True, exist_ok=False)
    settings = {'checkpoint': str(checkpoint), 'checkpoint_config_sha256': file_hash(checkpoint / 'config.json'),
                'data_file': str(data_file), 'data_sha256': data_hash, 'manifest_sha256': manifest_hash, 'split': args.split,
                'model_input_fields': ['query'], 'device': str(device), 'batch_size': args.batch_size,
                'max_length': model.config.router_max_length, 'review_coverage': coverage,
                'partial_test_explicitly_allowed': args.allow_partial_test if args.split == 'Test' else False}
    write_json(output / 'metrics.json', metrics)
    write_json(output / 'predictions.json', predictions)
    write_json(output / 'settings.json', settings)
    n = metrics['count']
    hits = sum(p['known_acceptable_route_match'] for p in predictions)
    report = ('# ModernBERT routing evaluation\n\n'
              f'Split: {args.split}. Evaluated {n} accepted reviewed questions.\n\n'
              '| Metric | Result |\n| --- | ---: |\n'
              f'| Routing accuracy | {hits}/{n} ({100*metrics["known_acceptable_route_match_rate"]:.2f}%) |\n'
              f'| Sufficient-set coverage | {100*metrics["sufficient_set_coverage"]:.2f}% |\n'
              f'| Mean selected modalities | {metrics["mean_selected_modalities"]:.2f} |\n\n'
              'Accuracy matches any listed acceptable route; coverage also credits supersets. '
              'These measure agreement with provisional assistant labels, not human-gold or answer accuracy.\n\n'
              f'Predicted action counts: `{json.dumps(metrics["predicted_route_counts"])}`.\n\n'
              f'Target-kind counts and matches: `{json.dumps(metrics["by_target_kind"])}`.\n\n'
              f'Review coverage: `{json.dumps(coverage)}`.\n\n'
              f'Input SHA256: `{data_hash}`. No training or checkpoint selection occurred.\n')
    (output / 'REPORT.md').write_text(report)
    print(json.dumps({key: metrics[key] for key in ['count', 'known_acceptable_route_match_rate',
                       'sufficient_set_coverage', 'mean_selected_modalities']}, indent=2))
    print('Results: ' + str(output))
    return output


if __name__ == '__main__':
    main()
