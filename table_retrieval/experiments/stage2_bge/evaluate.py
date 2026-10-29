"""Evaluate BGE cells on v2 labels and saved Stage 1 Top-5 tables."""
import argparse
import hashlib
import json
from pathlib import Path

from table_retrieval.data import COORDINATES, SCHEMA_VERSION, DEFAULT_DATA_DIR, RESULTS_DIR, write_json
from table_retrieval.live.stage1.artifacts import resolve_device
from table_retrieval.live.stage2.bge.model import BGECellSelector, DEFAULT_MODEL, DEFAULT_REVISION, resolve_checkpoint
from table_retrieval.live.stage2.representation import build_structural_cells
from .dataset import check_stage1_run, checkpoint_hashes, load_inputs
from .metrics import score_queries


def load_selector(checkpoint, device):
    """Use float32 on both CPU and GPU for reproducible evaluation."""
    return BGECellSelector(str(checkpoint), device=device, batch_size=32, max_length=512)


def evaluate_inputs(selector, corpus, queries, labels, candidates, cutoffs):
    """Rank complete candidate pools; labels never enter the scoring text."""
    records = []
    for index, query in enumerate(queries, 1):
        qid = str(query['query_id'])
        cells = []
        for table_id in candidates[qid]:
            prepared, _ = build_structural_cells(table_id, corpus[table_id])
            cells.extend(prepared)
        text_hash = hashlib.sha256(json.dumps(
            [(cell['table_id'], cell['cell_id'], cell['text']) for cell in cells],
            ensure_ascii=False).encode('utf-8')).hexdigest()
        ranked = selector.rank_prepared_cells(query['query'], cells)
        ranking = [(cell['table_id'], cell['cell_id']) for cell in ranked]
        pool = {(cell['table_id'], cell['cell_id']) for cell in cells}
        if len(ranking) != len(pool) or set(ranking) != pool:
            raise ValueError(f'Selector changed the candidate cell pool: {qid}')
        records.append({
            'query_id': qid, 'candidate_table_ids': candidates[qid],
            'candidate_text_sha256': text_hash,
            'ranking': ranking, 'gold': sorted(labels[qid]),
            'top_cells': [{key: cell[key] for key in (
                'table_id', 'cell_id', 'row', 'column', 'value', 'stage2_bge_score')}
                for cell in ranked[:max(cutoffs)]],
        })
        print(f'{index}/{len(queries)} questions ranked', flush=True)
    return records, {str(k): score_queries(records, k=k) for k in cutoffs}


def run(data_dir=DEFAULT_DATA_DIR, stage1_run=None, output=None, *, split='val', checkpoint=DEFAULT_MODEL,
        device='auto', cutoffs=(5, 10, 20)):
    """Score the complete saved candidate pools and write a new evaluation run."""
    if split not in ('val', 'test'):
        raise ValueError('Choose split val or test.')
    mode = 'pretrained' if checkpoint == DEFAULT_MODEL else 'trained'
    stage1_run = Path(stage1_run) if stage1_run is not None else RESULTS_DIR / 'stage1' / split / 'fusion.run'
    output = Path(output) if output is not None else RESULTS_DIR / 'stage2_bge' / f'{split}_{mode}'
    if output.exists():
        raise ValueError(f'Output already exists: {output}. Choose a new --output directory.')
    if not cutoffs or any(k < 1 for k in cutoffs):
        raise ValueError('Choose positive evaluation cutoffs.')
    stage1_run = check_stage1_run(stage1_run, split)
    device = resolve_device(device)
    checkpoint = resolve_checkpoint(checkpoint)
    pretrained = checkpoint == DEFAULT_MODEL
    corpus, queries, labels, candidates, provenance = load_inputs(data_dir, split, stage1_run)
    provenance.update(
        model=str(checkpoint), model_revision=DEFAULT_REVISION if pretrained else None,
        checkpoint_sha256=None if pretrained else checkpoint_hashes(checkpoint),
        device=device, dtype='float32', max_length=512, encoding_batch_size=32,
        cutoffs=sorted(set(cutoffs)), schema_version=SCHEMA_VERSION,
    )
    output.mkdir(parents=True)
    write_json(output / 'experiment.json', provenance)
    write_json(output / 'candidate_tables.json', [
        {'query_id': str(query['query_id']), 'candidate_table_ids': candidates[str(query['query_id'])]}
        for query in queries])
    selector = load_selector(checkpoint, device)
    records, metrics = evaluate_inputs(selector, corpus, queries, labels, candidates, provenance['cutoffs'])
    write_json(output / 'predictions.json', {
        'schema_version': SCHEMA_VERSION, 'cell_coordinates': COORDINATES,
        'split': split, 'records': records})
    write_json(output / 'metrics.json', metrics)
    return metrics


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--data-dir', type=Path, default=DEFAULT_DATA_DIR, help='Dataset directory (default: Behrooz v2)')
    parser.add_argument('--stage1-run', type=Path,
                        help='Saved ranking (default: results/stage1/<split>/fusion.run)')
    parser.add_argument('--split', choices=('val', 'test'), default='val')
    model = parser.add_mutually_exclusive_group()
    model.add_argument('--pretrained', action='store_true', help='Pinned pretrained BGE-M3 (default)')
    model.add_argument('--checkpoint', type=Path, help='Checkpoint or training run folder')
    parser.add_argument('--device', default='auto')
    parser.add_argument('--k', type=int, nargs='+', default=[5, 10, 20])
    parser.add_argument('--output', type=Path,
                        help='New output directory (default: results/stage2_bge/<split>_<model>)')
    args = parser.parse_args(argv)
    try:
        checkpoint = args.checkpoint if args.checkpoint is not None else DEFAULT_MODEL
        metrics = run(args.data_dir, args.stage1_run, args.output, split=args.split,
                      checkpoint=checkpoint, device=args.device, cutoffs=args.k)
        print(json.dumps(metrics, indent=2))
    except (OSError, ValueError, KeyError, RuntimeError) as error:
        parser.exit(1, f'Evaluation failed: {error}\n')


if __name__ == '__main__':
    main()
