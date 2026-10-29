"""Train BGE on v2 cells and select by saved Stage 1 validation nDCG@5."""
import argparse
from functools import partial
import hashlib
import json
import math
import random
from pathlib import Path

from table_retrieval.data import COORDINATES, SCHEMA_VERSION, DEFAULT_DATA_DIR, RESULTS_DIR, digest, load_json, write_json
from table_retrieval.live.stage1.artifacts import resolve_device
from table_retrieval.live.stage2.bge.model import encode
from table_retrieval.live.stage2.ranking import rank_scored_cells
from table_retrieval.live.stage2.representation import build_structural_cells, REPRESENTATION_VERSION
from .dataset import HERE, checkpoint_hashes, load_inputs, load_split, split_paths, training_pairs
from .metrics import score_queries


def encode_batches(model, tokenizer, texts, config, device):
    """Encode validation cells in fixed-size batches and return CPU vectors."""
    import torch
    size = config['encoding_batch_size']
    vectors = [encode(model, tokenizer, texts[start:start + size], device, config['max_length']).cpu()
               for start in range(0, len(texts), size)]
    return torch.cat(vectors)


def contrastive_loss(model, tokenizer, batch, config, device):
    """Each positive cell competes with its sampled explicit negatives."""
    import torch
    queries = encode(model, tokenizer, [item['query'] for item in batch], device, config['max_length'])
    texts = [text for item in batch for text in [item['positive'], *item['negative_texts']]]
    candidates = encode(model, tokenizer, texts, device, config['max_length'])
    candidates = candidates.reshape(len(batch), 1 + config['negatives_per_positive'], -1)
    logits = torch.einsum('bd,bcd->bc', queries, candidates) / config['temperature']
    targets = torch.zeros(len(batch), dtype=torch.long, device=device)
    return torch.nn.functional.cross_entropy(logits, targets)


def train_epoch(model, tokenizer, loader, optimizer, config, device, epoch):
    """Accumulate gradients, including a correctly scaled final partial group."""
    import torch
    model.train()
    optimizer.zero_grad(set_to_none=True)
    accumulation = config['gradient_accumulation']
    remainder = len(loader) % accumulation
    total_loss = 0.0
    for step, batch in enumerate(loader, 1):
        with torch.autocast(device_type=device.split(':')[0], dtype=torch.bfloat16,
                            enabled=config.get('amp_bf16', False)):
            loss = contrastive_loss(model, tokenizer, batch, config, device)
        divisor = accumulation
        if remainder and step > len(loader) - remainder:
            divisor = remainder
        (loss / divisor).backward()
        if step % accumulation == 0 or step == len(loader):
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            optimizer.zero_grad(set_to_none=True)
        total_loss += loss.item()
        if step % 50 == 0 or step == len(loader):
            print(f'epoch {epoch} step {step}/{len(loader)} loss={loss.item():.6f}', flush=True)
    return total_loss / len(loader)


def fit(model, tokenizer, pairs, config, output, device, validate, *, data_info):
    """Keep epoch zero eligible and save a checkpoint only when nDCG@5 improves."""
    import torch
    import transformers
    if output.exists():
        raise ValueError(f'Output already exists; choose a new --output: {output}')
    if not pairs:
        raise ValueError('Training requires at least one positive pair.')
    random.seed(config['seed'])
    torch.manual_seed(config['seed'])
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(config['seed'])
    loader = torch.utils.data.DataLoader(
        pairs, batch_size=config['batch_size'], shuffle=True,
        generator=torch.Generator().manual_seed(config['seed']), collate_fn=list)
    optimizer = torch.optim.AdamW(model.parameters(), lr=config['learning_rate'])
    output.mkdir(parents=True)
    write_json(output / 'data_manifest.json', data_info)

    history, best, checkpoint = [], -1.0, None
    for epoch in range(config['epochs'] + 1):
        loss = None
        if epoch > 0:
            loss = train_epoch(model, tokenizer, loader, optimizer, config, device, epoch)
        metrics = validate(model, tokenizer)
        write_json(output / f'validation_epoch_{epoch}.metrics.json', metrics)
        ndcg = metrics['ndcg@5']
        history.append({'epoch': epoch, 'training_loss': loss, 'validation_ndcg': ndcg})
        print(f'epoch {epoch}: validation nDCG={ndcg:.6f}', flush=True)
        if ndcg > best:
            best = ndcg
            checkpoint = output / f'epoch_{epoch}'
            model.save_pretrained(checkpoint)
            tokenizer.save_pretrained(checkpoint)

    result = {
        'config': config, 'best_validation_ndcg': best, 'checkpoint': checkpoint.name,
        'epochs': history, 'training_examples': len(data_info['training']['included_query_ids']),
        'positive_pairs': len(pairs), 'negative_assignments': data_info['training']['negative_assignments'],
        'device': device, 'torch_version': torch.__version__,
        'dtype': str(next(model.parameters()).dtype), 'transformers_version': transformers.__version__,
        'data_manifest_sha256': digest(output / 'data_manifest.json'),
        'checkpoint_sha256': checkpoint_hashes(checkpoint),
    }
    write_json(output / 'training.json', result)
    return result


def validation_cells(corpus, queries, candidates):
    """Freeze scoring text once; representations are shared with live/evaluation code."""
    tables, prepared = {}, []
    text_hash = hashlib.sha256()
    for query in queries:
        qid = str(query['query_id'])
        cells = []
        for table_id in candidates[qid]:
            if table_id not in tables:
                values, _ = build_structural_cells(table_id, corpus[table_id])
                tables[table_id] = [{key: cell[key] for key in ('table_id', 'cell_id', 'text')}
                                    for cell in values]
            cells.extend(tables[table_id])
        prepared.append({'query_id': qid, 'query': query['query'].strip(),
                         'candidate_table_ids': candidates[qid], 'cells': cells})
        text_hash.update(json.dumps(
            [qid, [(cell['table_id'], cell['cell_id'], cell['text']) for cell in cells]],
            ensure_ascii=False).encode('utf-8'))
    return prepared, text_hash.hexdigest()


def evaluate_validation(model, tokenizer, prepared, labels, config, device):
    """Match float32 baseline scoring, including all queries and unavailable gold."""
    import torch
    model.eval()
    if next(model.parameters()).dtype != torch.float32:
        raise ValueError('V2 checkpoint selection requires float32 model parameters.')
    records = []
    with torch.inference_mode():
        for query in prepared:
            cells = query['cells']
            question = encode(model, tokenizer, [query['query']], device, config['max_length'])[0].cpu()
            vectors = encode_batches(model, tokenizer, [cell['text'] for cell in cells], config, device)
            ranked = rank_scored_cells(cells, vectors @ question)
            records.append({
                'query_id': query['query_id'], 'candidate_table_ids': query['candidate_table_ids'],
                'ranking': [(cell['table_id'], cell['cell_id']) for cell in ranked],
                'gold': sorted(labels[query['query_id']]),
            })
    return score_queries(records, k=5)


def prepare_data(data_dir, val_stage1_run, config):
    corpus, validation, val_labels, candidates, val_provenance = load_inputs(
        data_dir, 'val', val_stage1_run)
    queries, labels, grades = load_split(data_dir, 'train', corpus)
    if ({str(q['query_id']) for q in queries} & {str(q['query_id']) for q in validation} or
            {q['query'].strip() for q in queries} & {q['query'].strip() for q in validation}):
        raise ValueError('Training and validation questions overlap.')
    pairs, counts, summary = training_pairs(queries, labels, grades, corpus, config)
    prepared, text_hash = validation_cells(corpus, validation, candidates)
    paths = split_paths(data_dir, 'train')
    training_tables = {pair['positive_key'][0] for pair in pairs}
    val_tables = {table for gold in val_labels.values() for table, _ in gold}
    manifest = {
        'schema_version': SCHEMA_VERSION, 'cell_coordinates': COORDINATES,
        'representation': REPRESENTATION_VERSION,
        'corpus_sha256': val_provenance['sha256']['corpus'],
        'training': dict(summary, paths={key: str(path) for key, path in paths.items()},
                         sha256={key: digest(path) for key, path in paths.items()}),
        'validation': dict(val_provenance, questions=len(validation), scoring_text_sha256=text_hash,
                           selection_metric='pipeline binary nDCG@5; all validation questions'),
        'split_overlap': {
            'positive_tables': len(training_tables & val_tables),
            'positive_papers': len({corpus[t]['paper_id'] for t in training_tables} &
                                   {corpus[t]['paper_id'] for t in val_tables}),
            'policy': 'preserve provided question splits; paper/table overlap is allowed and reported',
        },
        'test_data_used': False,
    }
    return pairs, counts, prepared, val_labels, manifest


def load_config(path):
    config = dict(load_json(path))
    for name in ('epochs', 'batch_size', 'gradient_accumulation', 'negatives_per_positive',
                 'max_length', 'encoding_batch_size'):
        if type(config[name]) is not int or config[name] < 1:
            raise ValueError(f'{name} must be a positive integer.')
    for name in ('learning_rate', 'temperature'):
        if not math.isfinite(config[name]) or config[name] <= 0:
            raise ValueError(f'{name} must be finite and positive.')
    if type(config['seed']) is not int:
        raise ValueError('seed must be an integer.')
    if config['max_length'] != 512 or config['encoding_batch_size'] != 32:
        raise ValueError('V2 validation requires max_length=512 and encoding_batch_size=32 to match evaluation.')
    config.update(train_split='Train.json', training_representation=REPRESENTATION_VERSION,
                  selection_metric='validation pipeline binary nDCG@5',
                  validation_precision='float32', initial_checkpoint_eligible=True)
    return config


def load_training_model(config, device):
    """Keep float32 parameters; CUDA may use bfloat16 only inside training steps."""
    import torch
    from transformers import AutoModel, AutoTokenizer

    config['amp_bf16'] = False
    if device.startswith('cuda'):
        with torch.cuda.device(device):
            config['amp_bf16'] = torch.cuda.is_bf16_supported()
        torch.backends.cuda.matmul.allow_tf32 = False
        torch.backends.cudnn.allow_tf32 = False
    options = {'revision': config['revision']}
    tokenizer = AutoTokenizer.from_pretrained(config['model'], **options)
    model = AutoModel.from_pretrained(config['model'], torch_dtype=torch.float32,
                                     attn_implementation='eager', **options).to(device)
    if device.startswith('cuda'):
        model.gradient_checkpointing_enable()
    return model, tokenizer


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--data-dir', type=Path, default=DEFAULT_DATA_DIR, help='Dataset directory (default: Behrooz v2)')
    parser.add_argument('--val-stage1-run', type=Path, default=RESULTS_DIR / 'stage1/val/fusion.run',
                        help='Saved validation ranking (default: results/stage1/val/fusion.run)')
    parser.add_argument('--config', type=Path, default=HERE / 'config.json')
    parser.add_argument('--output', type=Path, default=RESULTS_DIR / 'stage2_bge/train',
                        help='New training output directory (default: results/stage2_bge/train)')
    parser.add_argument('--device', default='auto')
    parser.add_argument('--check-data', action='store_true',
                        help='Validate and prepare data without loading a model or writing outputs')
    args = parser.parse_args(argv)
    try:
        if args.output.exists():
            raise ValueError('Output already exists; choose a new --output.')
        config = load_config(args.config)
        pairs, _, prepared, labels, data_info = prepare_data(args.data_dir, args.val_stage1_run, config)
        data_info['recipe_sha256'] = digest(args.config)
        print(json.dumps({
            'training_questions': data_info['training']['included_queries'],
            'positive_pairs': len(pairs),
            'excluded_queries': len(data_info['training']['excluded_queries']),
            'excluded_positive_pairs': data_info['training']['excluded_positive_pairs'],
            'validation_questions': len(prepared), 'selection_metric': config['selection_metric'],
        }, indent=2), flush=True)
        if args.check_data:
            return
        device = resolve_device(args.device)
        model, tokenizer = load_training_model(config, device)
        validate = partial(evaluate_validation, prepared=prepared, labels=labels, config=config, device=device)
        result = fit(model, tokenizer, pairs, config, args.output, device, validate, data_info=data_info)
        print(json.dumps(result, indent=2))
    except (OSError, ValueError, RuntimeError, KeyError, ImportError) as error:
        parser.exit(1, f'Training failed: {error}\n')


if __name__ == '__main__':
    main()
