# Fine-tunes BGE from the pinned upstream revision and selects by validation nDCG.
import argparse
import json
import random
from pathlib import Path

from table_retrieval.data import digest, load_json, write_json
from table_retrieval.stage1.artifacts import resolve_device
from .dataset import DATA, HERE, read_data, training_cells, training_pairs
from .metrics import validation_ndcg


# Training and inference both use normalized CLS vectors and cosine similarity.
def encode(model, tokenizer, texts, device, max_length):
    import torch
    tokens = tokenizer(texts, padding=True, truncation=True, max_length=max_length, return_tensors='pt')
    output = model(**{key: value.to(device) for key, value in tokens.items()})
    return torch.nn.functional.normalize(output.last_hidden_state[:, 0].float(), dim=-1)


# Each positive competes with four explicit negatives from the fixed training split.
def contrastive_loss(model, tokenizer, batch, config, device):
    import torch
    queries = encode(model, tokenizer, [item['query'] for item in batch], device, config['max_length'])
    texts = [text for item in batch for text in [item['positive'], *item['negative_texts']]]
    candidates = encode(model, tokenizer, texts, device, config['max_length'])
    candidates = candidates.reshape(len(batch), 1 + config['negatives_per_positive'], -1)
    logits = torch.einsum('bd,bcd->bc', queries, candidates) / config['temperature']
    return torch.nn.functional.cross_entropy(logits, torch.zeros(len(batch), dtype=torch.long, device=device))


# Validation receives each annotated table directly, matching historical epoch selection.
def evaluate_validation(model, tokenizer, examples, config, device):
    import torch
    model.eval()
    rankings = []
    with torch.inference_mode():
        for example in examples:
            cells = training_cells(example, config['training_representation'])
            query = encode(model, tokenizer, [example['question']], device, config['max_length'])[0].cpu()
            vectors = []
            size = config['encoding_batch_size']
            for start in range(0, len(cells), size):
                vectors.append(encode(model, tokenizer, [cell['text'] for cell in cells[start:start + size]],
                                      device, config['max_length']).cpu())
            scores = torch.cat(vectors) @ query
            order = torch.argsort(scores, descending=True, stable=True).tolist()
            rankings.append(([cells[index]['cell_id'] for index in order],
                             {cell['cell_id'] for cell in example['evidence_cells']}))
    return validation_ndcg(rankings)


# Save one checkpoint per improving epoch; never delete or overwrite an earlier run.
def fit(model, tokenizer, examples, validation, config, output, device):
    import torch
    import transformers
    if output.exists():
        raise ValueError(f'Output already exists; choose a new --output: {output}')
    random.seed(config['seed'])
    torch.manual_seed(config['seed'])
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(config['seed'])
    pairs, counts = training_pairs(examples, config)
    loader = torch.utils.data.DataLoader(
        pairs, batch_size=config['batch_size'], shuffle=True,
        generator=torch.Generator().manual_seed(config['seed']), collate_fn=list)
    optimizer = torch.optim.AdamW(model.parameters(), lr=config['learning_rate'])
    output.mkdir(parents=True)
    history, best = [], -1.0
    accumulation = config['gradient_accumulation']
    checkpoint = None
    for epoch in range(1, config['epochs'] + 1):
        model.train()
        optimizer.zero_grad(set_to_none=True)
        total_loss = 0.0
        for step, batch in enumerate(loader, 1):
            loss = contrastive_loss(model, tokenizer, batch, config, device)
            remainder = len(loader) % accumulation
            divisor = remainder if remainder and step > len(loader) - remainder else accumulation
            (loss / divisor).backward()
            if step % accumulation == 0 or step == len(loader):
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()
                optimizer.zero_grad(set_to_none=True)
            total_loss += loss.item()
            if step % 50 == 0 or step == len(loader):
                print(f'epoch {epoch} step {step}/{len(loader)} loss={loss.item():.6f}', flush=True)
        ndcg = evaluate_validation(model, tokenizer, validation, config, device)
        history.append({'epoch': epoch, 'training_loss': total_loss / len(loader), 'validation_ndcg': ndcg})
        print(f'epoch {epoch}: validation nDCG={ndcg:.6f}', flush=True)
        if ndcg > best:
            best = ndcg
            checkpoint = output / f'epoch_{epoch}'
            model.save_pretrained(checkpoint)
            tokenizer.save_pretrained(checkpoint)
    result = {'config': config, 'best_validation_ndcg': best,
              'checkpoint': checkpoint.name, 'epochs': history,
              'training_examples': len(examples), 'positive_pairs': len(pairs),
              'negative_assignments': counts, 'device': device,
              'torch_version': torch.__version__, 'dtype': str(next(model.parameters()).dtype),
              'transformers_version': transformers.__version__,
              'data_manifest_sha256': digest(DATA / 'manifest.json'),
              'checkpoint_sha256': {path.name: digest(path) for path in checkpoint.iterdir() if path.is_file()}}
    write_json(output / 'training.json', result)
    return result


# The default recipe reproduces the checkpoint used for the reported 31.53% result.
def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__ or 'Train the Stage 2 BGE checkpoint.')
    parser.add_argument('--config', type=Path, default=HERE / 'config.json')
    parser.add_argument('--output', type=Path, default=HERE.parents[2] / 'results/stage2_bge/train')
    parser.add_argument('--device', default='auto')
    args = parser.parse_args(argv)
    from .dataset import validate_data
    validate_data()
    config = load_json(args.config)
    if args.output.exists():
        parser.error('Output already exists; choose a new --output.')
    import torch
    from transformers import AutoModel, AutoTokenizer
    device = resolve_device(args.device)
    dtype = torch.bfloat16 if device.startswith('cuda') and torch.cuda.is_bf16_supported() else torch.float32
    options = {'revision': config['revision']}
    tokenizer = AutoTokenizer.from_pretrained(config['model'], **options)
    model = AutoModel.from_pretrained(config['model'], torch_dtype=dtype,
                                     attn_implementation='eager', **options).to(device)
    if device.startswith('cuda'):
        model.gradient_checkpointing_enable()
    result = fit(model, tokenizer, read_data(config['train_split'])['annotations'],
                 read_data('val.json')['annotations'], config, args.output, device)
    print(json.dumps(result, indent=2))


if __name__ == '__main__':
    main()
