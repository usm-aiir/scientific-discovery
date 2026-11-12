"""Load, train and evaluate the ModernBERT classifier."""

import json
from pathlib import Path
from types import SimpleNamespace

import torch
from tqdm import tqdm

from .data import (ROUTES, ROUTE_LABELS, checkpoint_task,
                   configure_classifier, routing_metrics, target_mask)

ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT / 'data'
MODEL_DIR = ROOT / 'model'
RESULTS_DIR = ROOT / 'results'


def write_json(path, value):
    Path(path).write_text(json.dumps(value, ensure_ascii=False, indent=2) + '\n', encoding='utf-8')


def checkpoint_file_task(checkpoint):
    return checkpoint_task(SimpleNamespace(**json.loads((Path(checkpoint) / 'config.json').read_text())))


def latest_checkpoint():
    """Find the newest completed checkpoint without relying on results folders."""
    candidates = [p.parent for p in sorted(MODEL_DIR.glob('run_*/config.json'), reverse=True)]
    candidates.append(MODEL_DIR / 'modernbert')
    for candidate in candidates:
        try:
            info = json.loads((candidate / 'training.json').read_text())
            if info.get('status') != 'complete':
                continue
            required = ['config.json', 'tokenizer.json', 'tokenizer_config.json']
            if not all((candidate / name).is_file() for name in required):
                continue
            if not any((candidate / name).is_file() for name in ('model.safetensors', 'model.safetensors.index.json')):
                continue
            checkpoint_file_task(candidate)
            return candidate
        except (OSError, ValueError, TypeError, AttributeError):
            continue
    raise FileNotFoundError('No completed ModernBERT checkpoint. Run python -m modernbert_classifier.train first.')


def create_classifier(model_name, revision, max_length):
    """Load a pretrained encoder with a seven-route classification head."""
    from transformers import AutoTokenizer, ModernBertConfig, ModernBertForSequenceClassification

    config = ModernBertConfig.from_pretrained(model_name, revision=revision)
    # Resolve main once, then use the same commit for tokenizer and weights.
    revision = getattr(config, "_commit_hash", None) or revision
    tokenizer = AutoTokenizer.from_pretrained(model_name, revision=revision)
    tokenizer.padding_side = "right"
    if tokenizer.pad_token_id is None or tokenizer.cls_token_id is None:
        raise ValueError("ModernBERT needs a tokenizer with padding and CLS tokens.")
    configure_classifier(config, max_length)
    config.classifier_pooling = "cls"
    config.pad_token_id = tokenizer.pad_token_id
    model, loading = ModernBertForSequenceClassification.from_pretrained(
        model_name, revision=revision, config=config, dtype=torch.float32,
        attn_implementation="sdpa", output_loading_info=True,
    )
    # The classifier may be new. Every encoder weight must come from pretraining.
    missing_encoder = [key for key in loading["missing_keys"] if key.startswith("model.")]
    if missing_encoder or loading.get("mismatched_keys") or loading.get("error_msgs"):
        raise ValueError(f"Pretrained encoder weights did not load correctly: {loading}")
    return model, tokenizer


def load_checkpoint(checkpoint, device):
    """Load a saved classifier and verify its checkpoint contract."""
    from transformers import AutoTokenizer, ModernBertForSequenceClassification

    model, loading = ModernBertForSequenceClassification.from_pretrained(
        checkpoint, dtype=torch.float32, attn_implementation="sdpa", output_loading_info=True,
    )
    checkpoint_task(model.config)
    if loading["missing_keys"] or loading.get("mismatched_keys") or loading.get("error_msgs"):
        raise ValueError(f"Incomplete ModernBERT checkpoint: {loading}")
    tokenizer = AutoTokenizer.from_pretrained(checkpoint)
    return model.to(device).eval(), tokenizer


def encode_queries(tokenizer, queries, max_length, device):
    """Tokenize question text without including metadata."""
    tokens = tokenizer(
        queries, padding=True, truncation=True, max_length=max_length,
        return_tensors="pt", return_token_type_ids=False,
    )
    return {key: tokens[key].to(device) for key in ("input_ids", "attention_mask")}


def mixed_precision(device):
    """Use BF16 on supported GPUs while keeping weights in FP32."""
    enabled = device.type == "cuda" and torch.cuda.is_bf16_supported()
    return torch.autocast(device_type=device.type, dtype=torch.bfloat16, enabled=enabled)


def acceptable_route_loss(logits, mask, reduction='mean'):
    """Minimize the negative log probability of choosing any acceptable route."""
    if (logits.ndim != 2 or logits.shape[1] != len(ROUTES)
            or mask.shape != logits.shape or mask.dtype != torch.bool
            or not mask.any(dim=1).all()):
        raise ValueError('Expected [batch,7] logits and a Boolean mask with a target for every example')
    if not torch.isfinite(logits).all():
        raise ValueError('Route logits must be finite')
    # Sum probability across alternatives; unlisted routes are not proven negatives.
    losses = torch.logsumexp(logits, dim=-1) - torch.logsumexp(logits.masked_fill(~mask, -torch.inf), dim=-1)
    if reduction == 'none':
        return losses
    if reduction == 'sum':
        return losses.sum()
    if reduction == 'mean':
        return losses.mean()
    raise ValueError('Unknown loss reduction')


def train_epoch(model, tokenizer, loader, optimizer, scheduler, accumulation_steps, device):
    """Train one epoch, averaging gradients over each accumulation window."""
    model.train()
    checkpoint_task(model.config)
    optimizer.zero_grad(set_to_none=True)
    loss_sum, example_count, window_count = 0.0, 0, 0
    progress = tqdm(loader, desc="Train")
    for step, batch in enumerate(progress, start=1):
        inputs = encode_queries(tokenizer, batch["query"], model.config.router_max_length, device)
        labels = batch['acceptable_mask'].to(device)
        with mixed_precision(device):
            logits = model(**inputs).logits.float()
            # Summing here lets us average by the actual number of examples in
            # each accumulation window, including the final incomplete window.
            loss = acceptable_route_loss(logits, labels, reduction='sum')
        if not torch.isfinite(loss):
            raise RuntimeError("Training loss is not finite; stop and inspect the run.")
        loss.backward()
        loss_sum += loss.item()
        example_count += len(labels)
        window_count += len(labels)
        if step % accumulation_steps == 0 or step == len(loader):
            for parameter in model.parameters():
                if parameter.grad is not None:
                    parameter.grad.div_(window_count)
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0, error_if_nonfinite=True)
            optimizer.step()
            scheduler.step()
            optimizer.zero_grad(set_to_none=True)
            window_count = 0
        progress.set_postfix(loss=f"{loss_sum / example_count:.4f}")
    return loss_sum / example_count


@torch.inference_mode()
def evaluate(model, tokenizer, records, batch_size, device):
    model.eval()
    checkpoint_task(model.config)
    predictions, outputs = [], []
    loss_sum = 0.0
    for start in tqdm(range(0, len(records), batch_size), desc='Evaluate'):
        batch = records[start:start + batch_size]
        inputs = encode_queries(tokenizer, [r['query'] for r in batch], model.config.router_max_length, device)
        masks = [target_mask(record) for record in batch]
        labels = torch.tensor(masks, dtype=torch.bool, device=device)
        with mixed_precision(device):
            logits = model(**inputs).logits.float()
            loss_sum += acceptable_route_loss(logits, labels, reduction='sum').item()
        chosen = logits.argmax(-1).cpu().tolist()
        scores = logits.softmax(-1).cpu().tolist()
        predictions.extend(chosen)
        for record, index, probabilities, mask in zip(batch, chosen, scores, masks):
            outputs.append({**record, 'predicted_label': ROUTE_LABELS[index],
                            'predicted_route': ROUTES[index].copy(),
                            'probabilities': dict(zip(ROUTE_LABELS, probabilities)),
                            'known_acceptable_route_match': mask[index]})
    metrics = routing_metrics(records, predictions)
    metrics['loss'] = loss_sum / len(records)
    return metrics, outputs
