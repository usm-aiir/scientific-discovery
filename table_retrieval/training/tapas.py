"""Research-only TAPAS fine-tuning for reproducing the model comparison."""
import argparse
import json
import os
import subprocess
from types import SimpleNamespace
import math
from pathlib import Path
import random
import numpy as np
import torch
from torch.nn import functional as F
from tqdm.auto import tqdm
from ..data import load_json, load_corpus, load_queries, normalize_table_id, training_examples
from ..retrievers.tapas import checkpoint_fingerprint, select_device, PaperRetriever

class BertAdam(torch.optim.Optimizer):
    def __init__(self, params, lr=1.25e-5, eps=1e-6):
        super().__init__(params, dict(lr=lr, eps=eps, betas=(0.9, 0.999), weight_decay=0.0))

    @torch.no_grad()
    def step(self, closure=None):
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()
        for group in self.param_groups:
            beta1, beta2 = group['betas']
            for parameter in group['params']:
                if parameter.grad is None:
                    continue
                gradient = parameter.grad
                if gradient.is_sparse:
                    raise ValueError('BertAdam requires dense gradients')
                state = self.state[parameter]
                if not state:
                    state['m'] = torch.zeros_like(parameter)
                    state['v'] = torch.zeros_like(parameter)
                m, v = state['m'], state['v']
                m.mul_(beta1).add_(gradient, alpha=1 - beta1)
                v.mul_(beta2).addcmul_(gradient, gradient, value=1 - beta2)
                update = m / (v.sqrt() + group['eps'])
                if group['weight_decay']:
                    update.add_(parameter, alpha=group['weight_decay'])
                parameter.add_(update, alpha=-group['lr'])
        return loss


def load_negatives(run_path, examples, corpus):
    """Highest-ranked nonpositive per training query; unjudged != irrelevant."""
    lookup = {normalize_table_id(uid): uid for uid in corpus}
    positives = {qid: relevant for qid, _, relevant in examples}
    ranked = {}
    for line in Path(run_path).read_text().splitlines():
        if not line.strip():
            continue
        qid, marker, uid, rank, score, _ = line.split()
        if qid not in positives:
            raise ValueError(f'Negative run contains a non-training query: {qid}')
        uid = lookup.get(normalize_table_id(uid))
        if marker != 'Q0' or int(rank) < 1 or not math.isfinite(float(score)) or uid is None:
            raise ValueError(f'Invalid negative-mining run line: {line}')
        if uid not in positives[qid]:
            ranked.setdefault(qid, []).append((int(rank), uid))
    missing = set(positives) - set(ranked)
    if missing:
        raise ValueError(f'No eligible mined negative for {len(missing)} training queries')
    return {qid: min(hits)[1] for qid, hits in ranked.items()}


def paper_batch(examples, rng, negatives=None):
    targets = [rng.choice(sorted(relevant)) for _, _, relevant in examples]
    candidates = targets + ([negatives[qid] for qid, _, _ in examples] if negatives else [])
    # Keep the sampled gold target; mask duplicate and other known positives.
    # This extends the release's duplicate-table mask to multi-positive qrels.
    mask = torch.tensor([[uid in relevant for uid in candidates]
                         for _, _, relevant in examples], dtype=torch.bool)
    mask[torch.arange(len(examples)), torch.arange(len(examples))] = False
    # Duplicate question texts should not create false negatives either.
    for i, (_, text, _) in enumerate(examples):
        for j, (_, other, _) in enumerate(examples):
            if i != j and text == other:
                mask[i, j] = True
                if negatives:
                    mask[i, len(examples) + j] = True
    return candidates, mask


def paper_loss(queries, tables, mask):
    # Only transfer the small projected vectors. Autograd carries gradients
    # back to the table GPU; parameters and Adam state stay on their own GPU.
    logits = queries @ tables.to(queries.device).T
    mask = mask.to(queries.device)
    if mask.shape != logits.shape:
        raise ValueError('Invalid negative mask shape')
    if mask.diagonal().any():
        raise ValueError('Gold targets cannot be masked')
    return F.cross_entropy(logits.masked_fill(mask, -1e9),
                           torch.arange(len(queries), device=queries.device))


@torch.inference_mode()
def validation_recall(model, examples, corpus, batch_size):
    """Exact Recall@10 on the union of available validation-positive tables."""
    model.eval()
    ids = sorted(set().union(*(relevant for _, _, relevant in examples)))
    vectors = torch.cat([model.encode_tables([corpus[uid] for uid in ids[start:start + batch_size]]).cpu()
                         for start in range(0, len(ids), batch_size)])
    recalls = []
    for start in range(0, len(examples), batch_size):
        batch = examples[start:start + batch_size]
        queries = model.encode_queries([text for _, text, _ in batch]).cpu()
        scores = queries @ vectors.T
        if not torch.isfinite(scores).all():
            raise ValueError('Non-finite validation scores')
        hits = torch.argsort(scores, descending=True, stable=True)[:, :10]
        for (_, _, relevant), indices in zip(batch, hits.tolist()):
            recalls.append(len(relevant.intersection(ids[i] for i in indices)) / len(relevant))
    return float(np.mean(recalls))


def train(args):
    if min(args.batch_size, args.eval_batch_size, args.max_steps, args.eval_every, args.patience) < 1:
        raise ValueError('batch sizes, steps, evaluation interval and patience must be positive')
    if args.batch_size < 2 and args.negative_run is None:
        raise ValueError('In-batch training needs batch-size >= 2')
    if not 0 <= args.warmup_ratio < 1 or not 0 <= args.dropout < 1:
        raise ValueError('warmup-ratio and dropout must be in [0, 1)')
    if not math.isfinite(args.learning_rate) or args.learning_rate <= 0:
        raise ValueError('learning-rate must be finite and positive')
    if args.output.exists() and (not args.output.is_dir() or any(args.output.iterdir())):
        raise ValueError('output must be new or empty')
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    rng = random.Random(args.seed)
    corpus = load_corpus(args.corpus)
    examples, train_report = training_examples(load_queries(args.queries), args.qrels, corpus)
    validation, val_report = training_examples(load_queries(args.val_queries), args.val_qrels, corpus)
    if {qid for qid, _, _ in examples} & {qid for qid, _, _ in validation}:
        raise ValueError('Training and validation query IDs overlap')
    if len(examples) < args.batch_size:
        raise ValueError('batch-size exceeds number of training examples')
    negatives = load_negatives(args.negative_run, examples, corpus) if args.negative_run else None
    query_device = select_device(args.device)
    table_device = select_device(args.table_device) if args.table_device else query_device
    model = PaperRetriever.load(args.model).place_encoders(query_device, table_device)
    print(f'Query encoder: {query_device}; table encoder: {table_device}', flush=True)
    for encoder in (model.query_encoder, model.table_encoder):
        encoder.config.hidden_dropout_prob = args.dropout
        encoder.config.attention_probs_dropout_prob = args.dropout
        for module in encoder.modules():
            if isinstance(module, torch.nn.Dropout):
                module.p = args.dropout
        if args.gradient_checkpointing:
            encoder.gradient_checkpointing_enable()
    decay, no_decay = [], []
    for name, parameter in model.named_parameters():
        (no_decay if 'bias' in name or 'LayerNorm' in name else decay).append(parameter)
    # Google's BERT Adam has no bias correction and epsilon=1e-6.
    optimizer = BertAdam([{'params': decay, 'weight_decay': 0.01},
                       {'params': no_decay, 'weight_decay': 0.0}],
                      lr=args.learning_rate, eps=1e-6)
    warmup = int(args.max_steps * args.warmup_ratio)
    def schedule(step):
        return step / warmup if step < warmup else max(0.0, 1 - step / args.max_steps)
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, schedule)
    settings = {key: str(value) if isinstance(value, Path) else value for key, value in vars(args).items()}
    settings.update(source_fingerprint=checkpoint_fingerprint(args.model),
                    query_device=str(query_device), table_device=str(table_device),
                    train_data_report=train_report, validation_data_report=val_report,
                    validation_scope='available validation-positive tables only',
                    negative_filter='known qrel positives, not reference-answer filtering',
                    objective='single sampled gold, raw dot-product cross entropy; known positives masked')
    args.output.mkdir(parents=True, exist_ok=True)
    history = []
    best, stale, step = -1.0, 0, 0
    progress = tqdm(total=args.max_steps, unit='step')
    while step < args.max_steps and stale < args.patience:
        rng.shuffle(examples)
        for start in range(0, len(examples), args.batch_size):
            batch = examples[start:start + args.batch_size]
            if len(batch) < 2 and negatives is None:
                continue
            model.train()
            candidates, mask = paper_batch(batch, rng, negatives)
            # Batches with no eligible negatives provide no retrieval signal.
            if not ((~mask).sum(dim=1) > 1).any():
                raise ValueError('Batch has no eligible negatives; increase batch size')
            optimizer.zero_grad(set_to_none=True)
            queries = model.encode_queries([text for _, text, _ in batch])
            tables = model.encode_tables([corpus[uid] for uid in candidates])
            loss = paper_loss(queries, tables, mask.to(queries.device))
            if not torch.isfinite(loss):
                raise ValueError('Non-finite loss')
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            optimizer.zero_grad(set_to_none=True)
            scheduler.step()
            step += 1
            progress.update(1)
            progress.set_postfix(loss=float(loss.detach()), refresh=False)
            if step % args.eval_every == 0 or step == args.max_steps:
                recall = validation_recall(model, validation, corpus, args.eval_batch_size)
                history.append({'step': step, 'loss': float(loss.detach()), 'dev_recall@10': recall})
                if recall > best:
                    best, stale = recall, 0
                    model.save(args.output / 'best', dict(settings, best_step=step, best_dev_recall=best))
                else:
                    stale += 1
                (args.output / 'training.json').write_text(json.dumps(dict(settings, history=history), indent=2) + '\n')
            if step >= args.max_steps or stale >= args.patience:
                break
    progress.close()
    print(f'Best development Recall@10: {best:.4f}; checkpoint: {args.output / "best"}')


ROOT = Path(__file__).resolve().parents[2]


def train_from_settings(settings):
    configured_device = settings['device']
    if configured_device == 'auto':
        configured_device = ('cuda:1' if torch.cuda.device_count() >= 2 else
                             'cuda:0' if torch.cuda.is_available() else 'cpu')
    config = dict(settings['tapas']['training'])
    data = Path(settings['data'])
    two_gpus = settings['device'] == 'auto' and torch.cuda.device_count() >= 2
    config.update(corpus=data / 'Corpus.json', queries=data / 'Train.json', qrels=data / 'Train_table_qrels.tsv',
                  val_queries=data / 'Val.json', val_qrels=data / 'Val_table_qrels.tsv',
                  device='cuda:0' if two_gpus else configured_device,
                  table_device='cuda:1' if two_gpus else configured_device)
    for key in ['model', 'output', 'negative_run']:
        config[key] = Path(config[key]) if config[key] else None
    for key in ['corpus', 'queries', 'qrels', 'val_queries', 'val_qrels']:
        if not config[key].is_file():
            raise FileNotFoundError(config[key])
    if config['output'].exists() and any(config['output'].iterdir()):
        raise ValueError('Training output must be new or empty')
    if not (config['model'] / 'retriever.json').exists():
        if config['model'] != Path('results/dtr_paper/pretrained'):
            raise FileNotFoundError(f"Initial TAPAS checkpoint missing: {config['model']}")
        subprocess.run(['bash', str(ROOT / 'bin/prepare_tapas')], check=True)
    train(SimpleNamespace(**config))


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--settings', type=Path,
                        default=Path(__file__).resolve().parents[1] / 'settings.json')
    parser.add_argument('--experiment', help='Save a separate research checkpoint under outputs/checkpoints')
    args = parser.parse_args(argv)
    try:
        settings = load_json(args.settings.resolve())
        if args.experiment:
            if Path(args.experiment).name != args.experiment or args.experiment in ('.', '..'):
                raise ValueError('Experiment must be a simple directory name')
            settings['tapas']['training']['output'] = str(
                Path(settings.get('output_root', 'outputs')) / 'checkpoints' / args.experiment)
        os.chdir(ROOT)
        os.environ.setdefault('OMP_NUM_THREADS', '4')
        os.environ.setdefault('TOKENIZERS_PARALLELISM', 'false')
        train_from_settings(settings)
    except (ValueError, FileNotFoundError, subprocess.CalledProcessError) as error:
        parser.exit(1, f'{error}\n')


if __name__ == '__main__':
    main()
