"""
TAPAS-based table retrieval, indexing, and training.

Provides model loading, table/query encoding, corpus indexing, retrieval,
fine-tuning, negative sampling, and Recall@10 validation for TAPAS retrievers.
"""
import argparse
from collections import defaultdict
from dataclasses import asdict, dataclass
import hashlib
import json
import math
import os
from pathlib import Path
import random
import subprocess
from types import SimpleNamespace
import warnings

import numpy as np
import pandas as pd
import torch
from torch import nn
from torch.nn import functional as F
from tqdm.auto import tqdm
from transformers import BertModel, BertTokenizer, TapasModel, TapasTokenizer

from ...data import (check_id, digest, load_corpus, load_json, load_queries,
                    normalize_table_id)
from ..retrievers.ranking import rank_tables

@dataclass
class InputLimits:
    """Deterministic limits shared by training and corpus indexing."""

    query_length: int = 128
    table_length: int = 512
    context_tokens: int = 64
    cell_tokens: int = 16
    max_rows: int = 64
    max_columns: int = 32

    def __post_init__(self):
        if any(value < 1 for value in asdict(self).values()):
            raise ValueError("Input limits must be positive.")
        if self.query_length < 3 or self.table_length < self.context_tokens + 5:
            raise ValueError("Sequence lengths must leave room for special tokens and cells.")


def clip_text(tokenizer, text, budget):
    """Bound WordPiece length before TAPAS packs rows into its token budget."""
    text = "" if text is None else str(text)
    tokens = tokenizer.tokenize(text)
    return text if len(tokens) <= budget else tokenizer.convert_tokens_to_string(tokens[:budget])


def table_inputs(table, tokenizer, limits):
    """Keep a rectangular table, with row/column structure available to TAPAS.

    The first row is the provisional header, as in the BM25 baseline. Short
    rows are padded; extra cells get an empty header. Large tables retain the
    leading rows/columns. TAPAS may then drop trailing rows to fit the budget.
    """
    rows = table.get("rows") or []
    context = " ".join(str(table.get(key) or "") for key in
                       ("paper_title", "caption", "sub_caption", "reference_text"))
    context = clip_text(tokenizer, context, limits.context_tokens)
    width = max((len(row) for row in rows), default=1)
    width = min(max(1, width), limits.max_columns,
                (limits.table_length - limits.context_tokens - 3) // 2)
    # Reserve enough space for headers and at least one data row.
    cell_budget = min(limits.cell_tokens,
                      (limits.table_length - limits.context_tokens - 3) // (2 * width))

    def cells(row):
        return [clip_text(tokenizer, row[i] if i < len(row) else "", cell_budget)
                for i in range(width)]

    headers = cells(rows[0] if rows else [])
    body = [cells(row) for row in rows[1:limits.max_rows + 1]]
    frame = pd.DataFrame(body, columns=headers, dtype=str)
    # Transformers 4.44 uses positional Series indexing internally. Suppress
    # only that known pandas deprecation while tokenizing, not other warnings.
    with warnings.catch_warnings():
        warnings.filterwarnings(
            "ignore",
            message=r"Series\.__getitem__ treating keys as positions is deprecated\.",
            category=FutureWarning,
            module=r"transformers\.models\.tapas\.tokenization_tapas$",
        )
        return dict(tokenizer(table=frame, queries=context, max_length=limits.table_length,
                              truncation="drop_rows_to_fit", padding=False))


def table_features(table, tokenizer, config, table_length=512):
    rows = table.get('rows') or []
    max_cols = config.type_vocab_sizes[1] - 1
    max_rows = config.type_vocab_sizes[2] - 1
    width = min(max_cols, max(1, max(map(len, rows), default=1)))
    height = min(max_rows, max(0, len(rows) - 1))
    title = str(table.get('paper_title') or '')
    title_tokens = tokenizer.tokenize(title)
    if len(title_tokens) > table_length - 4:
        # Some extracted titles contain much more than a title. Retain a prefix
        # and leave room for table evidence, without changing the source corpus.
        warnings.warn('Oversized paper title truncated for encoding; source corpus unchanged.',
                      UserWarning)
        title_tokens = title_tokens[:min(128, table_length // 2)]
        title = tokenizer.convert_tokens_to_string(title_tokens)
        while len(tokenizer.tokenize(title)) > min(128, table_length // 2):
            title_tokens.pop()
            title = tokenizer.convert_tokens_to_string(title_tokens)
    while True:
        def cells(row):
            # Preserve zero-valued numeric cells too.
            return [str(row[i]) if i < len(row) and row[i] is not None else '' for i in range(width)]
        frame = pd.DataFrame([cells(row) for row in rows[1:height + 1]],
                             columns=cells(rows[0] if rows else []), dtype=str)
        # Compute the budget before asking HF to encode. An untruncated
        # encode raises on overflow and would otherwise skip cell trimming.
        tokenized = tokenizer._tokenize_table(frame)
        budget = tokenizer._get_max_num_tokens(
            tokenizer.tokenize(title), tokenized, width, height, table_length)
        if budget is not None:
            with warnings.catch_warnings():
                warnings.filterwarnings('ignore', category=FutureWarning,
                                        module=r'transformers\.models\.tapas\.tokenization_tapas$')
                encoded = dict(tokenizer(table=frame, queries=title,
                    max_length=table_length, truncation='drop_rows_to_fit', padding=False))
            break
        if width >= height and width > 1:
            width -= 1
        elif height > 0:
            height -= 1
        else:
            raise ValueError('Cannot fit table into sequence')
    if len(encoded['input_ids']) > table_length:
        raise ValueError('Table packing exceeded the sequence budget')
    for channel in encoded['token_type_ids']:
        channel[6] = 0
    return encoded


def training_examples(queries, qrels_path, corpus, min_grade=1):
    """Resolve qrels to corpus IDs; never use explanations as encoder inputs."""
    lookup = {}
    for uid in corpus:
        normalized = normalize_table_id(check_id(uid))
        if normalized in lookup:
            raise ValueError(f"Ambiguous corpus table ID: {normalized}")
        lookup[normalized] = uid
    positives = defaultdict(set)
    query_ids = [check_id(query["query_id"]) for query in queries]
    if len(set(query_ids)) != len(query_ids):
        raise ValueError("Duplicate training query IDs.")
    allowed = set(query_ids)
    missing = set()
    for line in Path(qrels_path).read_text().splitlines():
        if not line.strip():
            continue
        qid, _, uid, grade = line.split()
        if qid not in allowed or int(grade) < min_grade:
            continue
        normalized = normalize_table_id(uid)
        if normalized in lookup:
            positives[qid].add(lookup[normalized])
        else:
            missing.add(uid)
    examples = []
    skipped = []
    for qid, query in zip(query_ids, queries):
        if not isinstance(query["query"], str) or not query["query"].strip():
            raise ValueError(f"Training query {qid} must contain nonempty text.")
        if not positives[qid]:
            skipped.append(qid)
        elif len(positives[qid]) == len(corpus):
            raise ValueError(f"Query {qid} has no negative tables in the corpus.")
        else:
            examples.append((qid, query["query"], positives[qid]))
    if not examples:
        raise ValueError("No training queries have positive judgments in the corpus.")
    report = {"missing_judged_table_ids": sorted(missing), "skipped_query_ids": skipped}
    if missing or skipped:
        warnings.warn(f"Training data: {len(missing)} missing judged table IDs; "
                      f"{len(skipped)} queries skipped. Details saved with the checkpoint.")
    return examples, report


class DenseTableRetriever(nn.Module):
    """Independent encoders trained into a shared cosine-similarity space.

    Table vectors depend only on table data, never on the retrieval query.
    TAPAS receives all seven structural token-type channels from its tokenizer.
    """

    def __init__(self, query_encoder, table_encoder, query_tokenizer, table_tokenizer,
                 limits=None):
        super().__init__()
        self.query_encoder = query_encoder
        self.table_encoder = table_encoder
        self.query_tokenizer = query_tokenizer
        self.table_tokenizer = table_tokenizer
        self.limits = limits or InputLimits()
        self.dimension = table_encoder.config.hidden_size
        self.similarity = "cosine"
        if query_encoder.config.hidden_size != table_encoder.config.hidden_size:
            raise ValueError("Query and table encoders must have the same hidden size.")
        sizes = table_encoder.config.type_vocab_sizes
        if self.limits.max_columns >= sizes[1] or self.limits.max_rows >= sizes[2]:
            raise ValueError("Row/column limits exceed TAPAS structural embeddings.")
        if (self.limits.query_length > query_encoder.config.max_position_embeddings or
                self.limits.table_length > table_encoder.config.max_position_embeddings):
            raise ValueError("Sequence limits exceed model position embeddings.")

    @classmethod
    def from_pretrained(cls, query_model="bert-base-uncased", table_model="google/tapas-base",
                        limits=None):
        return cls(BertModel.from_pretrained(query_model), TapasModel.from_pretrained(table_model),
                   BertTokenizer.from_pretrained(query_model),
                   TapasTokenizer.from_pretrained(table_model), limits)

    @classmethod
    def load(cls, directory):
        directory = Path(directory)
        settings = json.loads((directory / "retriever.json").read_text())
        return cls.from_pretrained(str(directory / "query"), str(directory / "table"),
                                   InputLimits(**settings["limits"]))

    def save(self, directory, training):
        directory = Path(directory)
        directory.mkdir(parents=True, exist_ok=True)
        self.query_encoder.save_pretrained(directory / "query", safe_serialization=True)
        self.query_tokenizer.save_pretrained(directory / "query")
        self.table_encoder.save_pretrained(directory / "table", safe_serialization=True)
        self.table_tokenizer.save_pretrained(directory / "table")
        (directory / "retriever.json").write_text(json.dumps(
            {"limits": asdict(self.limits), "pooling": "normalized_cls", "training": training},
            indent=2) + "\n")

    def _encode(self, encoder, inputs):
        device = next(encoder.parameters()).device
        inputs = {key: value.to(device) for key, value in inputs.items()}
        output = encoder(**inputs).last_hidden_state[:, 0]
        return F.normalize(output, p=2, dim=-1)

    def encode_queries(self, texts):
        inputs = self.query_tokenizer(texts, padding=True, truncation=True,
                                      max_length=self.limits.query_length, return_tensors="pt")
        return self._encode(self.query_encoder, inputs)

    def encode_tables(self, tables):
        examples = [table_inputs(table, self.table_tokenizer, self.limits) for table in tables]
        inputs = self.table_tokenizer.pad(examples, padding=True, return_tensors="pt")
        return self._encode(self.table_encoder, inputs)


def checkpoint_fingerprint(directory):
    """Bind a corpus index to the exact weights, tokenizers, and input limits."""
    directory = Path(directory)
    digest = hashlib.sha256()
    for path in sorted(directory.rglob("*")):
        if path.is_file():
            digest.update(str(path.relative_to(directory)).encode())
            with path.open("rb") as source:
                for chunk in iter(lambda: source.read(1024 * 1024), b""):
                    digest.update(chunk)
    return digest.hexdigest()


def load_retriever(directory):
    """Dispatch by saved architecture; old baseline checkpoints remain valid."""
    settings = json.loads((Path(directory) / "retriever.json").read_text())
    architecture = settings.get("architecture")
    if architecture == "paper_tapas_dual_encoder":
        return PaperRetriever.load(directory)
    if architecture is not None:
        raise ValueError(f"Unknown retriever architecture: {architecture}")
    return DenseTableRetriever.load(directory)


def select_device(name):
    return "cuda" if name == "auto" and torch.cuda.is_available() else "cpu" if name == "auto" else name


class PaperRetriever(nn.Module):
    """Match the released implementation: tanh pooler, projection, inner product.

    Position indices are absolute, unlike the default Hugging Face TAPAS config.
    Query structural channels are zero. The retrieval title is not an answer
    question, so table numeric-relation features are zero as in the source code.
    """

    def __init__(self, query_encoder, table_encoder, query_tokenizer, table_tokenizer,
                 dimension=256, query_length=128, table_length=512):
        super().__init__()
        self.query_encoder = query_encoder
        self.table_encoder = table_encoder
        self.query_tokenizer = query_tokenizer
        self.table_tokenizer = table_tokenizer
        self.query_projection = nn.Linear(query_encoder.config.hidden_size, dimension, bias=False)
        self.table_projection = nn.Linear(table_encoder.config.hidden_size, dimension, bias=False)
        self.dimension = dimension
        self.query_length = query_length
        self.table_length = table_length
        self.similarity = 'inner_product'

    @classmethod
    def load(cls, directory):
        directory = Path(directory)
        settings = json.loads((directory / 'retriever.json').read_text())
        if settings['architecture'] != 'paper_tapas_dual_encoder':
            raise ValueError('Expected a converted paper retriever')
        model = cls(TapasModel.from_pretrained(directory / 'query'),
                    TapasModel.from_pretrained(directory / 'table'),
                    BertTokenizer.from_pretrained(directory / 'query'),
                    TapasTokenizer.from_pretrained(directory / 'table'),
                    settings['dimension'], settings['query_length'], settings['table_length'])
        from safetensors.torch import load_file
        projections = load_file(str(directory / 'projections.safetensors'))
        model.query_projection.load_state_dict({'weight': projections['query']})
        model.table_projection.load_state_dict({'weight': projections['table']})
        return model

    def save(self, directory, training):
        from safetensors.torch import save_file
        directory = Path(directory)
        directory.mkdir(parents=True, exist_ok=True)
        for name in ('query', 'table'):
            getattr(self, f'{name}_encoder').save_pretrained(directory / name)
            getattr(self, f'{name}_tokenizer').save_pretrained(directory / name)
        save_file({'query': self.query_projection.weight.detach().cpu().contiguous(),
                   'table': self.table_projection.weight.detach().cpu().contiguous()},
                  str(directory / 'projections.safetensors'))
        (directory / 'retriever.json').write_text(json.dumps({
            'architecture': 'paper_tapas_dual_encoder', 'dimension': self.dimension,
            'query_length': self.query_length, 'table_length': self.table_length,
            'similarity': self.similarity, 'pooling': 'tanh_pooler_then_linear_projection',
            'representation': 'paper_title + rectangularized table; dynamic cell trimming',
            'oversized_title_policy': 'if over table_length-4 tokens, retain at most min(128, table_length//2)',
            'training': training}, indent=2) + '\n')

    def query_inputs(self, texts):
        # The release serializes CLS/text/SEP, then slices to max_query_length.
        # In particular it does not force a final SEP on a truncated query.
        examples = [self.query_tokenizer(text, truncation=False) for text in texts]
        examples = [{key: value[:self.query_length] for key, value in example.items()}
                    for example in examples]
        inputs = dict(self.query_tokenizer.pad(examples, padding=True, return_tensors='pt'))
        inputs['token_type_ids'] = torch.zeros((*inputs['input_ids'].shape, 7), dtype=torch.long)
        return inputs

    def table_inputs(self, table):
        return table_features(table, self.table_tokenizer, self.table_encoder.config, self.table_length)

    def forward_inputs(self, inputs, side):
        device = next(getattr(self, f'{side}_encoder').parameters()).device
        inputs = {k: v.to(device) for k, v in inputs.items()}
        output = getattr(self, f'{side}_encoder')(**inputs).pooler_output
        return getattr(self, f'{side}_projection')(output).float()

    def place_encoders(self, query_device, table_device=None):
        """Place each tower and its projection together; saved weights stay portable."""
        for side, device in [('query', query_device), ('table', table_device or query_device)]:
            getattr(self, f'{side}_encoder').to(device)
            getattr(self, f'{side}_projection').to(device)
        return self

    def encode_queries(self, texts):
        return self.forward_inputs(self.query_inputs(texts), 'query')

    def encode_tables(self, tables):
        examples = [self.table_inputs(table) for table in tables]
        return self.forward_inputs(self.table_tokenizer.pad(examples, padding=True, return_tensors='pt'), 'table')

def index_tapas(args):
    if args.batch_size < 1:
        raise ValueError("batch-size must be positive")
    if args.output.exists() and (not args.output.is_dir() or any(args.output.iterdir())):
        raise ValueError("output must be a new or empty directory")
    corpus = load_corpus(args.corpus)
    table_ids = sorted(check_id(uid) for uid in corpus)
    device = select_device(args.device)
    model = load_retriever(args.model).to(device).eval()
    fingerprint = checkpoint_fingerprint(args.model)
    args.output.mkdir(parents=True, exist_ok=True)
    destination = getattr(args, 'embeddings', args.output / 'embeddings.npy')
    if destination.exists():
        raise ValueError('Embeddings already exist; choose a new experiment')
    destination.parent.mkdir(parents=True, exist_ok=True)
    if destination != args.output / 'embeddings.npy':
        (args.output / 'embeddings.npy').symlink_to(destination.resolve())
    embeddings = np.lib.format.open_memmap(destination, mode="w+",
        dtype=np.float32, shape=(len(table_ids), model.dimension))
    with torch.inference_mode():
        for start in range(0, len(table_ids), args.batch_size):
            ids = table_ids[start:start + args.batch_size]
            vectors = model.encode_tables([corpus[uid] for uid in ids]).cpu().numpy()
            if not np.isfinite(vectors).all():
                raise ValueError(f"Non-finite embeddings for batch starting at {ids[0]}.")
            embeddings[start:start + len(ids)] = vectors
            if start == 0 or start // args.batch_size % 25 == 0:
                print(f"Encoded {start + len(ids):,}/{len(table_ids):,} tables.", flush=True)
    embeddings.flush()
    (args.output / "table_ids.json").write_text(json.dumps(table_ids) + "\n")
    # Written last: an interrupted index is never accepted as complete.
    (args.output / "index.json").write_text(json.dumps({
        "model_fingerprint": fingerprint, "corpus": str(args.corpus),
        "corpus_sha256": digest(args.corpus),
        "table_count": len(table_ids), "dimension": embeddings.shape[1],
        "similarity": model.similarity, "model": str(args.model)}, indent=2) + "\n")
    print(f"Indexed {len(table_ids):,} tables in {args.output}.")


def load_search_index(index, checkpoint, device):
    """Load a reusable TAPAS resource without requiring saved queries."""
    metadata = load_json(index / "index.json")
    if checkpoint_fingerprint(checkpoint) != metadata["model_fingerprint"]:
        raise ValueError("Index and model do not match. Rebuild the index with this checkpoint.")
    table_ids = load_json(index / "table_ids.json")
    vectors = np.load(index / "embeddings.npy", mmap_mode="r", allow_pickle=False)
    if (vectors.shape != (len(table_ids), metadata["dimension"]) or
            len(table_ids) != metadata["table_count"] or not table_ids or
            table_ids != sorted(set(table_ids))):
        raise ValueError("Invalid index shape or table IDs.")
    for uid in table_ids:
        check_id(uid)
    model = load_retriever(checkpoint).to(select_device(device)).eval()
    if metadata["similarity"] != model.similarity or metadata["dimension"] != model.dimension:
        raise ValueError("Index scoring or dimension does not match the model.")
    return table_ids, vectors, model, metadata


@torch.inference_mode()
def search_texts(resource, texts, top_k):
    table_ids, vectors, model, _ = resource
    query_vectors = model.encode_queries(texts).cpu().numpy()
    indices, scores = rank_tables(query_vectors, vectors, top_k)
    return [[(table_ids[index], float(score)) for index, score in zip(hits, values)]
            for hits, values in zip(indices, scores)]


def search(args):
    if args.top_k < 1 or args.batch_size < 1:
        raise ValueError("top-k and batch-size must be positive")
    resource = load_search_index(args.index, args.model, args.device)
    queries = load_queries(args.queries)
    rankings = {}
    for start in range(0, len(queries), args.batch_size):
        batch = queries[start:start + args.batch_size]
        hits = search_texts(resource, [q['query'] for q in batch], args.top_k)
        rankings.update((check_id(q['query_id']), row) for q, row in zip(batch, hits))
    return rankings, resource[3], resource[2].similarity


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


ROOT = Path(__file__).resolve().parents[3]


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
                        default=Path(__file__).resolve().parents[2] / 'settings.json')
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
