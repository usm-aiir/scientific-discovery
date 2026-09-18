# Objective: Dense table retrieval: BERT queries and table-aware TAPAS tables.
"""Dense table retrieval: BERT queries and table-aware TAPAS tables."""

from dataclasses import asdict, dataclass
import hashlib
import json
from pathlib import Path
import warnings

import numpy as np
import pandas as pd
import torch
from torch import nn
from torch.nn import functional as F
from transformers import BertModel, BertTokenizer, TapasModel, TapasTokenizer


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


def contrastive_loss(queries, tables, positives, temperature=0.05):
    """Average log-probability of all known positives present in each batch."""
    if not np.isfinite(temperature) or temperature <= 0:
        raise ValueError("Temperature must be finite and positive.")
    if not positives.any(dim=1).all() or positives.all(dim=1).any():
        raise ValueError("Each query needs a positive and a negative table in its batch.")
    log_probabilities = F.log_softmax(queries @ tables.T / temperature, dim=1)
    return -(log_probabilities * positives).sum(dim=1).div(positives.sum(dim=1)).mean()


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


def select_device(name):
    return "cuda" if name == "auto" and torch.cuda.is_available() else "cpu" if name == "auto" else name
