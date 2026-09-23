"""TAPAS encoders, checkpoint loading, and corpus indexing."""
from ..data import load_json, load_queries
from . import rank_tables
from dataclasses import asdict
import hashlib
import json
from pathlib import Path
import numpy as np
import torch
from torch import nn
from torch.nn import functional as F
from transformers import BertModel, BertTokenizer, TapasModel, TapasTokenizer
from ..data import (check_id, InputLimits, table_inputs, table_features, load_corpus, digest)

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


def search(args):
    import torch
    if args.top_k < 1 or args.batch_size < 1:
        raise ValueError("top-k and batch-size must be positive")
    metadata = load_json(args.index / "index.json")
    if checkpoint_fingerprint(args.model) != metadata["model_fingerprint"]:
        raise ValueError("Index and model do not match. Rebuild the index with this checkpoint.")
    table_ids = load_json(args.index / "table_ids.json")
    vectors = np.load(args.index / "embeddings.npy", mmap_mode="r", allow_pickle=False)
    if (vectors.shape != (len(table_ids), metadata["dimension"]) or
            len(table_ids) != metadata["table_count"] or not table_ids or
            table_ids != sorted(set(table_ids))):
        raise ValueError("Invalid index shape or table IDs.")
    for uid in table_ids:
        check_id(uid)
    queries = load_queries(args.queries)
    query_ids = [check_id(query["query_id"]) for query in queries]
    model = load_retriever(args.model).to(select_device(args.device)).eval()
    if metadata["similarity"] != model.similarity or metadata["dimension"] != model.dimension:
        raise ValueError("Index scoring or dimension does not match the model.")
    rankings = {}
    with torch.inference_mode():
        for start in range(0, len(queries), args.batch_size):
            batch = queries[start:start + args.batch_size]
            query_vectors = model.encode_queries([q["query"] for q in batch]).cpu().numpy()
            indices, scores = rank_tables(query_vectors, vectors, args.top_k)
            for qid, hits, values in zip(query_ids[start:], indices, scores):
                rankings[qid] = [(table_ids[index], float(score)) for index, score in zip(hits, values)]
    return rankings, metadata, model.similarity
