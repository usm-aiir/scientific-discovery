"""Loads and searches the BGE-M3 dense backend used by Stage 1."""
from .ranking import rank_tables
from pathlib import Path
import numpy as np
import torch
from torch.nn import functional as F
from transformers import AutoModel, AutoTokenizer
from table_retrieval.data import digest, load_json

# Use Hugging Face's default cache location for downloaded models.
CACHE = None

# Loads the BGE model specified in the prepared manifest...settings.json.
# The manifest contains the model name and exact model revision.
# Uses a local saved model if available; otherwise Hugging Face downloads it.
def load_model(manifest, device, artifact_root=None):
    reference = manifest['model']
    if artifact_root is not None:
        local = Path(artifact_root) / reference
        if local.exists():
            reference = str(local.resolve())
    model = AutoModel.from_pretrained(reference, revision=manifest['revision'],
                                     torch_dtype=torch.float16 if device.startswith('cuda') else torch.float32,
                                     cache_dir=CACHE, attn_implementation='eager').to(device).eval()
    return model


# Sends tokenized text through the BGE model and returns normalized embeddings.
@torch.inference_mode()
def encode(model, tokenizer, sequences, device):
    features = [{'input_ids': seq, 'attention_mask': [1] * len(seq)} for seq in sequences]
    inputs = tokenizer.pad(features, padding=True, return_tensors='pt')
    output = model(**{key: value.to(device) for key, value in inputs.items()})
    vectors = F.normalize(output.last_hidden_state[:, 0].float(), dim=-1).cpu().numpy()
    if not np.isfinite(vectors).all():
        raise ValueError('Non-finite dense vectors')
    return vectors

# Loads the saved Stage 1 BGE index, tokenizer, model, table IDs, and embeddings.
def load_search_index(output, device):
    """Load a reusable dense search resource, independent of query files."""
    manifest = load_json(output / 'prepared.json')
    metadata = load_json(output / 'index.json')
    if metadata['prepared_sha256'] != digest(output / 'prepared.json'):
        raise ValueError('Index and prepared manifest differ')
    if digest(output / 'table_ids.json') != manifest['ids_sha256']:
        raise ValueError('Prepared table IDs changed')
    ids = load_json(output / 'table_ids.json')
    vectors = np.load(output / 'embeddings.npy', mmap_mode='r', allow_pickle=False)
    if vectors.shape != (len(ids), metadata['dimension']) or not np.isfinite(vectors).all():
        raise ValueError('Invalid dense index')
    tokenizer = AutoTokenizer.from_pretrained(output / 'tokenizer')
    model = load_model(manifest, device, output)
    return manifest, ids, vectors, tokenizer, model, device


def search_texts(resource, texts, top_k):
    """Encode queries with the saved token budget and search the dense index."""
    manifest, ids, vectors, tokenizer, model, device = resource
    seqs, clipped = [], []
    budget = manifest['max_length'] - tokenizer.num_special_tokens_to_add(pair=False)
    for index, text in enumerate(texts):
        tokens = tokenizer(text, add_special_tokens=False)['input_ids']
        if len(tokens) > budget:
            clipped.append(index)
        seqs.append(tokenizer(text, add_special_tokens=True, truncation=True,
                              max_length=manifest['max_length'])['input_ids'])
    qvectors = encode(model, tokenizer, seqs, device)
    hits, scores = rank_tables(qvectors, vectors, top_k)
    return [[(ids[i], float(score)) for i, score in zip(row, values)]
            for row, values in zip(hits, scores)], clipped
