"""
BGE-M3 dense retrieval and search.

Encodes tables and queries with a frozen BGE-M3 model, builds a dense
embedding index, and performs exact similarity search for ranked retrieval.
"""
from .ranking import rank_tables
from pathlib import Path
import numpy as np
import torch
from torch.nn import functional as F
from transformers import AutoModel, AutoTokenizer
from tqdm.auto import tqdm
from ...data import check_id, digest, load_json, load_queries, write_json

# Let Transformers use its standard, user-configurable Hugging Face cache.
# A repository-relative cache makes otherwise identical service deployments
# depend on their process working directory.
CACHE = None

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


@torch.inference_mode()
def encode(model, tokenizer, sequences, device):
    features = [{'input_ids': seq, 'attention_mask': [1] * len(seq)} for seq in sequences]
    inputs = tokenizer.pad(features, padding=True, return_tensors='pt')
    output = model(**{key: value.to(device) for key, value in inputs.items()})
    vectors = F.normalize(output.last_hidden_state[:, 0].float(), dim=-1).cpu().numpy()
    if not np.isfinite(vectors).all():
        raise ValueError('Non-finite dense vectors')
    return vectors

def index_bge(args):
    manifest = load_json(args.output / 'prepared.json')
    if (args.output / 'index.json').exists() or (args.output / 'embeddings.npy').exists():
        raise ValueError('Index already exists or is partial; choose a new output directory for another run')
    if digest(args.output / 'table_inputs.json') != manifest['inputs_sha256']:
        raise ValueError('Prepared table inputs changed')
    sequences = load_json(args.output / 'table_inputs.json')
    tokenizer = AutoTokenizer.from_pretrained(args.output / 'tokenizer')
    model = load_model(manifest, args.device)
    destination = getattr(args, 'embeddings', args.output / 'embeddings.npy')
    if destination.exists():
        raise ValueError('Embeddings already exist; choose a new experiment')
    destination.parent.mkdir(parents=True, exist_ok=True)
    if destination != args.output / 'embeddings.npy':
        (args.output / 'embeddings.npy').symlink_to(destination.resolve())
    embeddings = np.lib.format.open_memmap(destination, mode='w+',
                   dtype=np.float32, shape=(len(sequences), model.config.hidden_size))
    for start in tqdm(range(0, len(sequences), args.batch_size), desc='Encoding tables'):
        batch = sequences[start:start + args.batch_size]
        embeddings[start:start + len(batch)] = encode(model, tokenizer, batch, args.device)
    embeddings.flush()
    write_json(args.output / 'index.json', {'prepared_sha256': digest(args.output / 'prepared.json'),
               'dimension': model.config.hidden_size, 'table_count': len(sequences),
               'precision': 'fp16' if args.device.startswith('cuda') else 'fp32'})


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
    """Search arbitrary text using the same query encoding as batch evaluation."""
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


def search(args):
    manifest = load_json(args.output / 'prepared.json')
    if digest(manifest['qrels']) != manifest['qrels_sha256']:
        raise ValueError('Judgments changed since preparation')
    if digest(args.output / 'queries.json') != manifest['saved_queries_sha256']:
        raise ValueError('Prepared queries changed')
    queries = load_queries(args.output / 'queries.json')
    resource = load_search_index(args.output, args.device)
    ranks, query_clipped = {}, []
    for start in tqdm(range(0, len(queries), args.batch_size), desc='Retrieving queries'):
        batch = queries[start:start + args.batch_size]
        hits, clipped = search_texts(resource, [q['query'] for q in batch],
                                    manifest['candidate_depth'])
        query_clipped.extend(str(batch[i]['query_id']) for i in clipped)
        ranks.update((check_id(q['query_id']), row) for q, row in zip(batch, hits))
    return ranks, query_clipped
