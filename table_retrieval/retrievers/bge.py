"""Frozen BGE-M3 encoding and exact dense search."""
from ..data import load_queries, check_id
from . import rank_tables
from pathlib import Path
import numpy as np
import torch
from torch.nn import functional as F
from transformers import AutoModel, AutoTokenizer
from tqdm.auto import tqdm
from ..data import load_json, digest, write_json

CACHE = Path('results/model_cache')

def load_model(manifest, device):
    model = AutoModel.from_pretrained(manifest['model'], revision=manifest['revision'],
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


def search(args):
    from transformers import AutoTokenizer
    from tqdm.auto import tqdm
    manifest = load_json(args.output / 'prepared.json')
    metadata = load_json(args.output / 'index.json')
    if metadata['prepared_sha256'] != digest(args.output / 'prepared.json'):
        raise ValueError('Index and prepared manifest differ')
    if digest(manifest['qrels']) != manifest['qrels_sha256']:
        raise ValueError('Judgments changed since preparation')
    if (digest(args.output / 'table_ids.json') != manifest['ids_sha256'] or
            digest(args.output / 'queries.json') != manifest['saved_queries_sha256']):
        raise ValueError('Prepared table IDs or queries changed')
    ids, queries = load_json(args.output / 'table_ids.json'), load_queries(args.output / 'queries.json')
    vectors = np.load(args.output / 'embeddings.npy', mmap_mode='r', allow_pickle=False)
    if vectors.shape != (len(ids), metadata['dimension']) or not np.isfinite(vectors).all():
        raise ValueError('Invalid dense index')
    tokenizer = AutoTokenizer.from_pretrained(args.output / 'tokenizer')
    model = load_model(manifest, args.device)
    ranks = {}
    query_clipped = []
    for start in tqdm(range(0, len(queries), args.batch_size), desc='Retrieving queries'):
        batch = queries[start:start + args.batch_size]
        seqs = []
        for query in batch:
            tokens = tokenizer(query['query'], add_special_tokens=False)['input_ids']
            budget = manifest['max_length'] - tokenizer.num_special_tokens_to_add(pair=False)
            if len(tokens) > budget:
                query_clipped.append(str(query['query_id']))
            seqs.append(tokenizer.build_inputs_with_special_tokens(tokens[:budget]))
        qvectors = encode(model, tokenizer, seqs, args.device)
        hits, scores = rank_tables(qvectors, vectors, manifest['candidate_depth'])
        for query, row, values in zip(batch, hits, scores):
            ranks[check_id(query['query_id'])] = [(ids[i], float(score)) for i, score in zip(row, values)]
    return ranks, query_clipped
