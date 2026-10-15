# Builds the BGE evaluation index and runs saved evaluation queries.
import numpy as np
from tqdm.auto import tqdm
from transformers import AutoTokenizer

from ...data import check_id, digest, load_json, write_json
from ..data import load_queries
from ...stage1.retrievers.bge import encode, load_model, load_search_index, search_texts

# Builds and saves BGE embeddings for all prepared tables in the evaluation set.
def index_bge(args):
    manifest = load_json(args.output / 'prepared.json')
    if (args.output / 'index.json').exists() or (args.output / 'embeddings.npy').exists():
        raise ValueError('Index already exists or is partial; choose a new output directory')
    if digest(args.output / 'table_inputs.json') != manifest['inputs_sha256']:
        raise ValueError('Prepared table inputs changed')
    sequences = load_json(args.output / 'table_inputs.json')
    tokenizer = AutoTokenizer.from_pretrained(args.output / 'tokenizer')
    model = load_model(manifest, args.device)
    destination = args.embeddings
    if destination.exists():
        raise ValueError('Embeddings already exist; choose a new evaluation name')
    destination.parent.mkdir(parents=True, exist_ok=True)
    if destination != args.output / 'embeddings.npy':
        (args.output / 'embeddings.npy').symlink_to(destination.resolve())
    embeddings = np.lib.format.open_memmap(
        destination, mode='w+', dtype=np.float32,
        shape=(len(sequences), model.config.hidden_size))
    for start in tqdm(range(0, len(sequences), args.batch_size), desc='Encoding tables'):
        batch = sequences[start:start + args.batch_size]
        embeddings[start:start + len(batch)] = encode(model, tokenizer, batch, args.device)
    embeddings.flush()
    write_json(args.output / 'index.json', {
        'prepared_sha256': digest(args.output / 'prepared.json'),
        'dimension': model.config.hidden_size,
        'table_count': len(sequences),
        'precision': 'fp16' if args.device.startswith('cuda') else 'fp32',
    })

# Runs all prepared evaluation queries against the saved BGE index.
def search(args):
    manifest = load_json(args.output / 'prepared.json')
    if digest(manifest['qrels']) != manifest['qrels_sha256']:
        raise ValueError('Judgments changed since preparation')
    if digest(args.output / 'queries.json') != manifest['saved_queries_sha256']:
        raise ValueError('Prepared queries changed')
    queries = load_queries(args.output / 'queries.json')
    resource = load_search_index(args.output, args.device)
    rankings, query_clipped = {}, []
    for start in tqdm(range(0, len(queries), args.batch_size), desc='Retrieving queries'):
        batch = queries[start:start + args.batch_size]
        hits, clipped = search_texts(resource, [query['query'] for query in batch],
                                     manifest['candidate_depth'])
        query_clipped.extend(str(batch[index]['query_id']) for index in clipped)
        rankings.update((check_id(query['query_id']), row) for query, row in zip(batch, hits))
    return rankings, query_clipped
