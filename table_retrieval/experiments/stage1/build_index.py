# Builds the frozen BM25 and BGE indexes used by Stage 1 retrieval.
import argparse
from pathlib import Path
from types import SimpleNamespace

import numpy as np
from tqdm.auto import tqdm
from transformers import AutoConfig, AutoTokenizer

from table_retrieval.data import check_id, digest, load_corpus, load_json, write_json
from table_retrieval.stage1.artifacts import resolve_device
from table_retrieval.stage1.retrievers.bge import encode, load_model
from table_retrieval.stage1.retrievers.bm25 import BM25
from table_retrieval.stage1.text import table_text

ROOT = Path(__file__).resolve().parents[3]
DEFAULT_INDEX = ROOT / 'results/stage1/index'

def prepare_text_inputs(corpus, tokenizer, max_length):
    """Tokenize once, recording tables truncated to the configured index budget."""
    ids = sorted(check_id(uid) for uid in corpus)
    encoded, clipped = [], []
    budget = max_length - tokenizer.num_special_tokens_to_add(pair=False)
    for uid in tqdm(ids, desc='Preparing shared table text'):
        text = table_text(corpus[uid])
        token_ids = tokenizer(text, add_special_tokens=False, truncation=False)['input_ids']
        if len(token_ids) > budget:
            clipped.append({'table_id': uid, 'original_tokens': len(token_ids)})
        encoded.append(tokenizer(text, add_special_tokens=True, truncation=True,
                                 max_length=max_length)['input_ids'])
    return ids, encoded, clipped


def prepare_tables(args):
    from transformers import AutoConfig, AutoTokenizer
    from ...stage1.retrievers.bge import CACHE
    if args.output.exists() and any(args.output.iterdir()):
        raise ValueError('prepare needs a new or empty output directory; inspect partial files or choose a new experiment')
    corpus = load_corpus(args.corpus)
    config = AutoConfig.from_pretrained(args.model, revision=args.revision, cache_dir=CACHE)
    revision = getattr(config, '_commit_hash', None) or args.revision
    tokenizer = AutoTokenizer.from_pretrained(args.model, revision=revision, cache_dir=CACHE)
    args.output.mkdir(parents=True, exist_ok=True)
    tokenizer.save_pretrained(args.output / 'tokenizer')
    ids, encoded, clipped = prepare_text_inputs(corpus, tokenizer, args.max_length)
    write_json(args.output / 'table_ids.json', ids)
    write_json(args.output / 'table_inputs.json', encoded)
    write_json(args.output / 'truncated_tables.json', clipped)
    manifest = {'model': args.model, 'revision': revision, 'max_length': args.max_length,
                'fields': 'paper_title + caption + first row (headers) + all subsequent rows',
                'table_count': len(ids), 'truncated_tables': len(clipped),
                'candidate_depth': args.depth, 'top_k': getattr(args, 'top_k', 10),
                'rrf_constant': getattr(args, 'rrf_constant', 60), 'rrf_weights': [1, 1],
                'bm25_k1': 1.2, 'bm25_b': 0.75,
                'corpus': str(args.corpus), 'corpus_sha256': digest(args.corpus),
                'inputs_sha256': digest(args.output / 'table_inputs.json'),
                'ids_sha256': digest(args.output / 'table_ids.json'),
                'pooling': 'L2-normalized CLS; no query instruction', 'finetuned': False}
    print(f'Prepared {len(ids):,} tables; {len(clipped):,} exceeded {args.max_length} tokens.', flush=True)
    return manifest


def index_lexical(args):
    manifest = load_json(args.output / 'prepared.json')
    if digest(manifest['corpus']) != manifest['corpus_sha256']:
        raise ValueError('Corpus changed since preparation')
    for filename, key in [('table_ids.json', 'ids_sha256'), ('table_inputs.json', 'inputs_sha256')]:
        if digest(args.output / filename) != manifest[key]:
            raise ValueError(f'Prepared {filename} changed')
    corpus = load_corpus(manifest['corpus'])
    ids = load_json(args.output / 'table_ids.json')
    sequences = load_json(args.output / 'table_inputs.json')
    if ids != sorted(corpus) or len(sequences) != len(ids):
        raise ValueError('Corpus and prepared inputs differ')
    tokenizer = AutoTokenizer.from_pretrained(args.output / 'tokenizer')
    wrapper = tokenizer('', add_special_tokens=True)['input_ids']
    if len(wrapper) != 2 or any(seq[0] != wrapper[0] or seq[-1] != wrapper[-1] for seq in sequences):
        raise ValueError('Unexpected tokenizer special-token layout')
    for name in ('bm25_full', 'bm25_matched'):
        path = args.output / f'{name}.npz'
        if path.exists():
            continue
        texts = ({uid: table_text(corpus[uid]) for uid in ids} if name == 'bm25_full' else
                 {uid: tokenizer.decode(seq[1:-1], clean_up_tokenization_spaces=False)
                  for uid, seq in zip(ids, sequences)})
        model = BM25(texts, k1=manifest['bm25_k1'], b=manifest['bm25_b'])
        terms = list(model.postings)
        indices, weights = zip(*(model.postings[term] for term in terms))
        offsets = np.cumsum([0] + [len(values) for values in indices])
        temporary = path.with_suffix('.tmp')
        with temporary.open('xb') as stream:
            np.savez(stream, ids=np.array(model.table_ids), terms=np.array(terms), offsets=offsets,
                     indices=np.concatenate(indices), weights=np.concatenate(weights),
                     inputs_sha256=manifest['inputs_sha256'], corpus_sha256=manifest['corpus_sha256'],
                     k1=manifest['bm25_k1'], b=manifest['bm25_b'])
        temporary.replace(path)


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

# Rebuild into a new directory; completed indexes are also usable by live search.
def build(corpus, output=DEFAULT_INDEX, *, model='BAAI/bge-m3',
          revision='5617a9f61b028005a4858fdac845db406aefb181',
          max_length=2048, batch_size=2, device='auto'):
    if not 4 <= max_length <= 8192 or batch_size < 1:
        raise ValueError('Require max_length 4..8192 and a positive batch_size.')
    args = SimpleNamespace(corpus=Path(corpus).resolve(), output=Path(output).resolve(),
                           model=model, revision=revision, max_length=max_length,
                           batch_size=batch_size, device=resolve_device(device),
                           depth=100, top_k=10, rrf_constant=60)
    args.embeddings = args.output / 'embeddings.npy'
    if not args.corpus.is_file():
        raise FileNotFoundError(f'External table corpus not found: {args.corpus}')
    manifest = prepare_tables(args)
    write_json(args.output / 'prepared.json', manifest)
    index_lexical(args)
    index_bge(args)
    return args.output


def main(argv=None):
    parser = argparse.ArgumentParser(description='Build the Stage 1 BM25/BGE indexes.')
    parser.add_argument('--corpus', type=Path, required=True, help='External full Corpus.json')
    parser.add_argument('--output', type=Path, default=DEFAULT_INDEX)
    parser.add_argument('--device', default='auto')
    parser.add_argument('--batch-size', type=int, default=2)
    args = parser.parse_args(argv)
    try:
        print(build(args.corpus, args.output, device=args.device, batch_size=args.batch_size))
    except (OSError, ValueError, RuntimeError) as error:
        parser.exit(1, f'Index build failed: {error}\n')


if __name__ == '__main__':
    main()
