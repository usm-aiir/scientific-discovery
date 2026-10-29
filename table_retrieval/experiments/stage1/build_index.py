"""Build the frozen BM25 and BGE indexes used by Stage 1 retrieval."""
import argparse
from pathlib import Path

import numpy as np
from tqdm.auto import tqdm

from table_retrieval.data import DEFAULT_DATA_DIR, RESULTS_DIR, check_id, digest, load_corpus, load_json, write_json
from table_retrieval.live.stage1.artifacts import resolve_device
from table_retrieval.live.stage1.retrievers.bm25 import BM25
from table_retrieval.live.stage1.text import table_text

DEFAULT_INDEX = RESULTS_DIR / 'stage1/index'


def prepare_text_inputs(corpus, tokenizer, max_length):
    """Convert table text into token IDs and record tables that need shortening."""
    ids = sorted(check_id(uid) for uid in corpus)
    encoded, clipped = [], []

    # Leave room for the special tokens added by the tokenizer.
    budget = max_length - tokenizer.num_special_tokens_to_add(pair=False)

    for uid in tqdm(ids, desc='Preparing shared table text'):
        text = table_text(corpus[uid])

        # Count the tokens before shortening the text.
        token_ids = tokenizer(text, add_special_tokens=False, truncation=False)['input_ids']
        if len(token_ids) > budget:
            clipped.append({'table_id': uid, 'original_tokens': len(token_ids)})

        # Prepare the model input, shortening tables that exceed the limit.
        encoded.append(tokenizer(text, add_special_tokens=True, truncation=True,
                                 max_length=max_length)['input_ids'])

    return ids, encoded, clipped


def prepare_tables(corpus_path, output, model, revision, max_length):
    """Prepare and save the table inputs, tokenizer, and retrieval settings."""
    from transformers import AutoConfig, AutoTokenizer

    # Use an empty folder to avoid mixing files from different index builds.
    if output.exists() and any(output.iterdir()):
        raise ValueError('Build needs a new or empty output directory; inspect partial files or choose a new output.')

    corpus = load_corpus(corpus_path)

    # Save the exact model version so the same version can be loaded later.
    config = AutoConfig.from_pretrained(model, revision=revision)
    revision = getattr(config, '_commit_hash', None) or revision
    tokenizer = AutoTokenizer.from_pretrained(model, revision=revision)

    output.mkdir(parents=True, exist_ok=True)
    tokenizer.save_pretrained(output / 'tokenizer')

    # Save table IDs, their token inputs, and a record of tables needing shortening.
    ids, encoded, clipped = prepare_text_inputs(corpus, tokenizer, max_length)
    write_json(output / 'table_ids.json', ids)
    write_json(output / 'table_inputs.json', encoded)
    write_json(output / 'truncated_tables.json', clipped)
    print(f'Prepared {len(ids):,} tables; {len(clipped):,} exceeded {max_length} tokens.', flush=True)

    # Record the settings and file fingerprints used to check for later changes.
    return {
        'model': model, 'revision': revision, 'max_length': max_length,
        'fields': 'paper_title + caption + first row (headers) + all subsequent rows',
        'table_count': len(ids), 'truncated_tables': len(clipped),
        'candidate_depth': 100, 'top_k': 10, 'rrf_constant': 60, 'rrf_weights': [1, 1],
        'bm25_k1': 1.2, 'bm25_b': 0.75,
        'corpus': str(corpus_path), 'corpus_sha256': digest(corpus_path),
        'inputs_sha256': digest(output / 'table_inputs.json'),
        'ids_sha256': digest(output / 'table_ids.json'),
        'pooling': 'L2-normalized CLS; no query instruction', 'finetuned': False,
    }

def save_lexical(path, model, manifest):
    """Save the BM25 index so it can be loaded later for searching."""

    # Collect the tables and weights associated with each search term.
    terms = list(model.postings)
    indices, weights = zip(*(model.postings[term] for term in terms))

    # Record where each term's entries begin in the combined arrays.
    offsets = np.cumsum([0] + [len(values) for values in indices])

    # Write to a temporary file, then rename it once saving is complete.
    temporary = path.with_suffix('.tmp')
    with temporary.open('xb') as stream:
        np.savez(stream, ids=np.array(model.table_ids), terms=np.array(terms), offsets=offsets,
                 indices=np.concatenate(indices), weights=np.concatenate(weights),
                 inputs_sha256=manifest['inputs_sha256'], corpus_sha256=manifest['corpus_sha256'],
                 k1=manifest['bm25_k1'], b=manifest['bm25_b'])
    temporary.replace(path)


def index_lexical(index):
    """Build BM25 using the same table text included in BGE inputs."""
    from transformers import AutoTokenizer

    # Check that the corpus and prepared files have not changed.
    manifest = load_json(index / 'prepared.json')
    if digest(manifest['corpus']) != manifest['corpus_sha256']:
        raise ValueError('Corpus changed since preparation')
    for filename, key in [('table_ids.json', 'ids_sha256'), ('table_inputs.json', 'inputs_sha256')]:
        if digest(index / filename) != manifest[key]:
            raise ValueError(f'Prepared {filename} changed')

    corpus = load_corpus(manifest['corpus'])
    ids = load_json(index / 'table_ids.json')
    sequences = load_json(index / 'table_inputs.json')

    # Check that the saved IDs match the corpus and each table has one input.
    if ids != sorted(corpus) or len(sequences) != len(ids):
        raise ValueError('Corpus and prepared inputs differ')

    # Check the start and end tokens before removing them to recover the text.
    tokenizer = AutoTokenizer.from_pretrained(index / 'tokenizer')
    wrapper = tokenizer('', add_special_tokens=True)['input_ids']
    if len(wrapper) != 2 or any(seq[0] != wrapper[0] or seq[-1] != wrapper[-1] for seq in sequences):
        raise ValueError('Unexpected tokenizer special-token layout')

    path = index / 'bm25_matched.npz'
    if path.exists():
        return

    # Recover only the text included in the prepared BGE inputs.
    texts = {uid: tokenizer.decode(seq[1:-1], clean_up_tokenization_spaces=False)
             for uid, seq in zip(ids, sequences)}
    model = BM25(texts, k1=manifest['bm25_k1'], b=manifest['bm25_b'])
    save_lexical(path, model, manifest)

def index_bge(index, device, batch_size):
    """Convert table inputs into BGE vectors and save them for searching."""
    from transformers import AutoTokenizer
    from table_retrieval.live.stage1.retrievers.bge import encode, load_model

    manifest = load_json(index / 'prepared.json')

    # Stop if files from an earlier or unfinished BGE index already exist.
    if (index / 'index.json').exists() or (index / 'embeddings.npy').exists():
        raise ValueError('Index already exists or is partial; choose a new output directory')
    if digest(index / 'table_inputs.json') != manifest['inputs_sha256']:
        raise ValueError('Prepared table inputs changed')

    sequences = load_json(index / 'table_inputs.json')
    tokenizer = AutoTokenizer.from_pretrained(index / 'tokenizer')
    model = load_model(manifest, device)

    # Create a file with space for one vector per table.
    # Each vector represents the table text as numbers for similarity search.
    embeddings = np.lib.format.open_memmap(
        index / 'embeddings.npy', mode='w+', dtype=np.float32,
        shape=(len(sequences), model.config.hidden_size))

    # Process a few tables at a time to limit memory use.
    for start in tqdm(range(0, len(sequences), batch_size), desc='Encoding tables'):
        batch = sequences[start:start + batch_size]
        embeddings[start:start + len(batch)] = encode(model, tokenizer, batch, device)

    # Finish writing the vectors before recording the completed index details.
    embeddings.flush()
    write_json(index / 'index.json', {
        'prepared_sha256': digest(index / 'prepared.json'),
        'dimension': model.config.hidden_size,
        'table_count': len(sequences),
        'precision': 'fp16' if device.startswith('cuda') else 'fp32',
    })


def build(corpus=None, output=DEFAULT_INDEX, *, model='BAAI/bge-m3',
          revision='5617a9f61b028005a4858fdac845db406aefb181',
          max_length=2048, batch_size=2, device='auto', data_dir=DEFAULT_DATA_DIR):
    """Build the BM25 and BGE indexes used by validation, test, and live search."""

    # Check the input settings and choose the CPU or GPU.
    if not 4 <= max_length <= 8192 or batch_size < 1:
        raise ValueError('Require max_length 4..8192 and a positive batch_size.')
    corpus = Path(corpus) if corpus is not None else Path(data_dir) / 'Corpus.json'
    corpus, output = corpus.resolve(), Path(output).resolve()
    device = resolve_device(device)
    if not corpus.is_file():
        raise FileNotFoundError(f'External table corpus not found: {corpus}')

    # Prepare the table inputs, then build both types of search index.
    manifest = prepare_tables(corpus, output, model, revision, max_length)
    write_json(output / 'prepared.json', manifest)
    index_lexical(output)
    index_bge(output, device, batch_size)
    return output


def main(argv=None):
    """Read command-line options and start the index build."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--data-dir', type=Path, default=DEFAULT_DATA_DIR, help='Dataset directory (default: Behrooz v2)')
    parser.add_argument('--corpus', type=Path, help='Override the dataset directory\'s Corpus.json')
    parser.add_argument('--output', type=Path, default=DEFAULT_INDEX)
    parser.add_argument('--device', default='auto')
    parser.add_argument('--batch-size', type=int, default=2)
    args = parser.parse_args(argv)

    try:
        print(build(args.corpus, args.output, device=args.device, batch_size=args.batch_size,
                    data_dir=args.data_dir))
    except (OSError, ValueError, RuntimeError) as error:
        parser.exit(1, f'Index build failed: {error}\n')


if __name__ == '__main__':
    main()
