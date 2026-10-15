# Builds BM25 evaluation indexes and runs saved evaluation queries.
import numpy as np
from transformers import AutoTokenizer

from ...data import check_id, digest, load_corpus, load_json
from ..data import load_queries
from ...stage1.retrievers.bm25 import BM25, load_lexical
from ...stage1.text import table_text

# Builds and saves the BM25 indexes used for evaluation.
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

# Runs evaluation queries against the saved BM25 indexes.
def search(args):
    manifest = load_json(args.output / 'prepared.json')
    queries = load_queries(args.output / 'queries.json')
    if digest(args.output / 'queries.json') != manifest['saved_queries_sha256']:
        raise ValueError('Prepared queries changed')
    results = {}
    for name in ('bm25_full', 'bm25_matched'):
        if (args.rankings / f'{name}.run').exists():
            continue
        source = args.output / f'{name}.npz'
        if not source.exists() and args.source:
            source = args.source / f'{name}.npz'
        model = load_lexical(source, manifest)
        results[name] = {
            check_id(query['query_id']): model.search(query['query'], manifest['candidate_depth'])
            for query in queries
        }
    return results
