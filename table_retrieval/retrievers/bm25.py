"""BM25 tokenization, persistent postings, and lexical search."""
from ..data import load_queries
from collections import Counter, defaultdict
import math
import re
import numpy as np
from transformers import AutoTokenizer
from ..data import check_id, table_to_text, table_text, load_json, load_corpus, digest

def tokenize(text):
    """Lowercase words/numbers; retain internal decimal points and hyphens.

    Examples: 'Qwen-0.5B' and '2.59' each remain one token. No stemming
    or stopword removal is applied in this first baseline.
    """
    return re.findall(r"\w+(?:[.\-]\w+)*", text.lower())

class BM25:
    """An in-memory inverted index: each term points to tables containing it.

    Uses positive IDF: log(1 + (N - df + 0.5) / (df + 0.5)).
    k1 controls term-frequency saturation; b controls length normalization.
    Table IDs break score ties deterministically. No training is required.
    """

    def __init__(self, corpus, k1=1.2, b=0.75, text_fn=table_to_text):
        if not corpus:
            raise ValueError("The corpus is empty.")
        if not math.isfinite(k1) or k1 <= 0 or not 0 <= b <= 1:
            raise ValueError("Require finite k1 > 0 and 0 <= b <= 1.")
        self.table_ids = sorted(check_id(uid) for uid in corpus)
        lengths = np.zeros(len(self.table_ids), dtype=np.float64)
        postings = defaultdict(list)
        for index, uid in enumerate(self.table_ids):
            counts = Counter(tokenize(text_fn(corpus[uid])))
            lengths[index] = sum(counts.values())
            for term, frequency in counts.items():
                postings[term].append((index, frequency))

        average_length = lengths.mean()
        if average_length == 0:
            raise ValueError("The corpus contains no searchable tokens.")
        normalization = k1 * (1 - b + b * lengths / average_length)
        self.postings = {}
        # Precompute each term's BM25 contribution for its matching tables.
        for term in list(postings):
            entries = np.asarray(postings.pop(term))
            indices = entries[:, 0].astype(np.int32)
            frequencies = entries[:, 1]
            df = len(indices)
            idf = math.log1p((len(lengths) - df + 0.5) / (df + 0.5))
            weights = idf * frequencies * (k1 + 1) / (frequencies + normalization[indices])
            self.postings[term] = (indices, weights)

    def search(self, query, top_k=10):
        """Return (table ID, score) pairs; omit tables with zero lexical overlap."""
        if top_k < 1:
            raise ValueError("top_k must be positive.")
        scores = np.zeros(len(self.table_ids), dtype=np.float64)
        for term, frequency in Counter(tokenize(query)).items():
            if term in self.postings:
                indices, weights = self.postings[term]
                scores[indices] += frequency * weights
        candidates = np.flatnonzero(scores > 0)
        order = np.argsort(-scores[candidates], kind="stable")[:top_k]
        return [(self.table_ids[i], float(scores[i])) for i in candidates[order]]

def index_lexical(args):
    """Persist BM25 postings as arrays (no pickle); reuse for every query split."""
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
    wrapper = tokenizer.build_inputs_with_special_tokens([])
    if len(wrapper) != 2 or any(seq[0] != wrapper[0] or seq[-1] != wrapper[-1] for seq in sequences):
        raise ValueError('Unexpected tokenizer special-token layout')
    for name in ['bm25_full', 'bm25_matched']:
        path = args.output / f'{name}.npz'
        if path.exists():
            continue
        texts = ({uid: table_text(corpus[uid]) for uid in ids} if name == 'bm25_full' else
                 {uid: tokenizer.decode(seq[1:-1], clean_up_tokenization_spaces=False)
                  for uid, seq in zip(ids, sequences)})
        model = BM25(texts, k1=manifest['bm25_k1'], b=manifest['bm25_b'], text_fn=lambda text: text)
        terms = list(model.postings)
        indices, weights = zip(*(model.postings[term] for term in terms))
        offsets = np.cumsum([0] + [len(values) for values in indices])
        # Atomic completion: partial files never look like complete indexes.
        temporary = path.with_suffix('.tmp')
        with temporary.open('xb') as stream:
            np.savez(stream, ids=np.array(model.table_ids), terms=np.array(terms), offsets=offsets,
                     indices=np.concatenate(indices), weights=np.concatenate(weights),
                     inputs_sha256=manifest['inputs_sha256'], corpus_sha256=manifest['corpus_sha256'],
                     k1=manifest['bm25_k1'], b=manifest['bm25_b'])
        temporary.replace(path)
        print(f'Indexed {len(ids):,} tables: {name}', flush=True)


def load_lexical(path, manifest):
    with np.load(path, allow_pickle=False) as arrays:
        for key in ['inputs_sha256', 'corpus_sha256']:
            if str(arrays[key]) != manifest[key]:
                raise ValueError('BM25 index and prepared inputs differ')
        if float(arrays['k1']) != manifest['bm25_k1'] or float(arrays['b']) != manifest['bm25_b']:
            raise ValueError('BM25 scoring parameters changed')
        model = BM25.__new__(BM25)
        model.table_ids = arrays['ids'].tolist()
        indices, weights, offsets = arrays['indices'], arrays['weights'], arrays['offsets']
        model.postings = {term: (indices[start:end], weights[start:end])
                          for term, start, end in zip(arrays['terms'].tolist(), offsets, offsets[1:])}
    return model


def search(args):
    manifest = load_json(args.output / 'prepared.json')
    queries = load_queries(args.output / 'queries.json')
    if digest(args.output / 'queries.json') != manifest['saved_queries_sha256']:
        raise ValueError('Prepared queries changed')
    results = {}
    for name in ['bm25_full', 'bm25_matched']:
        if (getattr(args, 'rankings', args.output) / f'{name}.run').exists():
            continue
        source = args.output / f'{name}.npz'
        if not source.exists() and getattr(args, 'source', None):
            source = args.source / f'{name}.npz'
        model = load_lexical(source, manifest)
        rankings = {check_id(q['query_id']): model.search(q['query'], manifest['candidate_depth'])
                    for q in queries}
        results[name] = rankings
    return results
