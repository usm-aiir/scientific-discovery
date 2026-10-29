"""Loads and searches the lexical BM25 backend used by Stage 1."""
from collections import Counter, defaultdict
import math
import re
import numpy as np
from table_retrieval.data import check_id

# Converts text into lowercase tokens while keeping decimals and hyphenated words together.
def tokenize(text):
    
    return re.findall(r"\w+(?:[.\-]\w+)*", text.lower())

# Builds a BM25 index from the prepared table text.
class BM25:
# Prepares table tokens and BM25 weights for fast searching.
    def __init__(self, texts, k1=1.2, b=0.75):
        if not texts:
            raise ValueError("The text collection is empty.")
        if any(not isinstance(text, str) for text in texts.values()):
            raise TypeError("BM25 expects table IDs mapped to prepared text strings.")
        if not math.isfinite(k1) or k1 <= 0 or not 0 <= b <= 1:
            raise ValueError("Require finite k1 > 0 and 0 <= b <= 1.")
        self.table_ids = sorted(check_id(uid) for uid in texts)
        lengths = np.zeros(len(self.table_ids), dtype=np.float64)
        postings = defaultdict(list)
        for index, uid in enumerate(self.table_ids):
            counts = Counter(tokenize(texts[uid]))
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

# Scores tables against the query and returns the top matching tables.
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

# Loads a saved BM25 index and checks that it still matches the prepared data.
def load_lexical(path, manifest):
    """Restore a validated, pickle-free BM25 index."""
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
