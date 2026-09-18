# Objective: Index table text and rank tables with the BM25 scoring formula.

from collections import Counter, defaultdict
import math

import numpy as np

from ..data import check_id, table_to_text, tokenize


class BM25:
    """An in-memory inverted index: each term points to tables containing it.

    Uses positive IDF: log(1 + (N - df + 0.5) / (df + 0.5)).
    k1 controls term-frequency saturation; b controls length normalization.
    Table IDs break score ties deterministically. No training is required.
    """

    def __init__(self, corpus, k1=1.2, b=0.75):
        if not corpus:
            raise ValueError("The corpus is empty.")
        if not math.isfinite(k1) or k1 <= 0 or not 0 <= b <= 1:
            raise ValueError("Require finite k1 > 0 and 0 <= b <= 1.")
        self.table_ids = sorted(check_id(uid) for uid in corpus)
        lengths = np.zeros(len(self.table_ids), dtype=np.float64)
        postings = defaultdict(list)
        for index, uid in enumerate(self.table_ids):
            counts = Counter(tokenize(table_to_text(corpus[uid])))
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
