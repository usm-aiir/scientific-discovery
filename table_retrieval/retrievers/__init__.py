"""Shared exact vector ranking with deterministic ties."""
import numpy as np

def rank_tables(query_vectors, table_vectors, top_k):
    """Return exact top-k indices and scores; corpus order breaks ties."""
    if top_k < 1:
        raise ValueError("top_k must be positive.")
    scores = query_vectors @ table_vectors.T
    if not np.isfinite(scores).all():
        raise ValueError("Non-finite retrieval scores.")
    indices = np.argsort(-scores, axis=1, kind="stable")[:, :top_k]
    return indices, np.take_along_axis(scores, indices, axis=1)
