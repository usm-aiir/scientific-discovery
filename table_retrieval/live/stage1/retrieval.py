# Runs BM25, BGE, or fused Stage 1 table retrieval.
# bm25_matched  : BM25 over the same truncated text window BGE sees


DEFAULT_TABLE_TOP_K = 5 # Default number of Stage 1 tables to return.

# Defines which backend resources each retrieval mode needs.
BACKENDS = {'fusion': ['bm25_matched', 'bge'], 'bm25': ['bm25_matched'], 'bge': ['bge']}

# Runs the selected retrieval method and returns ranked table IDs with scores.
def search_query(query, model, resources, top_k=DEFAULT_TABLE_TOP_K):
    """Rank a live query with already-loaded backends; return (table ID, score)."""
    if not query.strip() or top_k < 1:
        raise ValueError('Enter a nonempty query and a positive top-k')
    if model == 'bm25':
        return resources['bm25_matched'][0].search(query, top_k)
    if model in ('bge', 'fusion'):
        from .retrievers.bge import search_texts
        if model == 'bge':
            return search_texts(resources['bge'], [query], top_k)[0][0]
        from .fusion import fuse
        lexical, manifest = resources['bm25_matched']
        depth = manifest['candidate_depth']
        lexical_hits = lexical.search(query, depth)
        dense_hits = search_texts(resources['bge'], [query], depth)[0][0]
        return fuse(lexical_hits, dense_hits, depth, manifest['rrf_constant'])[:top_k]
    raise ValueError(f'Unknown retrieval model: {model}')
