"""Combine saved lexical and dense rankings using reciprocal rank fusion."""
from .data import load_json, read_run, load_queries, check_id
from collections import defaultdict

def fuse(first, second, depth=100, constant=60):
    """Equal-weight RRF; absent candidates contribute zero; IDs break ties."""
    scores = defaultdict(float)
    for hits in (first, second):
        for rank, (uid, _) in enumerate(hits[:depth], 1):
            scores[uid] += 1 / (constant + rank)
    return sorted(scores.items(), key=lambda item: (-item[1], item[0]))


def combine(args):
    manifest = load_json(args.output / 'prepared.json')
    lexical = read_run(getattr(args, 'rankings', args.output) / 'bm25_matched.run')
    dense = read_run(getattr(args, 'rankings', args.output) / 'bge.run')
    queries = load_queries(args.output / 'queries.json')
    combined = {check_id(q['query_id']): fuse(lexical.get(str(q['query_id']), []),
                dense.get(str(q['query_id']), []), manifest['candidate_depth'], manifest['rrf_constant'])[:getattr(args, 'top_k', 10)]
                for q in queries}
    return combined
