"""Combines saved BM25 and BGE runs for offline retrieval evaluation."""
from ...data import check_id, load_json
from .data import load_queries, read_run
from ...stage1.fusion import fuse


def combine(args):
    manifest = load_json(args.output / 'prepared.json')
    lexical = read_run(args.rankings / 'bm25_matched.run')
    dense = read_run(args.rankings / 'bge.run')
    queries = load_queries(args.output / 'queries.json')
    return {
        check_id(query['query_id']): fuse(
            lexical.get(str(query['query_id']), []),
            dense.get(str(query['query_id']), []),
            manifest['candidate_depth'],
            manifest['rrf_constant'],
        )[:args.top_k]
        for query in queries
    }
