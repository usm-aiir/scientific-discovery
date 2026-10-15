"""Query datasets, TREC runs, and table judgments for offline evaluation."""
from collections import defaultdict
from pathlib import Path

from ...data import check_id, load_json, normalize_table_id


def write_run(path, rankings, name):
    path = Path(path)
    temporary = path.with_suffix(path.suffix + '.tmp')
    if path.exists():
        raise ValueError(f'Existing rankings {path}; choose a new experiment/output path')
    with temporary.open('x') as stream:
        for qid, hits in rankings.items():
            for rank, (uid, score) in enumerate(hits, 1):
                stream.write(f'{qid} Q0 {uid} {rank} {score:.12g} {name}\n')
    temporary.replace(path)


def read_run(path):
    result = defaultdict(list)
    for line in Path(path).read_text().splitlines():
        qid, _, uid, rank, score, _ = line.split()
        result[qid].append((int(rank), uid, float(score)))
    return {qid: [(uid, score) for _, uid, score in sorted(hits)] for qid, hits in result.items()}


def load_queries(path):
    queries = load_json(path)
    if not isinstance(queries, list) or any(not isinstance(q, dict) or
            'query_id' not in q or 'query' not in q for q in queries):
        raise ValueError('Queries must be a list of objects with query_id and query fields')
    ids = [check_id(q['query_id']) for q in queries]
    if not queries or len(ids) != len(set(ids)):
        raise ValueError('Queries must be nonempty and have unique IDs')
    if any(not isinstance(q['query'], str) or not q['query'].strip() for q in queries):
        raise ValueError('Queries must contain nonempty text')
    return queries


def load_qrels(path, min_grade=1):
    """Read positive judgments, normalizing table IDs for matching."""
    relevant = defaultdict(set)
    for line in Path(path).read_text().splitlines():
        if line.strip():
            qid, _, uid, grade = line.split()
            if int(grade) >= min_grade:
                relevant[check_id(qid)].add(normalize_table_id(check_id(uid)))
    return relevant
