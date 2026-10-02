"""Load corpus, queries, judgments, and saved retrieval files for both stages."""

from collections import defaultdict
import hashlib
import json
import re
from pathlib import Path

DATA = Path("arxiv_data/SIGIRSciDis/tableGen/table_query_output")


def load_json(path):
    """Read a UTF-8 JSON file."""
    return json.loads(Path(path).read_text(encoding="utf-8"))


def check_id(value):
    """Return an identifier safe for whitespace-separated TREC files."""
    value = str(value)
    if not value or any(char.isspace() for char in value):
        raise ValueError(f"Invalid TREC identifier: {value!r}")
    return value


def normalize_table_id(uid):
    """Match corpus IDs to qrels IDs without changing either input file.

    For example, 2504.16674::T1::b becomes 2504.16674_1_b.
    Preserve subtable suffixes (including empty ones) and unrelated IDs.
    """
    match = re.fullmatch(r"(\d{4}\.\d{4,5}(?:v\d+)?)::T(\d+)(?:::(.*))?", uid)
    if match is None:
        return uid
    paper, table, subtable = match.groups()
    return f"{paper}_{table}" + (f"_{subtable}" if subtable is not None else "")


def digest(path):
    result = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b''):
            result.update(block)
    return result.hexdigest()


def write_json(path, value):
    path = Path(path)
    if path.exists():
        if load_json(path) == value:
            return
        raise ValueError(f'Existing {path} differs; choose a new experiment/output path')
    temporary = path.with_suffix(path.suffix + '.tmp')
    with temporary.open('x') as stream:
        stream.write(json.dumps(value, indent=2) + '\n')
    temporary.replace(path)


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


def load_corpus(path):
    corpus = load_json(path)
    if not isinstance(corpus, dict) or not corpus:
        raise ValueError('Corpus must be a nonempty object keyed by table ID')
    for uid, table in corpus.items():
        if not isinstance(table, dict):
            raise ValueError(f'Table {uid} must be an object')
        rows = table.get('rows')
        if rows is not None and (not isinstance(rows, list) or any(not isinstance(row, list) for row in rows)):
            raise ValueError(f'Table {uid} rows must be lists of cells')
    normalized = [normalize_table_id(check_id(uid)) for uid in corpus]
    if len(normalized) != len(set(normalized)):
        raise ValueError('Ambiguous corpus table IDs')
    return corpus


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
