# Validates TREC runs and calculates mean table Precision, Recall, and binary nDCG.
from collections import defaultdict
import math
from pathlib import Path

from table_retrieval.data import check_id, normalize_table_id


def load_qrels(path, min_grade=1):
    """Read positive judgments, normalizing table IDs for matching."""
    relevant = defaultdict(set)
    for line in Path(path).read_text().splitlines():
        if line.strip():
            qid, _, uid, grade = line.split()
            if int(grade) >= min_grade:
                relevant[check_id(qid)].add(normalize_table_id(check_id(uid)))
    return relevant


def evaluate(run_path, qrels_path, query_ids, k=10, min_grade=1):
    """Keep the original recall scope and add precision and binary nDCG.

Grades at or above min_grade are treated as relevant. Queries with no relevant
judgments are reported separately and excluded from the average. Judged queries
missing from the run receive a recall of zero. Table IDs are normalized before
matching and checking for duplicates.
    """
    if k < 1:
        raise ValueError("k must be positive.")
    query_ids = [str(qid) for qid in query_ids]
    if len(set(query_ids)) != len(query_ids):
        raise ValueError("Duplicate query IDs.")
    relevant = load_qrels(qrels_path, min_grade)
    runs = defaultdict(list)
    seen = set()
    allowed = set(query_ids)
    for line in Path(run_path).read_text().splitlines():
        if not line.strip():
            continue
        qid, placeholder, uid, rank, score, _ = line.split()
        uid = normalize_table_id(uid)
        rank, score = int(rank), float(score)
        if qid not in allowed or placeholder != "Q0" or rank < 1 or not math.isfinite(score):
            raise ValueError(f"Invalid TREC line: {line}")
        if (qid, uid) in seen:
            raise ValueError(f"Duplicate query/table pair: {qid}, {uid}")
        seen.add((qid, uid))
        runs[qid].append((rank, score, uid))
    per_query = {}
    unjudged = []
    precisions, ndcgs = [], []
    for qid in query_ids:
        ranked = sorted(runs[qid])
        if [r for r, _, _ in ranked] != list(range(1, len(ranked) + 1)):
            raise ValueError(f"Non-consecutive ranks for query {qid}")
        if any(a[1] < b[1] for a, b in zip(ranked, ranked[1:])):
            raise ValueError(f"Scores increase with rank for query {qid}")
        if not relevant[qid]:
            unjudged.append(qid)
            continue
        retrieved = {uid for _, _, uid in ranked[:k]}
        per_query[qid] = len(retrieved & relevant[qid]) / len(relevant[qid])
        precisions.append(len(retrieved & relevant[qid]) / k)
        ideal = sum(1 / math.log2(rank + 1) for rank in range(1, min(k, len(relevant[qid])) + 1))
        dcg = sum(1 / math.log2(rank + 1) for rank, _, uid in ranked[:k] if uid in relevant[qid])
        ndcgs.append(dcg / ideal)
    return {f"precision@{k}": sum(precisions) / len(precisions) if precisions else None,
            f"ndcg@{k}": sum(ndcgs) / len(ndcgs) if ndcgs else None,
            f"recall@{k}": sum(per_query.values()) / len(per_query) if per_query else None,
            "evaluated_queries": len(per_query), "min_relevance_grade": min_grade,
            "queries_without_positive_judgments": unjudged, "per_query": per_query}
