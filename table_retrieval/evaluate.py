# Objective: Validate a TREC run and calculate mean Recall@K against table qrels.

import argparse
from collections import defaultdict
import json
import math
from pathlib import Path
import re

from .data import DATA, load_json


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


def evaluate(run_path, qrels_path, query_ids, k=10, min_grade=1):
    """Score all requested queries, including queries absent from the run.

    Grades >= min_grade count as relevant. Queries without positive judgments
    are reported separately and excluded from the mean. Missing run entries
    for judged queries receive zero recall. Corpus and qrels table ID formats
    are normalized in memory before matching and duplicate detection.
    """
    if k < 1:
        raise ValueError("k must be positive.")
    query_ids = [str(qid) for qid in query_ids]
    if len(set(query_ids)) != len(query_ids):
        raise ValueError("Duplicate query IDs.")
    relevant = defaultdict(set)
    for line in Path(qrels_path).read_text().splitlines():
        if line.strip():
            qid, _, uid, grade = line.split()
            if int(grade) >= min_grade:
                relevant[qid].add(normalize_table_id(uid))
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
    return {f"recall@{k}": sum(per_query.values()) / len(per_query) if per_query else None,
            "evaluated_queries": len(per_query), "min_relevance_grade": min_grade,
            "queries_without_positive_judgments": unjudged, "per_query": per_query}


def main():
    """Evaluate a run using the full query list as the evaluation population."""
    parser = argparse.ArgumentParser(description="Evaluate TREC table rankings")
    parser.add_argument("--run", type=Path, default=Path("results/bm25_val.run"))
    parser.add_argument("--qrels", type=Path, default=DATA / "Val_table_qrels.tsv")
    parser.add_argument("--queries", type=Path, default=DATA / "Val.json")
    parser.add_argument("--k", type=int, default=10)
    parser.add_argument("--min-grade", type=int, default=1)
    parser.add_argument("--output", type=Path, default=Path("results/bm25_val.metrics.json"))
    args = parser.parse_args()
    report = evaluate(args.run, args.qrels,
                      [q["query_id"] for q in load_json(args.queries)], args.k, args.min_grade)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps({k: v for k, v in report.items() if k != "per_query"}, indent=2))


if __name__ == "__main__":
    main()
