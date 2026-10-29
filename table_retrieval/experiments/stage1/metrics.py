"""Compare retrieved tables with relevance labels to measure search quality.

Average precision, recall, nDCG, and reciprocal rank across judged questions.
"""
from collections import defaultdict
import math
from pathlib import Path

from table_retrieval.data import check_id, normalize_table_id


def load_qrels(path, min_grade=1):
    """Load the table IDs labeled as relevant to each question."""
    relevant = defaultdict(set)
    for line in Path(path).read_text().splitlines():
        if not line.strip():
            continue
        qid, _, uid, grade = line.split()

        # Keep labels meeting the minimum relevance grade.
        # Standardize table IDs so they can match the search results.
        if int(grade) >= min_grade:
            relevant[check_id(qid)].add(normalize_table_id(check_id(uid)))
    return relevant


def load_rankings(path, query_ids):
    """Load saved search results and check their IDs, ranks, and scores."""
    runs = defaultdict(list)
    seen = set()
    allowed = set(query_ids)

    for line in Path(path).read_text().splitlines():
        if not line.strip():
            continue

        # TREC format: question ID, placeholder, table ID, rank, score, run name.
        qid, placeholder, uid, rank, score, _ = line.split()
        uid = normalize_table_id(uid)
        rank, score = int(rank), float(score)

        # Reject unknown questions, invalid ranks, and invalid scores.
        # Q0 is the required placeholder in this results format.
        if qid not in allowed or placeholder != 'Q0' or rank < 1 or not math.isfinite(score):
            raise ValueError(f'Invalid TREC line: {line}')

        # A table must not appear twice for the same question.
        if (qid, uid) in seen:
            raise ValueError(f'Duplicate query/table pair: {qid}, {uid}')
        seen.add((qid, uid))
        runs[qid].append((rank, score, uid))

    for qid, hits in runs.items():
        hits.sort()

        # Ranks must start at 1 and continue without gaps or repeated positions.
        if [rank for rank, _, _ in hits] != list(range(1, len(hits) + 1)):
            raise ValueError(f'Non-consecutive ranks for query {qid}')

        # Scores must stay the same or decrease as rank positions increase.
        if any(first[1] < second[1] for first, second in zip(hits, hits[1:])):
            raise ValueError(f'Scores increase with rank for query {qid}')
    return runs


def mean(values):
    """Calculate the average, or return None if there are no values."""
    return sum(values) / len(values) if values else None


def evaluate(run_path, qrels_path, query_ids, k=10, min_grade=1):
    """Calculate aggregate metrics over questions with positive judgments.

    Grades at or above min_grade count as relevant, with equal importance.
    Questions with relevant labels but no results score zero.
    Questions without relevant labels are listed separately and excluded.
    MRR uses the full saved ranking; precision, recall, and nDCG use the first k.
    """
    if k < 1:
        raise ValueError('k must be positive.')
    query_ids = [str(qid) for qid in query_ids]
    if len(set(query_ids)) != len(query_ids):
        raise ValueError('Duplicate query IDs.')

    relevant = load_qrels(qrels_path, min_grade)
    runs = load_rankings(run_path, query_ids)
    unjudged = []
    precisions, recalls, ndcgs, reciprocal_ranks = [], [], [], []

    for qid in query_ids:
        gold = relevant[qid]

        # Skip questions that have no tables labeled as relevant.
        if not gold:
            unjudged.append(qid)
            continue

        # Count relevant tables found among the first k results.
        ranked = runs[qid][:k]
        retrieved = {uid for _, _, uid in ranked}
        matches = len(retrieved & gold)

        # Recall: what fraction of the known relevant tables did we find?
        recalls.append(matches / len(gold))

        # Precision: how many of the requested k results were relevant?
        # Missing results still count toward the denominator.
        precisions.append(matches / k)

        # Calculate the best possible ranking score for this question.
        ideal = sum(1 / math.log2(rank + 1) for rank in range(1, min(k, len(gold)) + 1))

        # Give more credit to relevant tables appearing near the top.
        dcg = sum(1 / math.log2(rank + 1) for rank, _, uid in ranked if uid in gold)

        # nDCG compares the actual ranking score with the best possible score.
        ndcgs.append(dcg / ideal)
        reciprocal_ranks.append(next((1 / rank for rank, _, uid in runs[qid] if uid in gold), 0.0))

    # Average each metric across evaluated questions.
    return {
        f'precision@{k}': mean(precisions),
        f'recall@{k}': mean(recalls),
        f'ndcg@{k}': mean(ndcgs),
        'mrr': mean(reciprocal_ranks),
        'evaluated_queries': len(recalls),
        'min_relevance_grade': min_grade,
        'queries_without_positive_judgments': unjudged,
    }
