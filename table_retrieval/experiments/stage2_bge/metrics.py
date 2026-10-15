# Scores ranked evidence with micro precision/recall and mean binary nDCG.
import math


# The ranking must be in descending model-score order, never sorted by cell ID.
def ndcg_at_k(ranked, gold, k=5):
    """Binary nDCG for one query with at least one annotated gold cell.

    DCG@k = sum(1/log2(rank+1) for gold cells at score ranks 1..k).
    IDCG@k = sum(1/log2(rank+1) for ranks 1..min(k, number of gold cells)).
    Gold includes evidence outside the Stage 1 pool. Zero retrieved gold yields
    zero, not exclusion. The caller averages all scorable queries equally.
    """
    if not gold or k < 1:
        raise ValueError('nDCG requires nonempty gold and a positive cutoff.')
    dcg = sum(1 / math.log2(rank + 1) for rank, cell in enumerate(ranked[:k], 1) if cell in gold)
    ideal = sum(1 / math.log2(rank + 1) for rank in range(1, min(k, len(gold)) + 1))
    return dcg / ideal


# Gold includes cells absent from Stage 1, so the end-to-end denominator is preserved.
def score_queries(records, k=5):
    """Score the same queries/labels for micro P/R/F1 and mean binary nDCG@k.

    Every supplied query is included, including zero-hit queries. Each query
    must have gold labels; fixed test evaluation validates all 91 queries first.
    """
    if not records or k < 1:
        raise ValueError('Scoring requires queries and a positive cutoff.')
    hits = returned = gold_count = available = table_hits = table_gold = table_count = 0
    ndcg = table_ndcg = mrr = 0.0
    hits_at = {1: 0, 3: 0, 5: 0}
    for record in records:
        gold = {tuple(item) for item in record['gold']}
        ranked = [tuple(item) for item in record['ranking']]
        if not gold or len(ranked) != len(set(ranked)):
            raise ValueError('Gold must be nonempty and rankings must have unique cell IDs.')
        hits += len(set(ranked[:k]) & gold)
        returned += len(ranked[:k])
        gold_count += len(gold)
        available += len(set(ranked) & gold)
        for cutoff in hits_at:
            hits_at[cutoff] += len(set(ranked[:cutoff]) & gold)
        ranks = [rank for rank, item in enumerate(ranked, 1) if item in gold]
        mrr += 1 / ranks[0] if ranks else 0
        ndcg += ndcg_at_k(ranked, gold, k)
        gold_tables = {item[0] for item in gold}
        tables = record['candidate_table_ids']
        table_hits += len(set(tables) & gold_tables)
        table_gold += len(gold_tables)
        table_count += len(tables)
        table_ideal = sum(1 / math.log2(i + 2) for i in range(min(len(tables), len(gold_tables))))
        table_ndcg += sum(1 / math.log2(i + 2) for i, table in enumerate(tables)
                          if table in gold_tables) / table_ideal
    precision = hits / returned if returned else 0.0
    recall = hits / gold_count
    return {'queries': len(records), 'gold_cells': gold_count, 'correct_cells': hits,
            'returned_cells': returned, f'precision@{k}': precision, f'recall@{k}': recall,
            f'f1@{k}': 2 * precision * recall / (precision + recall) if precision + recall else 0,
            f'ndcg@{k}': ndcg / len(records), 'mrr': mrr / len(records),
            'recall_by_cutoff': {str(cutoff): value / gold_count for cutoff, value in hits_at.items()},
            'stage1': {'gold_cell_coverage': available / gold_count,
                       'table_precision@5': table_hits / table_count,
                       'table_recall@5': table_hits / table_gold,
                       'table_ndcg@5': table_ndcg / len(records)}}


# Epoch selection uses full-ranking nDCG within each supplied validation table.
def validation_ndcg(rankings):
    values = []
    for ranked, gold in rankings:
        ideal = sum(1 / math.log2(i + 2) for i in range(len(gold)))
        dcg = sum(1 / math.log2(i + 2) for i, cell_id in enumerate(ranked) if cell_id in gold)
        values.append(dcg / ideal)
    return sum(values) / len(values)
