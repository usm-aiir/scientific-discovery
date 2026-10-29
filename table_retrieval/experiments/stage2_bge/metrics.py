"""Score ranked cells with micro precision/recall, mean binary nDCG, and MRR."""
import math


def ndcg_at_k(ranked, gold, k=5):
    """Use score order and include gold cells outside the Stage 1 candidate pool."""
    if not gold or k < 1:
        raise ValueError('nDCG requires nonempty gold and a positive cutoff.')
    dcg = sum(1 / math.log2(rank + 1) for rank, cell in enumerate(ranked[:k], 1) if cell in gold)
    ideal = sum(1 / math.log2(rank + 1) for rank in range(1, min(k, len(gold)) + 1))
    return dcg / ideal


def score_tables(records):
    """Measure the supplied Stage 1 tables against all gold evidence tables."""
    hits = returned = gold_count = 0
    ndcg = 0.0
    for record in records:
        gold = {cell[0] for cell in record['gold']}
        tables = record['candidate_table_ids']
        hits += len(set(tables) & gold)
        returned += len(tables)
        gold_count += len(gold)
        ideal = sum(1 / math.log2(i + 2) for i in range(min(len(tables), len(gold))))
        ndcg += sum(1 / math.log2(i + 2) for i, table in enumerate(tables) if table in gold) / ideal
    return {
        'table_precision@5': hits / returned,
        'table_recall@5': hits / gold_count,
        'table_ndcg@5': ndcg / len(records),
    }


def score_queries(records, k=5):
    """Include all questions and unavailable gold; MRR uses the full ranking."""
    if not records or k < 1:
        raise ValueError('Scoring requires queries and a positive cutoff.')
    hits = returned = gold_count = available = 0
    ndcg = mrr = 0.0
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
        mrr += next((1 / rank for rank, item in enumerate(ranked, 1) if item in gold), 0.0)
        ndcg += ndcg_at_k(ranked, gold, k)
    precision = hits / returned if returned else 0.0
    recall = hits / gold_count
    return {
        'queries': len(records), 'gold_cells': gold_count, 'correct_cells': hits,
        'returned_cells': returned, f'precision@{k}': precision, f'recall@{k}': recall,
        f'f1@{k}': 2 * precision * recall / (precision + recall) if precision + recall else 0,
        f'ndcg@{k}': ndcg / len(records), 'mrr': mrr / len(records),
        'recall_by_cutoff': {str(cutoff): value / gold_count for cutoff, value in hits_at.items()},
        'stage1': {'gold_cell_coverage': available / gold_count, **score_tables(records)},
    }


def summarize_metrics(scores):
    """Save shared counts and MRR once, with a small row for each cutoff."""
    first = next(iter(scores.values()))
    summary = {key: first[key] for key in ('queries', 'gold_cells', 'mrr', 'stage1')}
    summary['cutoffs'] = {}
    for cutoff, values in scores.items():
        row = {name: values[f'{name}@{cutoff}'] for name in ('precision', 'recall', 'f1', 'ndcg')}
        row.update(correct_cells=values['correct_cells'], returned_cells=values['returned_cells'])
        summary['cutoffs'][str(cutoff)] = row
    return summary
