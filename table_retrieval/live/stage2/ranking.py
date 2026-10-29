# Sorts Stage 2 cells by BGE score and assigns global ranks.

# Sorts cells from highest to lowest BGE score and adds their global rank.
def rank_scored_cells(cells, scores):
    """Rank globally; stable sorting preserves candidate and cell order on ties."""
    import torch
    if scores.ndim != 1 or len(scores) != len(cells):
        raise ValueError('Expected one BGE score per candidate cell.')
    if not torch.isfinite(scores).all():
        raise ValueError('BGE produced non-finite cell scores.')
    order = torch.argsort(scores, descending=True, stable=True).tolist()
    return [dict(cells[index], stage2_bge_score=float(scores[index]), global_rank=rank)
            for rank, index in enumerate(order, 1)]
