#!/usr/bin/env python3
"""
evaluate_clip_retrieval.py
==========================
Evaluates CLIP image retrieval against the 600 ground-truth queries in
claude_figure_queries.tsv (200 figures x 3 queries each).
 
For each query, the script encodes the query text with CLIP's text encoder,
computes cosine similarity against the full image embedding matrix, and checks
whether the target figure appears in the top-K results.
 
Metrics reported
----------------
  Recall@1   -- fraction of queries where the correct figure is rank 1
  Recall@5   -- fraction of queries where the correct figure is in top 5
  Recall@10  -- fraction of queries where the correct figure is in top 10
  MRR        -- Mean Reciprocal Rank across all matched queries
 
Usage
-----
    python evaluate_clip_retrieval.py
    python evaluate_clip_retrieval.py --queries_tsv claude_figure_queries.tsv
    python evaluate_clip_retrieval.py --cache clip_embeddings.pkl --top_k 20
 
Example
-------
    python evaluate_clip_retrieval.py \\
        --queries_tsv claude_figure_queries.tsv \\
        --cache clip_embeddings.pkl \\
        --model_name openai/clip-vit-large-patch14
"""
 
import argparse
import pickle
 
import numpy as np
import pandas as pd
import torch
from transformers import CLIPModel, CLIPTokenizer
 
# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
 
DEFAULT_QUERIES_TSV = "claude_figure_queries.tsv"
DEFAULT_CACHE       = "clip_embeddings.pkl"
DEFAULT_MODEL       = "openai/clip-vit-large-patch14"
DEFAULT_TOP_K       = 10
 
QUERY_COLUMNS = ["Query 1", "Query 2", "Query 3"]
 
 
# ---------------------------------------------------------------------------
# Argument parsing
# ---------------------------------------------------------------------------
 
def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(
        description="Evaluate CLIP text-to-image retrieval over sampled arXiv figures.",
        epilog="Example:\n  python evaluate_clip_retrieval.py --queries_tsv claude_figure_queries.tsv",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    ap.add_argument(
        "--queries_tsv", default=DEFAULT_QUERIES_TSV,
        help=f"TSV with 200 figures and 3 queries each (default: {DEFAULT_QUERIES_TSV})",
    )
    ap.add_argument(
        "--cache", default=DEFAULT_CACHE,
        help=f"Pickle cache of CLIP image embeddings from clip_search.py (default: {DEFAULT_CACHE})",
    )
    ap.add_argument(
        "--model_name", default=DEFAULT_MODEL,
        help=f"CLIP model name (default: {DEFAULT_MODEL})",
    )
    ap.add_argument(
        "--top_k", type=int, default=DEFAULT_TOP_K,
        help=f"Maximum K for Recall@K reporting (default: {DEFAULT_TOP_K})",
    )
    return ap.parse_args()
 
 
# ---------------------------------------------------------------------------
# Load embeddings cache
# ---------------------------------------------------------------------------
 
def load_cache(cache_path: str) -> tuple[list, np.ndarray]:
    """
    Load the CLIP image embeddings cache written by clip_search.py.
 
    Returns (image_ids, embedding_matrix) where:
      - image_ids is a list of (paper_id, fig_id) tuples
      - embedding_matrix is a float32 numpy array of shape (N, D)
    """
    print(f"Loading embedding cache from {cache_path} ...")
    with open(cache_path, "rb") as f:
        cache = pickle.load(f)
 
    # Support both dict-style and tuple-style cache formats.
    if isinstance(cache, dict):
        if "image_ids" in cache:
            image_ids = cache["image_ids"]
        elif "records" in cache:
            image_ids = cache["records"]
        else:
            raise KeyError(f"Cannot find image ID key in cache. Keys: {list(cache.keys())}")
        embedding_matrix = np.array(cache["embeddings"], dtype=np.float32)
    elif isinstance(cache, tuple) and len(cache) == 2:
        image_ids, embedding_matrix = cache
        embedding_matrix = np.array(embedding_matrix, dtype=np.float32)
    else:
        raise ValueError(
            f"Unrecognized cache format: {type(cache)}. "
            "Expected a dict with 'image_ids'/'embeddings' keys, "
            "or a (image_ids, embeddings) tuple."
        )
 
    # Records and embeddings may not align 1:1 if some images failed to encode.
    # Trim image_ids to the number of actual embedding rows.
    n_records    = len(image_ids)
    n_embeddings = embedding_matrix.shape[0]
    if n_records != n_embeddings:
        print(
            f"  WARNING: {n_records:,} records but {n_embeddings:,} embeddings "
            f"({n_records - n_embeddings:,} images skipped due to encoding failures)."
        )
        image_ids = image_ids[:n_embeddings]

    print(f"Loaded {len(image_ids):,} image embeddings, dim={embedding_matrix.shape[1]}")
    return image_ids, embedding_matrix
 
 
# ---------------------------------------------------------------------------
# Build lookup: (paper_id, fig_id) -> row index in embedding matrix
# ---------------------------------------------------------------------------
 
def build_lookup(image_ids: list) -> dict[tuple, int]:
    """
    Build a mapping from (paper_id, fig_id) to the row index in the
    embedding matrix. Handles both string paths and tuple image IDs.
    """
    lookup = {}
    for i, img_id in enumerate(image_ids):
        if isinstance(img_id, (tuple, list)) and len(img_id) >= 2:
            key = (str(img_id[0]), str(img_id[1]))
        elif isinstance(img_id, str):
            # img_id might be a file path like figures/24/10/2410.00003/1a.png
            import os
            parts   = img_id.replace("\\", "/").split("/")
            fig_stem = os.path.splitext(parts[-1])[0]   # e.g. "1a"
            paper_id = parts[-2] if len(parts) >= 2 else ""
            key = (paper_id, fig_stem)
        else:
            key = (str(img_id), "")
        lookup[key] = i
    return lookup
 
 
# ---------------------------------------------------------------------------
# Encode queries with CLIP text encoder
# ---------------------------------------------------------------------------
 
def encode_queries(queries: list[str], model: CLIPModel,
                   tokenizer: CLIPTokenizer, device: str) -> np.ndarray:
    """
    Encode a list of query strings with CLIP's text encoder.
    Returns a float32 numpy array of shape (N, D), L2-normalised.
    """
    model.eval()
    all_embeddings = []
 
    batch_size = 64
    for i in range(0, len(queries), batch_size):
        batch = queries[i : i + batch_size]
        inputs = tokenizer(
            batch,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=77,
        ).to(device)
 
        with torch.no_grad():
            text_features = model.get_text_features(**inputs)
 
        text_features = text_features / text_features.norm(dim=-1, keepdim=True)
        all_embeddings.append(text_features.cpu().numpy())
 
    return np.vstack(all_embeddings).astype(np.float32)
 
 
# ---------------------------------------------------------------------------
# Evaluation
# ---------------------------------------------------------------------------
 
def evaluate(
    queries_tsv: str,
    image_ids: list,
    embedding_matrix: np.ndarray,
    model: CLIPModel,
    tokenizer: CLIPTokenizer,
    device: str,
    top_k: int,
) -> None:
    """
    For each (figure, query) pair in the TSV, compute cosine similarity
    against the full image embedding matrix and report retrieval metrics.
    """
    df = pd.read_csv(queries_tsv, sep="\t", dtype=str).fillna("")
    print(f"Loaded {len(df)} figures from {queries_tsv}")
 
    lookup = build_lookup(image_ids)
 
    # Normalise the embedding matrix once (for cosine similarity via dot product).
    norms  = np.linalg.norm(embedding_matrix, axis=1, keepdims=True)
    norms[norms == 0] = 1.0
    normed_matrix = embedding_matrix / norms
 
    # Collect all (query_text, paper_id, fig_id) triples.
    triples = []
    for _, row in df.iterrows():
        paper_id = str(row.get("Paper ID", "")).strip()
        fig_id   = str(row.get("Figure ID", "")).strip()
        for col in QUERY_COLUMNS:
            q = str(row.get(col, "")).strip()
            if q:
                triples.append((q, paper_id, fig_id))
 
    print(f"Evaluating {len(triples)} queries ({len(df)} figures x {len(QUERY_COLUMNS)} queries) ...")
 
    # Encode all queries at once.
    query_texts      = [t[0] for t in triples]
    query_embeddings = encode_queries(query_texts, model, tokenizer, device)
 
    # Score each query.
    recall_hits   = {k: 0 for k in [1, 5, top_k]}
    reciprocal_ranks = []
    n_no_match    = 0
 
    for i, (query, paper_id, fig_id) in enumerate(triples):
        # Find the target figure's index in the embedding matrix.
        target_idx = lookup.get((paper_id, fig_id))
 
        if target_idx is None:
            # Try stripping sub-figure suffix (e.g. "1a" -> "1")
            fig_id_base = fig_id.rstrip("abcdefghij")
            target_idx  = lookup.get((paper_id, fig_id_base))
 
        if target_idx is None:
            n_no_match += 1
            continue
 
        # Cosine similarities (dot product on L2-normalised vectors).
        q_vec  = query_embeddings[i]
        q_vec  = q_vec / (np.linalg.norm(q_vec) + 1e-9)
        scores = normed_matrix @ q_vec
 
        # Rank of the target (1-indexed).
        rank = int((scores > scores[target_idx]).sum()) + 1
 
        for k in [1, 5, top_k]:
            if rank <= k:
                recall_hits[k] += 1
        reciprocal_ranks.append(1.0 / rank)
 
    n_evaluated = len(triples) - n_no_match
 
    print("\n" + "=" * 50)
    print("CLIP Retrieval Evaluation Results")
    print("=" * 50)
    print(f"Total queries:        {len(triples)}")
    print(f"No match in corpus:   {n_no_match}")
    print(f"Evaluated:            {n_evaluated}")
    print()
    for k in [1, 5, top_k]:
        recall = recall_hits[k] / n_evaluated if n_evaluated else 0.0
        print(f"  Recall@{k:<3}:         {recall:.1%}")
    mrr = float(np.mean(reciprocal_ranks)) if reciprocal_ranks else 0.0
    print(f"  MRR:                {mrr:.4f}")
    print("=" * 50)
 
 
# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
 
def main() -> None:
    args   = parse_args()
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")
 
    image_ids, embedding_matrix = load_cache(args.cache)
 
    print(f"Loading CLIP model {args.model_name} ...")
    tokenizer = CLIPTokenizer.from_pretrained(args.model_name)
    model     = CLIPModel.from_pretrained(args.model_name).to(device)
 
    evaluate(
        queries_tsv      = args.queries_tsv,
        image_ids        = image_ids,
        embedding_matrix = embedding_matrix,
        model            = model,
        tokenizer        = tokenizer,
        device           = device,
        top_k            = args.top_k,
    )
 
 
if __name__ == "__main__":
    main()
 