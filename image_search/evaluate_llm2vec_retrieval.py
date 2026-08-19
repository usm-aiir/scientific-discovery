"""
evaluate_llm2vec_retrieval.py
 
Evaluates LLM2Vec-based semantic retrieval over ArXiv figure captions.
Given a set of benchmark queries (each linked to a known figure), this script
measures how well the model ranks the correct figure against 539k candidates.
 
Metrics reported: Recall@1, Recall@5, Recall@10, Mean Reciprocal Rank (MRR)
 
Usage:
    python evaluate_llm2vec_retrieval.py
"""
 
import os
import csv
import torch
import pickle
import numpy as np
from collections import defaultdict
from llm2vec import LLM2Vec
 
 
# ── Configuration ─────────────────────────────────────────────────────────────
 
CAPTIONS_DIR     = os.path.expanduser("~/arxiv_data/captions")
EMBEDDINGS_CACHE = os.path.expanduser("~/arxiv_data/embeddings.pkl")
EVAL_TSV         = os.path.expanduser("~/claude_figure_queries.tsv")
 
RETRIEVAL_INSTRUCTION = (
    "Given a scientific figure search query, retrieve the most relevant figure caption:"
)
 
TOP_K_VALUES = [1, 5, 10]
 
 
# ── Step 1: Load captions ─────────────────────────────────────────────────────
 
def load_captions(captions_dir):
    """
    Read all caption TSV files from captions_dir.
    Each line is tab-separated: paper_id, fig_id, [subfig,] caption.
    Returns a list of (paper_id, fig_id, caption) tuples.
    """
    records = []
    for fname in os.listdir(captions_dir):
        fpath = os.path.join(captions_dir, fname)
        with open(fpath, "r", encoding="utf-8") as f:
            for line in f:
                parts = line.strip().split("\t")
                if len(parts) >= 3:
                    paper_id = parts[0]
                    fig_id   = parts[1]
                    caption  = parts[-1].strip('"')
                    records.append((paper_id, fig_id, caption))
    return records
 
 
print("Loading captions...")
records = load_captions(CAPTIONS_DIR)
print(f"Loaded {len(records):,} captions")
 
# Build lookup: (paper_id, fig_id) -> list of indices in records
index_map = defaultdict(list)
for i, (pid, fid, _) in enumerate(records):
    index_map[(pid, fid)].append(i)
 
 
# ── Step 2: Load LLM2Vec model ────────────────────────────────────────────────
 
print("\nLoading LLM2Vec model (Llama 3 8B)...")
l2v = LLM2Vec.from_pretrained(
    "McGill-NLP/LLM2Vec-Meta-Llama-3-8B-Instruct-mntp",
    peft_model_name_or_path="McGill-NLP/LLM2Vec-Meta-Llama-3-8B-Instruct-mntp-supervised",
    device_map="cuda" if torch.cuda.is_available() else "cpu",
    torch_dtype=torch.bfloat16,
    attn_implementation="eager",
)
 
 
# ── Step 3: Load or build caption embeddings ──────────────────────────────────
 
if os.path.exists(EMBEDDINGS_CACHE):
    print("\nLoading cached caption embeddings...")
    with open(EMBEDDINGS_CACHE, "rb") as f:
        embeddings = pickle.load(f)
else:
    print("\nEncoding captions — this will take several hours on first run...")
    captions   = [r[2] for r in records]
    embeddings = l2v.encode(captions, batch_size=32, show_progress_bar=True)
    with open(EMBEDDINGS_CACHE, "wb") as f:
        pickle.dump(embeddings, f)
    print("Embeddings saved to cache.")
 
# Convert to normalized numpy array for fast cosine similarity
if hasattr(embeddings, "numpy"):
    emb_np = embeddings.cpu().float().numpy()
else:
    emb_np = np.array(embeddings, dtype=np.float32)
 
norms = np.linalg.norm(emb_np, axis=1, keepdims=True)
norms[norms == 0] = 1
doc_embeddings_normalized = emb_np / norms
 
print(f"Embedding matrix shape: {doc_embeddings_normalized.shape}")
 
 
# ── Step 4: Load evaluation queries ──────────────────────────────────────────
 
print(f"\nLoading evaluation queries from {EVAL_TSV}...")
eval_rows = []
 
with open(EVAL_TSV, "r", encoding="utf-8") as f:
    reader = csv.DictReader(f, delimiter="\t")
    for row in reader:
        paper_id = row["Paper ID"].strip()
        fig_id   = row["Figure ID"].strip()
        for col in ["Query 1", "Query 2", "Query 3"]:
            query = row[col].strip()
            if query:
                eval_rows.append((paper_id, fig_id, query))
 
num_figures = len(set((r[0], r[1]) for r in eval_rows))
print(f"Loaded {len(eval_rows)} queries across {num_figures} figures")
 
 
# ── Step 5: Run retrieval evaluation ─────────────────────────────────────────
 
hits_at        = {k: 0 for k in TOP_K_VALUES}
reciprocal_ranks = []
not_found      = 0
 
print(f"\nEvaluating {len(eval_rows)} queries against {len(records):,} captions...\n")
 
for i, (paper_id, fig_id, query) in enumerate(eval_rows):
    if (i + 1) % 50 == 0:
        print(f"  Progress: {i+1}/{len(eval_rows)}")
 
    gt_indices = set(index_map.get((paper_id, fig_id), []))
    if not gt_indices:
        not_found += 1
        reciprocal_ranks.append(0.0)
        continue
 
    # Encode query with retrieval instruction prefix
    q_emb  = l2v.encode([[RETRIEVAL_INSTRUCTION, query]])
    q_np   = q_emb.cpu().float().numpy()
    q_norm = q_np / np.linalg.norm(q_np, axis=1, keepdims=True)
 
    # Rank all captions by cosine similarity (CPU numpy)
    scores         = (q_norm @ doc_embeddings_normalized.T)[0]
    ranked_indices = np.argsort(scores)[::-1]
 
    # Find the highest-ranked ground truth caption
    best_rank = None
    for rank, idx in enumerate(ranked_indices, start=1):
        if idx in gt_indices:
            best_rank = rank
            break
 
    if best_rank is None:
        reciprocal_ranks.append(0.0)
    else:
        reciprocal_ranks.append(1.0 / best_rank)
        for k in hits_at:
            if best_rank <= k:
                hits_at[k] += 1
 
 
# ── Step 6: Report results ────────────────────────────────────────────────────
 
n = len(eval_rows)
print("\n" + "=" * 50)
print("EVALUATION RESULTS")
print("=" * 50)
print(f"Total queries:          {n}")
print(f"Corpus size:            {len(records):,} captions")
print(f"Queries with no match:  {not_found}")
print()
for k in TOP_K_VALUES:
    print(f"Recall@{k:<3}  {hits_at[k]/n*100:.1f}%  ({hits_at[k]}/{n})")
print(f"MRR:        {np.mean(reciprocal_ranks):.4f}")
print("=" * 50)
 