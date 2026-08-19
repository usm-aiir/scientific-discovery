import os
import csv
import torch
import pickle
import numpy as np
from llm2vec import LLM2Vec
 
CAPTIONS_DIR = os.path.expanduser("~/arxiv_data/captions")
EMBEDDINGS_CACHE = os.path.expanduser("~/arxiv_data/embeddings.pkl")
EVAL_TSV = os.path.expanduser("~/claude_figure_queries.tsv")
 
# ── 1. Load captions ──────────────────────────────────────────────────────────
#Loads all the captions and reads all of the files and builds dictionary list
def load_captions(captions_dir):
    records = []
    for fname in os.listdir(captions_dir):
        fpath = os.path.join(captions_dir, fname)
        with open(fpath, "r", encoding="utf-8") as f:
            for line in f:
                parts = line.strip().split("\t")
                if len(parts) >= 3:
                    paper_id = parts[0]
                    fig_num  = parts[1]
                    caption  = parts[-1].strip('"')
                    records.append((paper_id, fig_num, caption))
    return records
 
print("Loading captions...")
records = load_captions(CAPTIONS_DIR)
print(f"Loaded {len(records)} captions")
 
# Build a lookup: (paper_id, fig_id) -> list of indices in records
from collections import defaultdict
index_map = defaultdict(list)
for i, (pid, fid, _) in enumerate(records):
    index_map[(pid, fid)].append(i)
 
# ── 2. Load model ─────────────────────────────────────────────────────────────
#Load llama model 
print("Loading LLM2Vec model...")
l2v = LLM2Vec.from_pretrained(
    "McGill-NLP/LLM2Vec-Meta-Llama-3-8B-Instruct-mntp",
    peft_model_name_or_path="McGill-NLP/LLM2Vec-Meta-Llama-3-8B-Instruct-mntp-supervised",
    device_map="cuda" if torch.cuda.is_available() else "cpu",
    torch_dtype=torch.bfloat16,
    attn_implementation="eager",
)
 
# ── 3. Load or build embeddings cache ─────────────────────────────────────────
#reads sample tsv and extracts 3 queries from each row and stores as paper_id figure_id and query all together

if os.path.exists(EMBEDDINGS_CACHE):
    print("Loading cached embeddings...")
    with open(EMBEDDINGS_CACHE, "rb") as f:
        embeddings = pickle.load(f)
else:
    print("Encoding captions (this will take a while)...")
    captions = [r[2] for r in records]
    embeddings = l2v.encode(captions, batch_size=32, show_progress_bar=True)
    with open(EMBEDDINGS_CACHE, "wb") as f:
        pickle.dump(embeddings, f)
    print("Embeddings cached!")
 
# Normalize all doc embeddings once
if hasattr(embeddings, 'numpy'):
    emb_np = embeddings.cpu().float().numpy()
else:
    emb_np = np.array(embeddings, dtype=np.float32)
 
norms = np.linalg.norm(emb_np, axis=1, keepdims=True)
norms[norms == 0] = 1
d_norm = emb_np / norms
 
print(f"Embeddings shape: {d_norm.shape}")
 
# ── 4. Load evaluation queries ────────────────────────────────────────────────

print(f"\nLoading eval queries from {EVAL_TSV}...")
eval_rows = []
with open(EVAL_TSV, "r", encoding="utf-8") as f:
    reader = csv.DictReader(f, delimiter="\t")
    for row in reader:
        paper_id = row["Paper ID"].strip()
        fig_id   = row["Figure ID"].strip()
        for q_col in ["Query 1", "Query 2", "Query 3"]:
            q = row[q_col].strip()
            if q:
                eval_rows.append((paper_id, fig_id, q))
 
print(f"Loaded {len(eval_rows)} queries across {len(set((r[0],r[1]) for r in eval_rows))} figures")
 
# ── 5. Run evaluation ─────────────────────────────────────────────────────────
instruction = "Given a scientific figure search query, retrieve the most relevant figure caption:"
 
hits_at = {1: 0, 5: 0, 10: 0}
reciprocal_ranks = []
not_found = 0
 
print(f"\nRunning evaluation on {len(eval_rows)} queries...\n")
 
for i, (paper_id, fig_id, query) in enumerate(eval_rows):
    if (i+1) % 50 == 0:
        print(f"  Progress: {i+1}/{len(eval_rows)}")
 
    # Get ground truth indices
    gt_indices = set(index_map.get((paper_id, fig_id), []))
    if not gt_indices:
        not_found += 1
        reciprocal_ranks.append(0.0)
        continue
 
    # Encode query
    q_emb = l2v.encode([[instruction, query]])
    q_np  = q_emb.cpu().float().numpy()
    q_norm = q_np / np.linalg.norm(q_np, axis=1, keepdims=True)
 
    # Cosine similarity
    scores = (q_norm @ d_norm.T)[0]
 
    # Rank all indices (highest score first)
    ranked_indices = np.argsort(scores)[::-1]
 
    # Find rank of best ground truth hit
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
 
# ── 6. Report results ─────────────────────────────────────────────────────────
n = len(eval_rows)
print("\n" + "="*50)
print("EVALUATION RESULTS")
print("="*50)
print(f"Total queries:          {n}")
print(f"Queries with no match:  {not_found}")
print(f"")
print(f"Recall@1:   {hits_at[1]/n*100:.1f}%  ({hits_at[1]}/{n})")
print(f"Recall@5:   {hits_at[5]/n*100:.1f}%  ({hits_at[5]}/{n})")
print(f"Recall@10:  {hits_at[10]/n*100:.1f}%  ({hits_at[10]}/{n})")
print(f"MRR:        {np.mean(reciprocal_ranks):.4f}")
print("="*50)
 


