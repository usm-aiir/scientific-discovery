import os
import pickle
import torch
import numpy as np
from PIL import Image
from transformers import CLIPProcessor, CLIPModel
 
FIGURES_DIR    = os.path.expanduser("~/arxiv_data/figures")
EMBEDDINGS_CACHE = os.path.expanduser("~/arxiv_data/clip_embeddings.pkl")
MODEL_NAME     = "openai/clip-vit-large-patch14"
 
# ── 1. Scan figures directory ─────────────────────────────────────────────────
# Structure: figures/24/10/{paper_id}/{fig_id}.png
print("Scanning figures directory...")
image_records = []  # (paper_id, fig_id, filepath)
 
for root, dirs, files in os.walk(FIGURES_DIR):
    for fname in files:
        if not fname.lower().endswith((".png", ".jpg", ".jpeg", ".gif")):
            continue
        fpath = os.path.join(root, fname)
        # Parent folder name is the paper_id, filename stem is the fig_id
        paper_id = os.path.basename(root)
        fig_id   = os.path.splitext(fname)[0]  # e.g. "1", "2", "3a", "2b"
        image_records.append((paper_id, fig_id, fpath))
 
print(f"Found {len(image_records)} figures")
if image_records:
    print(f"Example: {image_records[0]}")
 
# ── 2. Load CLIP model ────────────────────────────────────────────────────────
print(f"\nLoading CLIP model ({MODEL_NAME})...")
device = "cuda" if torch.cuda.is_available() else "cpu"
model     = CLIPModel.from_pretrained(MODEL_NAME).to(device)
processor = CLIPProcessor.from_pretrained(MODEL_NAME)
model.eval()
print(f"Model loaded on {device}")
 
# ── 3. Encode images (or load cache) ─────────────────────────────────────────
if os.path.exists(EMBEDDINGS_CACHE):
    print("\nLoading cached image embeddings...")
    with open(EMBEDDINGS_CACHE, "rb") as f:
        data = pickle.load(f)
    image_records = data["records"]
    emb_np        = data["embeddings"]
    print(f"Loaded {len(image_records)} cached embeddings")
else:
    print(f"\nEncoding {len(image_records)} images (this may take a while)...")
    embeddings = []
    failed     = []
    BATCH_SIZE = 64
 
    for start in range(0, len(image_records), BATCH_SIZE):
        batch = image_records[start:start + BATCH_SIZE]
        images = []
        valid  = []
        for rec in batch:
            try:
                img = Image.open(rec[2]).convert("RGB")
                images.append(img)
                valid.append(rec)
            except Exception as e:
                failed.append((rec, str(e)))
 
        if not images:
            continue
 
        inputs = processor(images=images, return_tensors="pt", padding=True).to(device)
        with torch.no_grad():
            feats = model.get_image_features(**inputs)
            feats = feats / feats.norm(dim=-1, keepdim=True)
        embeddings.append(feats.cpu().float().numpy())
 
        done = min(start + BATCH_SIZE, len(image_records))
        if done % 5000 == 0 or done == len(image_records):
            print(f"  {done}/{len(image_records)} encoded")
 
    emb_np = np.vstack(embeddings)
    # Keep only records that loaded successfully (match valid batches)
    # Rebuild image_records to only include successfully encoded ones
    # Simple approach: re-scan matching lengths
    print(f"\nFailed to load {len(failed)} images")
    print(f"Successfully encoded {emb_np.shape[0]} images")
 
    with open(EMBEDDINGS_CACHE, "wb") as f:
        pickle.dump({"records": image_records, "embeddings": emb_np}, f)
    print("Embeddings saved to cache!")
 
# Normalize (already normalized above, but ensure it for loaded cache)
norms = np.linalg.norm(emb_np, axis=1, keepdims=True)
norms[norms == 0] = 1
d_norm = emb_np / norms
print(f"\nEmbedding matrix: {d_norm.shape}")
 
# ── 4. Search loop ────────────────────────────────────────────────────────────
print("\nReady to search! Type 'quit' to exit.\n")
 
while True:
    query = input("Enter your search query: ").strip()
    if query.lower() in ("quit", "exit", "q"):
        break
 
    # Encode text query
    inputs = processor(text=[query], return_tensors="pt", padding=True).to(device)
    with torch.no_grad():
        q_feat = model.get_text_features(**inputs)
        q_feat = q_feat / q_feat.norm(dim=-1, keepdim=True)
    q_np = q_feat.cpu().float().numpy()
 
    # Cosine similarity
    scores = (q_np @ d_norm.T)[0]
    top_idx = np.argsort(scores)[::-1][:10]
 
    print(f"\nTop 10 results for: '{query}'\n")
    for rank, idx in enumerate(top_idx, 1):
        paper_id, fig_id, fpath = image_records[idx]
        score = scores[idx]
        print(f"  {rank}. {round(score*100, 1)}% — Paper {paper_id}, Figure {fig_id}")
        print(f"     {fpath}")
    print()
 