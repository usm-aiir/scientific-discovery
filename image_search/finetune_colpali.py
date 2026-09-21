import os
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

from colpali_engine.models import ColQwen2_5, ColQwen2_5_Processor
from PIL import Image
from pathlib import Path
import json, torch, torch.nn.functional as F
from torch.optim import AdamW
from torch.utils.data import Dataset, DataLoader

FIGURE_DIR = "/mnt/netstore1_home/behrooz.mansouri/SIGIRSciDis/25_04/figures/images"
DATA_DIR   = "/mnt/netstore1_home/behrooz.mansouri/SIGIRSciDis/figureGen/figure_query_output"
OUTPUT_DIR = "/home/adah.holt/scientific-discovery/colpali-finetuned"
MODEL_NAME = "vidore/colqwen2.5-v0.2"
BATCH_SIZE = 4
EPOCHS     = 1
LR         = 1e-5

def uid_to_path(uid):
    parts = uid.split("::")
    stem = f"{parts[0]}_{parts[-1].lstrip('F')}"
    return str(Path(FIGURE_DIR) / f"{stem}.png")

def load_pairs(json_path):
    with open(json_path, encoding="utf-8") as f:
        entries = json.load(f)
    pairs = []
    for entry in entries:
        for uid in entry["source_figure_uids"]:
            p = uid_to_path(uid)
            if not Path(p).exists():
                continue
            try:
                Image.open(p).convert("RGB")  # validate
                pairs.append((entry["query"], p))
            except Exception:
                continue
    print(f"Loaded {len(pairs)} pairs from {json_path}")
    return pairs

class PairDataset(Dataset):
    def __init__(self, pairs): self.pairs = pairs
    def __len__(self): return len(self.pairs)
    def __getitem__(self, i): return self.pairs[i]  # (query_str, img_path)

print("Loading pairs...")
train_pairs = load_pairs(f"{DATA_DIR}/Train.json")
val_pairs   = load_pairs(f"{DATA_DIR}/Val.json")

train_loader = DataLoader(PairDataset(train_pairs), batch_size=BATCH_SIZE, shuffle=True)
val_loader   = DataLoader(PairDataset(val_pairs),   batch_size=BATCH_SIZE, shuffle=False)

print("Loading model...")
model     = ColQwen2_5.from_pretrained(MODEL_NAME, torch_dtype=torch.bfloat16, device_map="cuda")
processor = ColQwen2_5_Processor.from_pretrained(MODEL_NAME)
optimizer = AdamW(model.parameters(), lr=LR)

def maxsim_scores(q_emb, d_emb):
    # q_emb: [B, Lq, D]  d_emb: [B, Ld, D]
    # returns [B, B] score matrix
    scores = torch.einsum("bnd,cmd->bcnm", q_emb, d_emb)  # [B, B, Lq, Ld]
    return scores.max(dim=-1).values.sum(dim=-1)            # [B, B]

def contrastive_loss(scores):
    # diagonal = correct pairs, rest = negatives
    labels = torch.arange(scores.size(0), device=scores.device)
    return F.cross_entropy(scores, labels)

def run_epoch(loader, train=True):
    model.train() if train else model.eval()
    total_loss, steps = 0, 0
    for queries, img_paths in loader:
        images = [Image.open(p).convert("RGB") for p in img_paths]
        q_inputs = processor.process_queries(list(queries)).to("cuda")
        d_inputs = processor.process_images(images).to("cuda")
        with torch.set_grad_enabled(train):
            q_emb = model(**q_inputs).to(torch.float32)
            d_emb = model(**d_inputs).to(torch.float32)
            scores = maxsim_scores(q_emb, d_emb)
            loss   = contrastive_loss(scores)
        if train:
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        total_loss += loss.item()
        steps += 1
        if steps % 50 == 0:
            print(f"  step {steps}, loss={total_loss/steps:.4f}")
    return total_loss / steps

for epoch in range(EPOCHS):
    print(f"\n=== Epoch {epoch+1}/{EPOCHS} ===")
    train_loss = run_epoch(train_loader, train=True)
    print(f"Train loss: {train_loss:.4f}")
    with torch.no_grad():
        val_loss = run_epoch(val_loader, train=False)
    print(f"Val loss:   {val_loss:.4f}")

os.makedirs(OUTPUT_DIR, exist_ok=True)
model.save_pretrained(OUTPUT_DIR)
processor.save_pretrained(OUTPUT_DIR)
print(f"\nDone! Model saved to {OUTPUT_DIR}")