import os
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

from colpali_engine.models import ColQwen2_5, ColQwen2_5_Processor
from colpali_engine.trainer.contrastive_trainer import ContrastiveTrainer
from colpali_engine.loss.colpali_losses import ColPaliLoss
from transformers import TrainingArguments
from torch.utils.data import Dataset as TorchDataset
from PIL import Image
from pathlib import Path
import json

FIGURE_DIR = "/mnt/netstore1_home/behrooz.mansouri/SIGIRSciDis/25_04/figures/images"
DATA_DIR = "/mnt/netstore1_home/behrooz.mansouri/SIGIRSciDis/figureGen/figure_query_output"
OUTPUT_DIR = "/home/adah.holt/scientific-discovery/colpali-finetuned"
MODEL_NAME = "vidore/colqwen2.5-v0.2"

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
            img_path = uid_to_path(uid)
            if not Path(img_path).exists():
                continue
            try:
                img = Image.open(img_path).convert("RGB")
                pairs.append({"query": entry["query"], "image": img})
            except Exception:
                continue
    print(f"Loaded {len(pairs)} pairs from {json_path}")
    return pairs

class ColPaliDataset(TorchDataset):
    def __init__(self, pairs):
        self.pairs = pairs
    def __len__(self):
        return len(self.pairs)
    def __getitem__(self, idx):
        return self.pairs[idx]["query"], self.pairs[idx]["image"]

print("Loading pairs...")
train_pairs = load_pairs(f"{DATA_DIR}/Train.json")
val_pairs   = load_pairs(f"{DATA_DIR}/Val.json")

train_dataset = ColPaliDataset(train_pairs)
val_dataset   = ColPaliDataset(val_pairs)

print("Loading model...")
model = ColQwen2_5.from_pretrained(MODEL_NAME, torch_dtype="auto", device_map="cuda")
processor = ColQwen2_5_Processor.from_pretrained(MODEL_NAME)

loss_fn = ColPaliLoss()

args = TrainingArguments(
    output_dir=OUTPUT_DIR,
    num_train_epochs=1,
    per_device_train_batch_size=4,
    per_device_eval_batch_size=4,
    learning_rate=1e-5,
    warmup_ratio=0.1,
    eval_strategy="steps",
    eval_steps=200,
    save_strategy="steps",
    save_steps=200,
    load_best_model_at_end=True,
    fp16=True,
    logging_steps=50,
    report_to="none",
)

trainer = ContrastiveTrainer(
    model=model,
    args=args,
    train_dataset=train_dataset,
    eval_dataset=val_dataset,
    loss=loss_fn,
    processor=processor,
)

print("Starting training...")
trainer.train()
model.save_pretrained(OUTPUT_DIR)
processor.save_pretrained(OUTPUT_DIR)
print(f"Done! Model saved to {OUTPUT_DIR}")