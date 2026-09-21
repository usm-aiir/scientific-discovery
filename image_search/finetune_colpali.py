import os
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

from sentence_transformers import MultiVectorEncoder
from sentence_transformers.trainer import MultiVectorTrainer
from sentence_transformers.training_args import MultiVectorTrainingArguments
from sentence_transformers.losses import MultipleNegativesRankingLoss
from datasets import Dataset
from PIL import Image
from pathlib import Path
import json

FIGURE_DIR = "/mnt/netstore1_home/behrooz.mansouri/SIGIRSciDis/25_04/figures/images"
DATA_DIR = "/mnt/netstore1_home/behrooz.mansouri/SIGIRSciDis/figureGen/figure_query_output"
OUTPUT_DIR = "/home/adah.holt/scientific-discovery/colpali-finetuned"

def uid_to_path(uid):
    parts = uid.split("::")
    stem = f"{parts[0]}_{parts[-1].lstrip('F')}"
    return str(Path(FIGURE_DIR) / f"{stem}.png")

def load_pairs(json_path):
    with open(json_path, encoding="utf-8") as f:
        entries = json.load(f)
    queries, images = [], []
    for entry in entries:
        for uid in entry["source_figure_uids"]:
            img_path = uid_to_path(uid)
            if not Path(img_path).exists():
                continue
            try:
                img = Image.open(img_path).convert("RGB")
                queries.append(entry["query"])
                images.append(img)
            except Exception:
                continue
    print(f"Loaded {len(queries)} pairs from {json_path}")
    return queries, images

print("Loading train pairs...")
train_queries, train_images = load_pairs(f"{DATA_DIR}/Train.json")

print("Loading val pairs...")
val_queries, val_images = load_pairs(f"{DATA_DIR}/Val.json")

train_dataset = Dataset.from_dict({"query": train_queries, "image": train_images})
val_dataset   = Dataset.from_dict({"query": val_queries,   "image": val_images})

model = MultiVectorEncoder("vidore/colqwen2.5-v0.2")
loss  = MultipleNegativesRankingLoss(model)

args = MultiVectorTrainingArguments(
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

trainer = MultiVectorTrainer(
    model=model,
    args=args,
    train_dataset=train_dataset,
    eval_dataset=val_dataset,
    loss=loss,
)

print("Starting training...")
trainer.train()
model.save_pretrained(OUTPUT_DIR)
print(f"Done! Model saved to {OUTPUT_DIR}")