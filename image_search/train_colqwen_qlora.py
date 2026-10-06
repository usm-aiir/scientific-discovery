"""
ColQwen2.5 QLoRA Finetuning Script
Finetunes ColQwen2.5 on (query, figure) pairs using 4-bit quantization + LoRA.
Saves best adapter based on validation nDCG@10.

Usage:
    python train_colqwen_qlora.py \
        --train_json /path/to/Train.json \
        --train_qrels /path/to/Train_figure_qrels.tsv \
        --val_json /path/to/Val.json \
        --val_qrels /path/to/Val_figure_qrels.tsv \
        --image_dir /path/to/figures/ \
        --output_dir colqwen-qlora-run \
        --epochs 5
"""

import argparse
import json
import os
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"
import torch
import numpy as np
from PIL import Image, UnidentifiedImageError
from tqdm import tqdm

# Allow very large images
Image.MAX_IMAGE_PIXELS = None
from torch.utils.data import Dataset, DataLoader
from transformers import BitsAndBytesConfig
from peft import LoraConfig, get_peft_model, TaskType
from colpali_engine.models import ColQwen2_5, ColQwen2_5_Processor
from ranx import Qrels, Run, evaluate

# ── Args ──────────────────────────────────────────────────────────────────────

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--train_json",  required=True)
    parser.add_argument("--train_qrels", required=True)
    parser.add_argument("--val_json",    required=True)
    parser.add_argument("--val_qrels",   required=True)
    parser.add_argument("--image_dir",   required=True)
    parser.add_argument("--output_dir",  default="colqwen-qlora-run")
    parser.add_argument("--model_name",  default="vidore/colqwen2.5-v0.2")
    parser.add_argument("--epochs",      type=int, default=5)
    parser.add_argument("--batch_size",  type=int, default=2)
    parser.add_argument("--lr",          type=float, default=5e-5)
    parser.add_argument("--lora_r",      type=int, default=16)
    parser.add_argument("--lora_alpha",  type=int, default=32)
    return parser.parse_args()

# ── Data ──────────────────────────────────────────────────────────────────────

def load_queries(path):
    with open(path) as f:
        data = json.load(f)
    if isinstance(data, list):
        return {str(item["query_id"]): item["query"] for item in data}
    return {str(k): v for k, v in data.items()}

def load_qrels(path):
    """Returns dict: {query_id: {doc_id: score}}"""
    qrels = {}
    with open(path) as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) < 4:
                continue
            qid, _, doc_id, score = parts[0], parts[1], parts[2], int(parts[3])
            if score > 0:
                qrels.setdefault(qid, {})[doc_id] = doc_id
    return qrels

class FigureQueryDataset(Dataset):
    def __init__(self, queries, qrels, image_dir):
        self.pairs = []
        for qid, pos_docs in qrels.items():
            if qid not in queries:
                continue
            query = queries[qid]
            for doc_id in pos_docs:
                img_path = os.path.join(image_dir, doc_id + ".png")
                if not os.path.exists(img_path):
                    img_path = os.path.join(image_dir, doc_id + ".jpg")
                if os.path.exists(img_path):
                    self.pairs.append((query, img_path))

    def __len__(self):
        return len(self.pairs)

    def __getitem__(self, idx):
        query, img_path = self.pairs[idx]
        try:
            image = Image.open(img_path).convert("RGB")
            if image.width * image.height > 10_000_000:
                ratio = (10_000_000 / (image.width * image.height)) ** 0.5
                new_size = (int(image.width * ratio), int(image.height * ratio))
                image = image.resize(new_size, Image.LANCZOS)
        except (UnidentifiedImageError, OSError, Exception):
            image = Image.new("RGB", (224, 224), color=(128, 128, 128))  # blank fallback
        return query, image

# ── Validation ────────────────────────────────────────────────────────────────

def run_validation(model, processor, val_queries, val_qrels_dict, image_dir, device, batch_size=4):
    model.eval()

    # Get all unique doc IDs from val qrels
    all_doc_ids = set()
    for docs in val_qrels_dict.values():
        all_doc_ids.update(docs.keys())
    all_doc_ids = sorted(all_doc_ids)

    # Encode all val images
    image_embeddings = []
    valid_doc_ids = []
    images_to_encode = []
    ids_to_encode = []

    for doc_id in all_doc_ids:
        img_path = os.path.join(image_dir, doc_id + ".png")
        if not os.path.exists(img_path):
            img_path = os.path.join(image_dir, doc_id + ".jpg")
        if os.path.exists(img_path):
            try:
                img = Image.open(img_path).convert("RGB")
                if img.width * img.height > 10_000_000:
                    ratio = (10_000_000 / (img.width * img.height)) ** 0.5
                    img = img.resize((int(img.width * ratio), int(img.height * ratio)), Image.LANCZOS)
                images_to_encode.append(img)
                ids_to_encode.append(doc_id)
            except (UnidentifiedImageError, OSError, Exception) as e:
                print(f"\nSkipping val image {doc_id}: {e}")

    for i in range(0, len(images_to_encode), batch_size):
        batch_imgs = images_to_encode[i:i+batch_size]
        try:
            with torch.no_grad():
                inputs = processor.process_images(batch_imgs).to(device)
                emb = model(**inputs)
            image_embeddings.append(emb.cpu())
            valid_doc_ids.extend(ids_to_encode[i:i+batch_size])
        except torch.cuda.OutOfMemoryError:
            print(f"\nOOM on val batch {i}, skipping")
        finally:
            torch.cuda.empty_cache()

    if not image_embeddings:
        return {"ndcg@10": 0.0, "recall@10": 0.0}

    # Pad to same sequence length before concatenating (images have variable patch counts)
    max_seq = max(e.shape[1] for e in image_embeddings)
    padded = [torch.nn.functional.pad(e, (0, 0, 0, max_seq - e.shape[1])) for e in image_embeddings]
    all_image_emb = torch.cat(padded, dim=0)

    # Build ranx qrels
    qrels_dict = {}
    run_dict = {}

    query_ids = list(val_queries.keys())
    query_texts = list(val_queries.values())

    for i in range(0, len(query_texts), batch_size):
        batch_qids = query_ids[i:i+batch_size]
        batch_qtexts = query_texts[i:i+batch_size]

        with torch.no_grad():
            q_inputs = processor.process_queries(batch_qtexts).to(device)
            q_emb = model(**q_inputs)

        scores = processor.score_multi_vector(q_emb, all_image_emb)

        for j, qid in enumerate(batch_qids):
            if qid not in val_qrels_dict:
                continue
            s = scores[j].cpu().float().numpy()
            sorted_idx = s.argsort()[::-1][:100]
            run_dict[qid] = {valid_doc_ids[k]: float(s[k]) for k in sorted_idx}
            qrels_dict[qid] = {doc_id: 1 for doc_id in val_qrels_dict[qid]}

    qrels_obj = Qrels(qrels_dict)
    run_obj = Run(run_dict)
    metrics = evaluate(qrels_obj, run_obj, ["ndcg@10", "recall@10"])
    return metrics

# ── Training ──────────────────────────────────────────────────────────────────

def main():
    args = parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    os.makedirs(args.output_dir, exist_ok=True)
    best_adapter_dir = os.path.join(args.output_dir, "best_adapter")

    # Load model in 4-bit
    print(f"Loading {args.model_name} in 4-bit...")
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16,
        bnb_4bit_use_double_quant=True,
    )
    model = ColQwen2_5.from_pretrained(
        args.model_name,
        quantization_config=bnb_config,
        device_map="auto",
        ignore_mismatched_sizes=True,
    )
    processor = ColQwen2_5_Processor.from_pretrained(args.model_name)

    # Apply LoRA
    lora_config = LoraConfig(
        r=args.lora_r,
        lora_alpha=args.lora_alpha,
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj",
                        "gate_proj", "up_proj", "down_proj"],
        lora_dropout=0.05,
        bias="none",
    )
    model = get_peft_model(model, lora_config)
    model.print_trainable_parameters()
    model.enable_input_require_grads()
    if hasattr(model, "gradient_checkpointing_enable"):
        model.gradient_checkpointing_enable()

    # Load data
    train_queries = load_queries(args.train_json)
    train_qrels   = load_qrels(args.train_qrels)
    val_queries   = load_queries(args.val_json)
    val_qrels     = load_qrels(args.val_qrels)

    dataset = FigureQueryDataset(train_queries, train_qrels, args.image_dir)
    print(f"Training pairs: {len(dataset)}")

    optimizer = torch.optim.AdamW(
        [p for p in model.parameters() if p.requires_grad], lr=args.lr
    )

    best_score = -1.0

    for epoch in range(1, args.epochs + 1):
        model.train()
        total_loss = 0.0
        count = 0

        # Shuffle pairs each epoch
        indices = torch.randperm(len(dataset)).tolist()

        for i in tqdm(range(0, len(indices), args.batch_size), desc=f"Epoch {epoch}"):
            batch_idx = indices[i:i + args.batch_size]
            queries_batch = []
            images_batch  = []
            for idx in batch_idx:
                q, img = dataset[idx]
                queries_batch.append(q)
                images_batch.append(img)

            # Need ≥2 pairs for in-batch negatives; skip singleton batches
            if len(batch_idx) < 2:
                continue

            try:
                q_inputs   = processor.process_queries(queries_batch).to(device)
                img_inputs = processor.process_images(images_batch).to(device)

                q_emb   = model(**q_inputs)
                img_emb = model(**img_inputs)

                # Contrastive loss (in-batch negatives, bidirectional)
                scores = processor.score_multi_vector(q_emb, img_emb).to(device)  # [B, B]
                labels = torch.arange(len(batch_idx), device=device)
                loss = (
                    torch.nn.functional.cross_entropy(scores, labels) +
                    torch.nn.functional.cross_entropy(scores.T, labels)
                ) / 2

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                total_loss += loss.item()
                count += 1
            except torch.cuda.OutOfMemoryError:
                print(f"\nOOM on training batch {i}, skipping")
                optimizer.zero_grad()
            finally:
                torch.cuda.empty_cache()

        avg_loss = total_loss / max(count, 1)
        print(f"Epoch {epoch} — avg loss: {avg_loss:.4f}")

        # Save epoch checkpoint
        epoch_dir = os.path.join(args.output_dir, f"epoch_{epoch}_adapter")
        model.save_pretrained(epoch_dir)
        processor.save_pretrained(epoch_dir)

        # Validate
        print(f"Running validation...")
        val_metrics = run_validation(
            model, processor, val_queries, val_qrels, args.image_dir, device, args.batch_size
        )
        print(f"Val nDCG@10: {val_metrics['ndcg@10']:.4f}  Recall@10: {val_metrics['recall@10']:.4f}")

        # Save best
        score = val_metrics["ndcg@10"]
        if score > best_score:
            best_score = score
            model.save_pretrained(best_adapter_dir)
            processor.save_pretrained(best_adapter_dir)
            print(f"  ✓ New best model saved (nDCG@10={score:.4f})")

    print(f"\nTraining complete. Best val nDCG@10: {best_score:.4f}")
    print(f"Best adapter saved to: {best_adapter_dir}")

if __name__ == "__main__":
    main()