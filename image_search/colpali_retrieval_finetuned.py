"""
ColQwen2.5 Finetuned Retrieval Script
Loads the saved LoRA adapter and runs retrieval on the test set.

Features:
  - Parallel image loading via DataLoader (num_workers=4)
  - Checkpoint: saves embeddings every --checkpoint_every images so a crash
    can resume from the last saved point instead of starting over.
  - All image-quality guards: corrupt files, extreme aspect ratios, oversized.

Usage:
    python colqwen_retrieval_finetuned.py \
        --queries_json /path/to/Test.json \
        --image_dir /path/to/figures/images \
        --adapter_dir colqwen-qlora-run/best_adapter \
        --output_trec run_colqwen_finetuned.trec

To resume after a crash (checkpoint file is detected automatically):
    Same command — the script picks up from the last checkpoint.
"""

import argparse
import json
import os
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"
import torch
from PIL import Image
from tqdm import tqdm
from peft import PeftModel
from transformers import BitsAndBytesConfig
from colpali_engine.models import ColQwen2_5, ColQwen2_5_Processor
from torch.utils.data import Dataset, DataLoader

# Raise PIL decompression bomb limit before any image is opened
Image.MAX_IMAGE_PIXELS = 400_000_000

MAX_PIXELS = 10 * 1_000_000   # 10MP cap for model input
MAX_ASPECT_RATIO = 200         # Qwen2.5-VL hard limit


# ---------------------------------------------------------------------------
# Dataset / DataLoader helpers
# ---------------------------------------------------------------------------

class ImageDataset(Dataset):
    def __init__(self, image_ids, image_dir):
        self.image_ids = image_ids
        self.image_dir = image_dir

    def __len__(self):
        return len(self.image_ids)

    def __getitem__(self, idx):
        img_id = self.image_ids[idx]
        img_path = os.path.join(self.image_dir, img_id + ".png")
        if not os.path.exists(img_path):
            img_path = os.path.join(self.image_dir, img_id + ".jpg")
        if not os.path.exists(img_path):
            return img_id, None
        try:
            img = Image.open(img_path).convert("RGB")
            img.load()  # force full decode — catches corrupt files early
            if img.width * img.height > MAX_PIXELS:
                scale = (MAX_PIXELS / (img.width * img.height)) ** 0.5
                img = img.resize(
                    (int(img.width * scale), int(img.height * scale)),
                    Image.LANCZOS,
                )
            min_dim = min(img.width, img.height)
            if min_dim == 0:
                return img_id, None
            if max(img.width, img.height) / min_dim > MAX_ASPECT_RATIO:
                return img_id, None
            return img_id, img
        except Exception as e:
            print(f"\nSkipping {img_id}: {e}")
            return img_id, None


def collate_fn(batch):
    """Filter out failed loads and return (ids, images) lists."""
    valid = [(img_id, img) for img_id, img in batch if img is not None]
    if not valid:
        return [], []
    ids, imgs = zip(*valid)
    return list(ids), list(imgs)


# ---------------------------------------------------------------------------
# Checkpointing helpers
# ---------------------------------------------------------------------------

def checkpoint_path(output_trec):
    return output_trec.replace(".trec", "_ckpt.pt")


def save_checkpoint(ckpt_file, image_embeddings, valid_image_ids):
    torch.save({"embeddings": image_embeddings, "ids": valid_image_ids}, ckpt_file)


def load_checkpoint(ckpt_file):
    data = torch.load(ckpt_file, map_location="cpu")
    return data["embeddings"], data["ids"]


# ---------------------------------------------------------------------------
# Argument parsing
# ---------------------------------------------------------------------------

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--queries_json",      required=True)
    parser.add_argument("--image_dir",         required=True)
    parser.add_argument("--adapter_dir",       required=True)
    parser.add_argument("--base_model",        default="vidore/colqwen2.5-v0.2")
    parser.add_argument("--output_trec",       default="run_colqwen_finetuned.trec")
    parser.add_argument("--batch_size",        type=int, default=4)
    parser.add_argument("--num_workers",       type=int, default=4,
                        help="DataLoader worker processes for parallel image loading")
    parser.add_argument("--top_k",             type=int, default=100)
    parser.add_argument("--checkpoint_every",  type=int, default=2000,
                        help="Save a checkpoint every N images encoded")
    return parser.parse_args()


def load_queries(path):
    with open(path) as f:
        data = json.load(f)
    if isinstance(data, list):
        return {str(item["query_id"]): item["query"] for item in data}
    return {str(k): v for k, v in data.items()}


def get_image_ids(image_dir):
    return [
        os.path.splitext(fname)[0]
        for fname in sorted(os.listdir(image_dir))
        if fname.endswith(".png") or fname.endswith(".jpg")
    ]


def pad_and_cat(embeddings):
    """Pad a list of [B, seq, dim] tensors to the same seq length, then cat."""
    max_seq = max(e.shape[1] for e in embeddings)
    dim = embeddings[0].shape[2]
    padded = []
    for e in embeddings:
        if e.shape[1] < max_seq:
            pad = torch.zeros(e.shape[0], max_seq - e.shape[1], dim, dtype=e.dtype)
            e = torch.cat([e, pad], dim=1)
        padded.append(e)
    return torch.cat(padded, dim=0)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    args = parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # ------------------------------------------------------------------ model
    print(f"Loading base model: {args.base_model}")
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16,
        bnb_4bit_use_double_quant=True,
    )
    base_model = ColQwen2_5.from_pretrained(
        args.base_model,
        quantization_config=bnb_config,
        device_map="auto",
    )
    print(f"Loading LoRA adapter from: {args.adapter_dir}")
    model = PeftModel.from_pretrained(base_model, args.adapter_dir).eval()
    processor = ColQwen2_5_Processor.from_pretrained(args.adapter_dir)

    # ------------------------------------------------------------------ queries
    queries = load_queries(args.queries_json)
    print(f"Loaded {len(queries)} queries")

    # ------------------------------------------------------------------ images
    all_image_ids = get_image_ids(args.image_dir)
    print(f"Found {len(all_image_ids)} images")

    # Check for existing checkpoint
    ckpt_file = checkpoint_path(args.output_trec)
    image_embeddings = []
    valid_image_ids = []

    if os.path.exists(ckpt_file):
        print(f"Resuming from checkpoint: {ckpt_file}")
        image_embeddings, valid_image_ids = load_checkpoint(ckpt_file)
        done_ids = set(valid_image_ids)
        remaining_ids = [i for i in all_image_ids if i not in done_ids]
        print(f"  Already encoded: {len(valid_image_ids)}, remaining: {len(remaining_ids)}")
    else:
        remaining_ids = all_image_ids

    # ------------------------------------------------------------------ encode
    dataset = ImageDataset(remaining_ids, args.image_dir)
    loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        collate_fn=collate_fn,
        pin_memory=False,
        prefetch_factor=2 if args.num_workers > 0 else None,
    )

    print("Encoding images...")
    images_since_ckpt = 0

    for batch_ids, batch_images in tqdm(loader, total=len(loader)):
        if not batch_images:
            continue
        try:
            with torch.no_grad():
                inputs = processor.process_images(batch_images).to(device)
                emb = model(**inputs)
            image_embeddings.append(emb.cpu())
            valid_image_ids.extend(batch_ids)
            images_since_ckpt += len(batch_ids)

            # Save checkpoint periodically
            if images_since_ckpt >= args.checkpoint_every:
                save_checkpoint(ckpt_file, image_embeddings, valid_image_ids)
                print(f"\n  Checkpoint saved ({len(valid_image_ids)} images encoded)")
                images_since_ckpt = 0

        except Exception as e:
            print(f"\nSkipping batch: {e}")
        finally:
            torch.cuda.empty_cache()

    # Final checkpoint
    save_checkpoint(ckpt_file, image_embeddings, valid_image_ids)
    print(f"Encoded {len(valid_image_ids)} images total")

    # ------------------------------------------------------------------ score
    all_image_emb = pad_and_cat(image_embeddings)

    print("Scoring queries...")
    results = {}
    query_ids   = list(queries.keys())
    query_texts = list(queries.values())

    for i in tqdm(range(0, len(query_texts), args.batch_size)):
        batch_qids   = query_ids[i:i + args.batch_size]
        batch_qtexts = query_texts[i:i + args.batch_size]

        with torch.no_grad():
            q_inputs = processor.process_queries(batch_qtexts).to(device)
            q_emb    = model(**q_inputs)

        scores = processor.score_multi_vector(q_emb, all_image_emb.to(device))

        for j, qid in enumerate(batch_qids):
            s = scores[j].cpu().float().numpy()
            sorted_idx = s.argsort()[::-1][:args.top_k]
            results[qid] = [(valid_image_ids[k], float(s[k])) for k in sorted_idx]

    # ------------------------------------------------------------------ write
    print(f"Writing TREC file to {args.output_trec}")
    with open(args.output_trec, "w") as f:
        for qid, ranked_docs in results.items():
            for rank, (doc_id, score) in enumerate(ranked_docs, start=1):
                f.write(f"{qid} Q0 {doc_id} {rank} {score:.6f} colqwen_finetuned\n")

    # Clean up checkpoint once TREC file is written successfully
    if os.path.exists(ckpt_file):
        os.remove(ckpt_file)
        print(f"Checkpoint deleted (run complete).")

    print("Done!")


if __name__ == "__main__":
    main()