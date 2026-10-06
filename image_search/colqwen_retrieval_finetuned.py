"""
ColQwen2.5 Finetuned Retrieval Script
Loads the saved LoRA adapter and runs retrieval on the validation set.

Usage:
    python colqwen_retrieval_finetuned.py \
        --queries_json /path/to/Val.json \
        --image_dir /path/to/figures/ \
        --adapter_dir colqwen-qlora-run/best_adapter \
        --output_trec run_colqwen_finetuned.trec
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

# Raise PIL decompression bomb limit before any image is opened
Image.MAX_IMAGE_PIXELS = 400_000_000

MAX_PIXELS = 10 * 1_000_000  # 10MP cap for model input

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--queries_json", required=True, help="Path to Val.json or Test.json")
    parser.add_argument("--image_dir",    required=True, help="Directory containing figure images")
    parser.add_argument("--adapter_dir",  required=True, help="Path to saved LoRA adapter (best_adapter/)")
    parser.add_argument("--base_model",   default="vidore/colqwen2.5-v0.2")
    parser.add_argument("--output_trec",  default="run_colqwen_finetuned.trec")
    parser.add_argument("--batch_size",   type=int, default=4)
    parser.add_argument("--top_k",        type=int, default=100)
    return parser.parse_args()

def load_queries(path):
    with open(path) as f:
        data = json.load(f)
    if isinstance(data, list):
        return {str(item["query_id"]): item["query"] for item in data}
    return {str(k): v for k, v in data.items()}

def get_image_ids(image_dir):
    ids = []
    for fname in sorted(os.listdir(image_dir)):
        if fname.endswith(".png") or fname.endswith(".jpg"):
            ids.append(os.path.splitext(fname)[0])
    return ids

def filename_to_qrel_id(img_id):
    """Convert filename ID (2504.15247_10) to qrel ID (2504.15247::F10).
    Also handles sub-panel suffixes like 2504.00008_11_a -> 2504.00008::F11::e
    but keeps it simple: paper_fignum[_panel] -> paper::Ffignum[::panel]
    """
    parts = img_id.split("_")
    paper = parts[0]          # e.g. 2504.15247
    fignum = parts[1]         # e.g. 10
    if len(parts) > 2:
        panel = parts[2]      # e.g. a, b, c
        return f"{paper}::F{fignum}::{panel}"
    return f"{paper}::F{fignum}"

def main():
    args = parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

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
    model = PeftModel.from_pretrained(base_model, args.adapter_dir)
    model = model.eval()

    processor = ColQwen2_5_Processor.from_pretrained(args.adapter_dir)

    # Load queries
    queries = load_queries(args.queries_json)
    print(f"Loaded {len(queries)} queries")

    # Get all image IDs
    image_ids = get_image_ids(args.image_dir)
    print(f"Found {len(image_ids)} images")

    # Encode images
    print("Encoding images...")
    image_embeddings = []
    valid_image_ids = []

    for i in tqdm(range(0, len(image_ids), args.batch_size)):
        batch_ids = image_ids[i:i + args.batch_size]
        images = []
        loaded_ids = []
        for img_id in batch_ids:
            img_path = os.path.join(args.image_dir, img_id + ".png")
            if not os.path.exists(img_path):
                img_path = os.path.join(args.image_dir, img_id + ".jpg")
            if os.path.exists(img_path):
                try:
                    img = Image.open(img_path).convert("RGB")
                    if img.width * img.height > MAX_PIXELS:
                        scale = (MAX_PIXELS / (img.width * img.height)) ** 0.5
                        new_w = int(img.width * scale)
                        new_h = int(img.height * scale)
                        img = img.resize((new_w, new_h), Image.LANCZOS)
                    images.append(img)
                    loaded_ids.append(img_id)
                except Exception as e:
                    print(f"Skipping {img_id}: {e}")
                    continue

        if not images:
            continue

        with torch.no_grad():
            inputs = processor.process_images(images).to(device)
            emb = model(**inputs)
        image_embeddings.append(emb)
        valid_image_ids.extend(loaded_ids)

    all_image_emb = torch.cat(image_embeddings, dim=0)
    print(f"Encoded {len(valid_image_ids)} images")

    # Encode queries and score
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

        scores = processor.score_multi_vector(q_emb, all_image_emb)

        for j, qid in enumerate(batch_qids):
            s = scores[j].cpu().float().numpy()
            sorted_idx = s.argsort()[::-1][:args.top_k]
            results[qid] = [(valid_image_ids[k], float(s[k])) for k in sorted_idx]

    # Write TREC file
    print(f"Writing TREC file to {args.output_trec}")
    with open(args.output_trec, "w") as f:
        for qid, ranked_docs in results.items():
            for rank, (doc_id, score) in enumerate(ranked_docs, start=1):
                qrel_id = filename_to_qrel_id(doc_id)
                f.write(f"{qid} Q0 {qrel_id} {rank} {score:.6f} colqwen_finetuned\n")

    print("Done!")

if __name__ == "__main__":
    main()