"""
ColQwen2.5 Pretrained Retrieval Script
Generates a TREC run file using the pretrained ColQwen2.5 model (no finetuning).
Usage:
    python colqwen_retrieval_pretrained.py \
        --queries_json /path/to/Test.json \
        --image_dir /path/to/figures/ \
        --output_trec run_colqwen_pretrained.trec
"""

import argparse
import json
import os
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"
import torch
from PIL import Image, UnidentifiedImageError
from tqdm import tqdm
from colpali_engine.models import ColQwen2_5, ColQwen2_5_Processor

# Allow very large images and avoid decompression bomb errors
Image.MAX_IMAGE_PIXELS = None

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--queries_json", required=True, help="Path to Test.json")
    parser.add_argument("--image_dir", required=True, help="Directory containing figure images")
    parser.add_argument("--output_trec", default="run_colqwen_pretrained.trec", help="Output TREC file path")
    parser.add_argument("--model_name", default="vidore/colqwen2.5-v0.2", help="HuggingFace model name")
    parser.add_argument("--batch_size", type=int, default=1)
    parser.add_argument("--top_k", type=int, default=100, help="Number of results per query")
    return parser.parse_args()

def load_queries(queries_json):
    with open(queries_json) as f:
        data = json.load(f)
    # Support both list of dicts and dict formats
    if isinstance(data, list):
        return {str(item["query_id"]): item["query"] for item in data}
    else:
        return {str(k): v for k, v in data.items()}

def get_image_ids(image_dir):
    ids = []
    for fname in sorted(os.listdir(image_dir)):
        if fname.endswith(".png") or fname.endswith(".jpg"):
            ids.append(os.path.splitext(fname)[0])
    return ids

def main():
    args = parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    print(f"Loading model: {args.model_name}")
    model = ColQwen2_5.from_pretrained(
        args.model_name,
        torch_dtype=torch.bfloat16,
        device_map="auto"
    ).eval()
    processor = ColQwen2_5_Processor.from_pretrained(args.model_name)

    # Load queries
    queries = load_queries(args.queries_json)
    print(f"Loaded {len(queries)} queries")

    # Get all image IDs
    image_ids = get_image_ids(args.image_dir)
    print(f"Found {len(image_ids)} images")

    # Encode all images
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
                    # Resize if image is extremely large (> 50MP) to avoid OOM
                    if img.width * img.height > 50_000_000:
                        ratio = (50_000_000 / (img.width * img.height)) ** 0.5
                        new_size = (int(img.width * ratio), int(img.height * ratio))
                        img = img.resize(new_size, Image.LANCZOS)
                    images.append(img)
                    loaded_ids.append(img_id)
                except (UnidentifiedImageError, OSError, Exception) as e:
                    print(f"\nSkipping {img_id}: {e}")

        if not images:
            continue

        try:
            with torch.no_grad():
                batch_inputs = processor.process_images(images).to(device)
                embeddings = model(**batch_inputs)
            image_embeddings.append(embeddings.cpu())  # move to CPU to free GPU memory
            valid_image_ids.extend(loaded_ids)
        except torch.cuda.OutOfMemoryError:
            print(f"\nOOM on batch {i}, skipping {loaded_ids}")
        finally:
            torch.cuda.empty_cache()

    all_image_embeddings = torch.cat(image_embeddings, dim=0)
    print(f"Encoded {len(valid_image_ids)} images")

    # Encode queries and score
    print("Encoding queries and scoring...")
    results = {}

    query_ids = list(queries.keys())
    query_texts = list(queries.values())

    for i in tqdm(range(0, len(query_texts), args.batch_size)):
        batch_qids = query_ids[i:i + args.batch_size]
        batch_qtexts = query_texts[i:i + args.batch_size]

        with torch.no_grad():
            query_inputs = processor.process_queries(batch_qtexts).to(device)
            query_embeddings = model(**query_inputs)

        # Move image embeddings to GPU just for scoring, then back to CPU
        all_image_embeddings_gpu = all_image_embeddings.to(device)
        scores = processor.score_retrieval(query_embeddings, all_image_embeddings_gpu)
        del all_image_embeddings_gpu
        torch.cuda.empty_cache()

        for j, qid in enumerate(batch_qids):
            query_scores = scores[j].cpu().float().numpy()
            sorted_indices = query_scores.argsort()[::-1][:args.top_k]
            results[qid] = [
                (valid_image_ids[idx], float(query_scores[idx]))
                for idx in sorted_indices
            ]

    # Write TREC file
    print(f"Writing TREC file to {args.output_trec}")
    with open(args.output_trec, "w") as f:
        for qid, ranked_docs in results.items():
            for rank, (doc_id, score) in enumerate(ranked_docs, start=1):
                f.write(f"{qid} Q0 {doc_id} {rank} {score:.6f} colqwen_pretrained\n")

    print("Done!")

if __name__ == "__main__":
    main()