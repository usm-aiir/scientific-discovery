"""
ColQwen2.5 pretrained image retrieval -> TREC-format run file.
 
Given:
  - a directory of images (e.g. 100K figures)
  - a JSON file of queries (list of dicts with at least "query_id" and "query")
 
Produces a standard TREC run file:
  query_id Q0 doc_id rank score run_tag
 
doc_id = image filename without its extension (e.g. "2504.15247_10").
 
Streams images in batches: for each batch it computes embeddings, scores them
against ALL (already-encoded) queries, updates a running top-100 heap per
query, then discards the batch embeddings before moving to the next batch.
This avoids accumulating all image embeddings in memory at once.
"""
 
import argparse
import heapq
import json
import os
import sys
from pathlib import Path
 
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"
 
import torch
from PIL import Image
from colpali_engine.models import ColQwen2_5, ColQwen2_5_Processor
 
Image.MAX_IMAGE_PIXELS = 400_000_000
MAX_PIXELS      = 10 * 1_000_000
MAX_ASPECT_RATIO = 200
 
IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp"}
 
 
def find_images(image_dir):
    image_dir = Path(image_dir)
    return [
        p for p in sorted(image_dir.rglob("*"))
        if p.suffix.lower() in IMAGE_EXTS and p.is_file()
    ]
 
 
def doc_id_from_path(path: Path) -> str:
    return path.stem
 
 
def load_queries(query_json_path):
    with open(query_json_path, "r", encoding="utf-8") as f:
        data = json.load(f)
    if isinstance(data, dict):
        data = data.get("queries", list(data.values()))
    return [(str(item["query_id"]), item["query"]) for item in data]
 
 
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
 
 
def encode_queries(model, processor, queries, device, batch_size=16):
    """Encode all queries upfront; queries are few so we keep all embeddings."""
    all_embeddings = []
    texts = [q for _, q in queries]
    for i in range(0, len(texts), batch_size):
        batch_texts = texts[i : i + batch_size]
        inputs = processor.process_queries(batch_texts).to(device)
        with torch.no_grad():
            emb = model(**inputs)
        all_embeddings.append(emb.cpu())
    return pad_and_cat(all_embeddings)
 
 
def load_image_safe(path: Path):
    try:
        img = Image.open(path).convert("RGB")
        img.load()
        if img.width * img.height > MAX_PIXELS:
            scale = (MAX_PIXELS / (img.width * img.height)) ** 0.5
            img = img.resize(
                (int(img.width * scale), int(img.height * scale)),
                Image.LANCZOS,
            )
        min_dim = min(img.width, img.height)
        if min_dim == 0:
            return None
        if max(img.width, img.height) / min_dim > MAX_ASPECT_RATIO:
            return None
        return img
    except Exception as e:
        print(f"[WARN] Failed to load image {path}: {e}", file=sys.stderr)
        return None
 
 
def main():
    parser = argparse.ArgumentParser(description="ColQwen2.5 pretrained retrieval -> TREC run file")
    parser.add_argument(
        "--image_dir",
        default="/mnt/netstore1_home/adah.holt/figures/images",
        help="Directory containing corpus images (searched recursively).",
    )
    parser.add_argument("--query_json",  required=True, help="Path to the queries JSON file.")
    parser.add_argument("--output",      default="run_colqwen_pretrained.trec")
    parser.add_argument("--model_name",  default="vidore/colqwen2.5-v0.2")
    parser.add_argument("--image_batch_size", type=int, default=4)
    parser.add_argument("--query_batch_size", type=int, default=16)
    parser.add_argument("--top_k",       type=int, default=100)
    parser.add_argument("--run_tag",     default="colqwen_pretrained")
    parser.add_argument("--device",      default=None,
                        help="e.g. 'cuda:0' or 'cpu'. Default: cuda:0 if available.")
    args = parser.parse_args()
 
    device = torch.device(args.device if args.device else
                          ("cuda:0" if torch.cuda.is_available() else "cpu"))
    print(f"Using device: {device}")
 
    print(f"Loading model: {args.model_name}")
    model = ColQwen2_5.from_pretrained(
        args.model_name,
        torch_dtype=torch.bfloat16,
        device_map="auto",
    ).eval()
    processor = ColQwen2_5_Processor.from_pretrained(args.model_name)
 
    print("Loading queries...")
    queries = load_queries(args.query_json)
    print(f"  {len(queries)} queries loaded.")
 
    print("Encoding queries...")
    query_embeddings = encode_queries(model, processor, queries, device,
                                      batch_size=args.query_batch_size)
    query_embeddings_gpu = query_embeddings.to(device)
    print(f"  Query embeddings: {query_embeddings_gpu.shape}")
 
    print("Finding images...")
    image_paths = find_images(args.image_dir)
    print(f"  {len(image_paths)} images found.")
 
    # one min-heap of (score, doc_id) per query, capped at top_k
    heaps = [[] for _ in queries]
 
    batch_size = args.image_batch_size
    n_batches  = (len(image_paths) + batch_size - 1) // batch_size
 
    for b in range(n_batches):
        batch_paths = image_paths[b * batch_size : (b + 1) * batch_size]
        pil_images  = []
        valid_paths = []
        for p in batch_paths:
            img = load_image_safe(p)
            if img is not None:
                pil_images.append(img)
                valid_paths.append(p)
 
        if not pil_images:
            continue
 
        try:
            inputs = processor.process_images(pil_images).to(device)
            with torch.no_grad():
                image_embeddings = model(**inputs)
 
            # scores: (num_queries, num_images_in_batch)
            scores = processor.score_multi_vector(query_embeddings_gpu, image_embeddings)
            scores = scores.detach().cpu()
 
            doc_ids = [doc_id_from_path(p) for p in valid_paths]
 
            for qi in range(len(queries)):
                heap = heaps[qi]
                for di, doc_id in enumerate(doc_ids):
                    score = scores[qi, di].item()
                    if len(heap) < args.top_k:
                        heapq.heappush(heap, (score, doc_id))
                    elif score > heap[0][0]:
                        heapq.heapreplace(heap, (score, doc_id))
 
            del image_embeddings, inputs, scores
 
        except Exception as e:
            print(f"[WARN] Skipping batch {b}: {e}", file=sys.stderr)
 
        finally:
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
 
        if (b + 1) % 50 == 0 or (b + 1) == n_batches:
            print(f"  Processed {min((b + 1) * batch_size, len(image_paths))}/{len(image_paths)} images")
 
    print(f"Writing TREC run file to {args.output} ...")
    with open(args.output, "w", encoding="utf-8") as f:
        for (qid, _qtext), heap in zip(queries, heaps):
            ranked = sorted(heap, key=lambda x: x[0], reverse=True)
            for rank, (score, doc_id) in enumerate(ranked, start=1):
                f.write(f"{qid} Q0 {doc_id} {rank} {score:.6f} {args.run_tag}\n")
 
    print("Done.")
 
 
if __name__ == "__main__":
    main()
 