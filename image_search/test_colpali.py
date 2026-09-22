"""
ColPali image retrieval -> TREC-format run file.

Given:
  - a directory of images (e.g. 100K figures)
  - a JSON file of queries (list of dicts with at least "query_id" and "query")

Produces a standard TREC run file:
  query_id Q0 doc_id rank score run_tag

doc_id = image filename without its extension (e.g. "2504.15247::F10").

Because embedding + storing 100K multi-vector ColPali embeddings at once is
memory-prohibitive, this script streams images in batches: for each batch it
computes embeddings, scores them against ALL (already-encoded) queries, updates
a running top-100 heap per query, then discards the batch embeddings before
moving to the next one.
"""

import argparse
import heapq
import json
import os
import sys
from pathlib import Path

import torch
from PIL import Image
from transformers import ColPaliForRetrieval, ColPaliProcessor

IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp"}


def find_images(image_dir):
    """Recursively find image files, return sorted list of Paths."""
    image_dir = Path(image_dir)
    paths = [
        p for p in sorted(image_dir.rglob("*"))
        if p.suffix.lower() in IMAGE_EXTS and p.is_file()
    ]
    return paths


def doc_id_from_path(path: Path) -> str:
    """Figure id = filename without extension."""
    return path.stem


def load_queries(query_json_path):
    with open(query_json_path, "r", encoding="utf-8") as f:
        data = json.load(f)
    if isinstance(data, dict):
        # allow either a bare list or {"queries": [...]}
        data = data.get("queries", list(data.values()))
    queries = []
    for item in data:
        qid = item["query_id"]
        qtext = item["query"]
        queries.append((str(qid), qtext))
    return queries


def encode_queries(model, processor, queries, device, batch_size=16):
    """Encode all queries up front; queries are few, so we keep all embeddings."""
    all_embeddings = []
    texts = [q for _, q in queries]
    for i in range(0, len(texts), batch_size):
        batch_texts = texts[i : i + batch_size]
        inputs = processor(text=batch_texts).to(device)
        with torch.no_grad():
            emb = model(**inputs).embeddings
        # keep on CPU to save GPU memory; move back to device only when scoring
        all_embeddings.append(emb.cpu())
    # embeddings are variable-length (per-token), so keep as a list of tensors
    # padded per-batch by the processor; we concatenate along dim 0 but each
    # batch may have different padding length, so pad to the max length here.
    max_len = max(e.shape[1] for e in all_embeddings)
    dim = all_embeddings[0].shape[2]
    padded = []
    for e in all_embeddings:
        if e.shape[1] < max_len:
            pad = torch.zeros(e.shape[0], max_len - e.shape[1], dim, dtype=e.dtype)
            e = torch.cat([e, pad], dim=1)
        padded.append(e)
    query_embeddings = torch.cat(padded, dim=0)
    return query_embeddings


def load_image_safe(path: Path):
    try:
        img = Image.open(path).convert("RGB")
        img.load()
        return img
    except Exception as e:
        print(f"[WARN] Failed to load image {path}: {e}", file=sys.stderr)
        return None


def main():
    parser = argparse.ArgumentParser(description="ColPali retrieval -> TREC run file")
    parser.add_argument(
        "--image_dir",
        default="/mnt/netstore1_home/behrooz.mansouri/SIGIRSciDis/25_04/figures/images",
        help="Directory containing the corpus images (searched recursively).",
    )
    parser.add_argument("--query_json", required=True, help="Path to the queries JSON file.")
    parser.add_argument("--output", default="run.trec", help="Path to write the TREC run file.")
    parser.add_argument("--model_name", default="vidore/colpali-v1.3-hf")
    parser.add_argument("--image_batch_size", type=int, default=8)
    parser.add_argument("--query_batch_size", type=int, default=16)
    parser.add_argument("--top_k", type=int, default=100)
    parser.add_argument("--run_tag", default="colpali")
    parser.add_argument(
        "--device",
        default=None,
        help="Explicit device, e.g. 'cuda:0' or 'cpu'. "
        "Default: cuda:0 if available, else cpu. "
        "(device_map='auto' is deliberately NOT used here -- it triggers a "
        "known cpu/cuda index-mismatch bug in the PaliGemma masking code.)",
    )
    args = parser.parse_args()

    if args.device is not None:
        device = torch.device(args.device)
    else:
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    print(f"Loading model and processor on {device} ...")
    dtype = torch.bfloat16 if device.type == "cuda" else torch.float32
    model = ColPaliForRetrieval.from_pretrained(
        args.model_name,
        torch_dtype=dtype,
    ).to(device)
    model.eval()
    processor = ColPaliProcessor.from_pretrained(args.model_name)

    print("Loading queries...")
    queries = load_queries(args.query_json)
    print(f"  {len(queries)} queries loaded.")

    print("Encoding queries...")
    query_embeddings = encode_queries(
        model, processor, queries, device, batch_size=args.query_batch_size
    )
    query_embeddings_gpu = query_embeddings.to(device)

    print("Finding images...")
    image_paths = find_images(args.image_dir)
    print(f"  {len(image_paths)} images found.")

    # one min-heap of (score, doc_id) per query, capped at top_k
    heaps = [[] for _ in queries]

    batch_size = args.image_batch_size
    n_batches = (len(image_paths) + batch_size - 1) // batch_size

    for b in range(n_batches):
        batch_paths = image_paths[b * batch_size : (b + 1) * batch_size]
        pil_images = []
        valid_paths = []
        for p in batch_paths:
            img = load_image_safe(p)
            if img is not None:
                pil_images.append(img)
                valid_paths.append(p)

        if not pil_images:
            continue

        inputs_images = processor(images=pil_images).to(device)
        with torch.no_grad():
            image_embeddings = model(**inputs_images).embeddings

        # scores: (num_queries, num_images_in_batch)
        scores = processor.score_retrieval(query_embeddings_gpu, image_embeddings)
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

        del image_embeddings, inputs_images, scores
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        if (b + 1) % 50 == 0 or (b + 1) == n_batches:
            print(f"  Processed {min((b + 1) * batch_size, len(image_paths))}/{len(image_paths)} images")

    print(f"Writing TREC run file to {args.output} ...")
    with open(args.output, "w", encoding="utf-8") as f:
        for (qid, _qtext), heap in zip(queries, heaps):
            # sort descending by score
            ranked = sorted(heap, key=lambda x: x[0], reverse=True)
            for rank, (score, doc_id) in enumerate(ranked, start=1):
                f.write(f"{qid} Q0 {doc_id} {rank} {score:.6f} {args.run_tag}\n")

    print("Done.")


if __name__ == "__main__":
    main()