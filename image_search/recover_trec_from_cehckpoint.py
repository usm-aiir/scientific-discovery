"""
Recovery script: loads a colqwen_finetuned checkpoint (_ckpt.pt),
re-encodes queries, scores, and writes the TREC run file.

Use when the retrieval run finished encoding all images but died before
writing the TREC output.

Usage:
  python recover_trec_from_checkpoint.py \
      --ckpt      run_colqwen_finetuned_ckpt.pt \
      --query_json /path/to/Test.json \
      --output    run_colqwen_finetuned.trec \
      --adapter_dir /path/to/best_adapter      # omit for pretrained
"""

import argparse
import heapq
import json
import sys
from pathlib import Path

import torch
from colpali_engine.models import ColQwen2_5, ColQwen2_5_Processor


def load_queries(json_path):
    with open(json_path, "r", encoding="utf-8") as f:
        data = json.load(f)
    if isinstance(data, dict):
        data = data.get("queries", list(data.values()))
    return [(str(item["query_id"]), item["query"]) for item in data]


def pad_and_cat(embeddings):
    max_seq = max(e.shape[1] for e in embeddings)
    dim = embeddings[0].shape[2]
    padded = []
    for e in embeddings:
        if e.shape[1] < max_seq:
            pad = torch.zeros(e.shape[0], max_seq - e.shape[1], dim, dtype=e.dtype)
            e = torch.cat([e, pad], dim=1)
        padded.append(e)
    return torch.cat(padded, dim=0)


def encode_queries(model, processor, queries, input_device, batch_size=16):
    all_embeddings = []
    texts = [q for _, q in queries]
    for i in range(0, len(texts), batch_size):
        batch_texts = texts[i: i + batch_size]
        inputs = processor.process_queries(batch_texts).to(input_device)
        with torch.no_grad():
            emb = model(**inputs)
        all_embeddings.append(emb.cpu())
    return pad_and_cat(all_embeddings)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--ckpt",        required=True, help="_ckpt.pt from the retrieval run")
    parser.add_argument("--query_json",  required=True)
    parser.add_argument("--output",      default="run_colqwen_finetuned.trec")
    parser.add_argument("--adapter_dir", default=None,
                        help="Path to LoRA adapter dir. Omit for pretrained baseline.")
    parser.add_argument("--model_name",  default="vidore/colqwen2.5-v0.2")
    parser.add_argument("--top_k",       type=int, default=100)
    parser.add_argument("--run_tag",     default="colqwen_finetuned")
    parser.add_argument("--score_batch", type=int, default=2000,
                        help="Images per scoring chunk (reduce if GPU OOM)")
    args = parser.parse_args()

    # ── Load model ────────────────────────────────────────────────────────────
    print(f"Loading model: {args.model_name}")
    if args.adapter_dir:
        from transformers import BitsAndBytesConfig
        from peft import PeftModel
        bnb_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.bfloat16,
        )
        base = ColQwen2_5.from_pretrained(
            args.model_name,
            quantization_config=bnb_config,
            device_map="auto",
        )
        model = PeftModel.from_pretrained(base, args.adapter_dir).eval()
        print(f"  LoRA adapter loaded from {args.adapter_dir}")
    else:
        model = ColQwen2_5.from_pretrained(
            args.model_name,
            torch_dtype=torch.bfloat16,
            device_map="auto",
        ).eval()

    processor = ColQwen2_5_Processor.from_pretrained(args.model_name)
    input_device = next(model.parameters()).device
    print(f"Model input device: {input_device}")

    # ── Load checkpoint ───────────────────────────────────────────────────────
    print(f"Loading checkpoint: {args.ckpt}")
    ckpt = torch.load(args.ckpt, map_location="cpu", weights_only=False)
    image_embeddings = ckpt["embeddings"]
    valid_image_ids  = ckpt["ids"]

    # Checkpoint saves embeddings as a list of per-batch tensors — never
    # concat them all into one giant tensor (17G + model = OOM on 31G RAM).
    # Instead, iterate the list directly, scoring batch by batch.
    if isinstance(image_embeddings, list):
        emb_list = image_embeddings          # list of [B, seq, dim] tensors
        is_list  = True
    else:
        emb_list = None
        is_list  = False

    n_images = len(valid_image_ids)
    print(f"  {n_images} images in checkpoint  (list={is_list})")

    # ── Encode queries ────────────────────────────────────────────────────────
    print("Loading queries...")
    queries = load_queries(args.query_json)
    print(f"  {len(queries)} queries.")

    print("Encoding queries...")
    query_embeddings     = encode_queries(model, processor, queries, input_device)
    query_embeddings_gpu = query_embeddings.to(input_device)
    print(f"  Query embeddings: {query_embeddings_gpu.shape}")

    # ── Score — iterate over checkpoint batches without ever cat-ing them ────
    heaps = [[] for _ in queries]

    if is_list:
        # score one saved batch at a time; pad each to a common seq length
        # with the query embedding's seq length so score_multi_vector is happy
        total_batches = len(emb_list)
        images_done   = 0
        id_iter       = iter(valid_image_ids)

        # find max seq across all saved batches (needed for padding)
        print("  Computing max seq length across batches (one pass)...")
        max_seq = max(e.shape[1] for e in emb_list)
        print(f"  max_seq = {max_seq}")

        for bi, batch_emb in enumerate(emb_list):
            B = batch_emb.shape[0]
            batch_ids = [next(id_iter) for _ in range(B)]

            # pad this batch to max_seq so score_multi_vector accepts it
            if batch_emb.shape[1] < max_seq:
                pad = torch.zeros(B, max_seq - batch_emb.shape[1],
                                  batch_emb.shape[2], dtype=batch_emb.dtype)
                batch_emb = torch.cat([batch_emb, pad], dim=1)

            chunk_gpu = batch_emb.to(input_device)
            with torch.no_grad():
                scores = processor.score_multi_vector(query_embeddings_gpu, chunk_gpu)
            scores = scores.detach().cpu()

            for qi in range(len(queries)):
                heap = heaps[qi]
                for di, doc_id in enumerate(batch_ids):
                    score = scores[qi, di].item()
                    if len(heap) < args.top_k:
                        heapq.heappush(heap, (score, doc_id))
                    elif score > heap[0][0]:
                        heapq.heapreplace(heap, (score, doc_id))

            del chunk_gpu, scores, batch_emb
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

            images_done += B
            if (bi + 1) % 500 == 0 or (bi + 1) == total_batches:
                print(f"  Scored {images_done}/{n_images} images "
                      f"({bi+1}/{total_batches} batches)")

    else:
        # single stacked tensor path (shouldn't hit this for your checkpoint)
        print(f"Scoring (chunk size = {args.score_batch} images)...")
        for start in range(0, n_images, args.score_batch):
            end       = min(start + args.score_batch, n_images)
            chunk_emb = image_embeddings[start:end].to(input_device)
            chunk_ids = valid_image_ids[start:end]
            with torch.no_grad():
                scores = processor.score_multi_vector(query_embeddings_gpu, chunk_emb)
            scores = scores.detach().cpu()
            for qi in range(len(queries)):
                heap = heaps[qi]
                for di, doc_id in enumerate(chunk_ids):
                    score = scores[qi, di].item()
                    if len(heap) < args.top_k:
                        heapq.heappush(heap, (score, doc_id))
                    elif score > heap[0][0]:
                        heapq.heapreplace(heap, (score, doc_id))
            del chunk_emb, scores
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            print(f"  Scored {end}/{n_images} images")

    # ── Write TREC ────────────────────────────────────────────────────────────
    print(f"Writing TREC run file → {args.output}")
    with open(args.output, "w", encoding="utf-8") as f:
        for (qid, _), heap in zip(queries, heaps):
            ranked = sorted(heap, key=lambda x: x[0], reverse=True)
            for rank, (score, doc_id) in enumerate(ranked, start=1):
                f.write(f"{qid} Q0 {doc_id} {rank} {score:.6f} {args.run_tag}\n")

    print("Done.")
    print(f"You can now delete the checkpoint: rm {args.ckpt}")


if __name__ == "__main__":
    main()