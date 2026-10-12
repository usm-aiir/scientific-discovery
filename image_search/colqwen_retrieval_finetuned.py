"""
ColQwen2.5 Retrieval (Finetuned) -> TREC-format run file.

Runs the base ColQwen2.5 model plus an optional fine-tuned LoRA/QLoRA adapter
(as produced by train_colqwen_qlora.py). Omit --lora_adapter to run the base
pretrained model only.

Requires eval_utils.py to be present in the same directory.

Given:
  - a directory of images (e.g. 100K figures)
  - a JSON file of queries (list of dicts with at least "query_id" and "query")

Produces a standard TREC run file:
  query_id Q0 doc_id rank score run_tag

doc_id = canonical id from eval_utils.canonical_doc_items (the filename stem,
or the subfolder path joined with '_' if images live in subfolders).

Streams images in batches, scoring each batch against all (already-encoded)
queries and keeping a running top-K heap per query, so memory stays bounded
regardless of corpus size.

Changes vs. previous version:
  - a failed batch now STOPS the run by default (after checkpointing) instead of
    being silently skipped; pass --skip_failed_batches to tolerate failures,
    and the total is reported at the end either way
  - pixel_values cast to bf16 (matches training)
  - 4-bit path now loads non-quantized modules in bf16 too
  - adapter diagnostics printed after loading
  - default --output / --run_tag derived from the configuration, so base /
    fine-tuned / bf16 / 4-bit runs never overwrite each other
  - checkpoints record the run configuration and refuse to resume a different one
  - qrels doc_id coverage check before the long scan

Checkpointing: heap state + last completed batch index are saved every
--checkpoint_every batches. On a crash, re-run the same command and the
script resumes automatically from the last checkpoint.

Usage:
    python colqwen_retrieval_finetuned.py \
        --query_json /path/to/Test.json \
        --image_dir  /path/to/figures/images \
        --lora_adapter colqwen_qlora_run/best_adapter \
        --quantize_inference \
        --qrels /path/to/Test_figure_qrels.tsv
"""

import argparse
import heapq
import os
import sys
from pathlib import Path

import torch
from colpali_engine.models import ColQwen2_5, ColQwen2_5_Processor

from eval_utils import (
    find_images,
    canonical_doc_items,
    load_queries,
    load_qrels,
    evaluate_run,
    safe_open_image,
    report_qrels_coverage,
)


# ---------------------------------------------------------------------------
# Diagnostics
# ---------------------------------------------------------------------------

def describe_adapters(model, label):
    """Print which PEFT adapters are attached and active on `model`."""
    print(f"[ADAPTERS] {label}: model type = {type(model).__name__}")
    cfg = getattr(model, "peft_config", None)
    if not cfg:
        print("  No PEFT adapters attached. If you expected the pretrained ColQwen "
              "retrieval adapter here, the model is running WITHOUT it.")
        return
    for name, c in cfg.items():
        print(f"  adapter '{name}': r={getattr(c, 'r', None)}, "
              f"base_model={getattr(c, 'base_model_name_or_path', None)}")
    try:
        active = getattr(model, "active_adapters", None)
        active = active() if callable(active) else active
    except Exception as e:  # noqa: BLE001
        active = f"(could not query: {e})"
    print(f"  active adapters: {active}")


# ---------------------------------------------------------------------------
# Query encoding
# ---------------------------------------------------------------------------

def encode_queries(model, processor, queries_dict, device, batch_size=16):
    qids  = list(queries_dict.keys())
    texts = list(queries_dict.values())
    all_embeddings = []
    for i in range(0, len(texts), batch_size):
        batch_texts = texts[i : i + batch_size]
        inputs = processor.process_queries(batch_texts).to(device)
        with torch.no_grad():
            emb = model(**inputs)   # ColQwen2_5 returns the embedding tensor directly
        all_embeddings.append(emb.cpu())
    max_len = max(e.shape[1] for e in all_embeddings)
    dim = all_embeddings[0].shape[2]
    padded = []
    for e in all_embeddings:
        if e.shape[1] < max_len:
            pad = torch.zeros(e.shape[0], max_len - e.shape[1], dim, dtype=e.dtype)
            e = torch.cat([e, pad], dim=1)
        padded.append(e)
    return qids, torch.cat(padded, dim=0)


# ---------------------------------------------------------------------------
# Checkpointing
# ---------------------------------------------------------------------------

def ckpt_path(output_trec):
    # splitext (not .replace(".trec", ...)) so an output name without ".trec"
    # can never make the checkpoint overwrite the run file itself.
    root, _ = os.path.splitext(output_trec)
    return root + "_ckpt.pt"


def save_checkpoint(path, heaps, last_batch_idx, run_config, stats):
    """Save heap state, the index of the last COMPLETED batch, and what produced them."""
    torch.save(
        {"heaps": heaps, "last_batch_idx": last_batch_idx, "run_config": run_config, "stats": stats},
        path,
    )


def load_checkpoint(path, run_config):
    """Returns (heaps, last_batch_idx, stats) or (None, -1, None) if no checkpoint exists."""
    if not os.path.exists(path):
        return None, -1, None
    print(f"Found checkpoint: {path}")
    data = torch.load(path, map_location="cpu", weights_only=False)
    saved_config = data.get("run_config")
    if saved_config != run_config:
        raise SystemExit(
            "[ERROR] This checkpoint was written by a DIFFERENT configuration, so resuming "
            "would mix two models' results.\n"
            f"  checkpoint config: {saved_config}\n"
            f"  current config   : {run_config}\n"
            f"Delete {path} or use a different --output."
        )
    heaps = data["heaps"]
    last_batch_idx = data["last_batch_idx"]
    n_docs = sum(len(h) for h in heaps.values())
    print(f"  Resuming: {n_docs} heap entries, last completed batch {last_batch_idx}")
    return heaps, last_batch_idx, data.get("stats")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="ColQwen2.5 retrieval -> TREC run file")
    parser.add_argument(
        "--image_dir",
        default="/mnt/netstore1_home/adah.holt/figures/images",
        help="Directory containing the corpus images (searched recursively).",
    )
    parser.add_argument("--query_json", required=True,  help="Path to the queries JSON file.")
    parser.add_argument("--output", default=None,
                        help="Path to write the TREC run file. Default is derived from the "
                             "configuration, e.g. run_colqwen_base_bf16.trec.")
    parser.add_argument("--model_name", default="vidore/colqwen2.5-v0.2")
    parser.add_argument(
        "--lora_adapter",
        default=None,
        help="Path to a fine-tuned LoRA adapter directory (e.g. colqwen_qlora_run/best_adapter). "
             "If omitted, runs the base pretrained model.",
    )
    parser.add_argument(
        "--quantize_inference",
        action="store_true",
        help="Load the base model in 4-bit for inference. This MATCHES the QLoRA training "
             "conditions; compare fine-tuned vs. base under the same setting.",
    )
    parser.add_argument("--image_batch_size", type=int, default=4)
    parser.add_argument("--query_batch_size", type=int, default=16)
    parser.add_argument("--top_k",            type=int, default=100)
    parser.add_argument("--run_tag",          default=None,
                        help="Run tag written into the TREC file. Default derived from the configuration.")
    parser.add_argument(
        "--checkpoint_every", type=int, default=500,
        help="Save a heap checkpoint every N batches (default 500 ≈ 2000 images at batch_size=4). "
             "Re-run the same command after a crash to resume automatically.",
    )
    parser.add_argument(
        "--skip_failed_batches", action="store_true",
        help="Skip batches that raise an error instead of stopping. Skipped images can never "
             "be retrieved, which lowers metrics -- the total is reported at the end.",
    )
    parser.add_argument(
        "--qrels",
        default=None,
        help="Optional qrels TSV file. If given, doc_id coverage is checked up front and "
             "nDCG@10/MRR/Recall@10 are computed against the generated run at the end.",
    )
    parser.add_argument(
        "--device",
        default=None,
        help="Explicit device, e.g. 'cuda:0'. Default: auto-selected via device_map='auto'.",
    )
    args = parser.parse_args()

    # ------------------------------------------------------------------ naming
    adapter_tag = (
        "_".join(Path(args.lora_adapter).resolve().parts[-2:]) if args.lora_adapter else "base"
    )
    precision_tag = "4bit" if args.quantize_inference else "bf16"
    if args.output is None:
        args.output = f"run_colqwen_{adapter_tag}_{precision_tag}.trec"
    if args.run_tag is None:
        args.run_tag = f"colqwen_{adapter_tag}_{precision_tag}"
    print(f"Output run file: {args.output}  (run_tag={args.run_tag})")

    run_config = {
        "model_name": args.model_name,
        "lora_adapter": os.path.abspath(args.lora_adapter) if args.lora_adapter else None,
        "quantize_inference": args.quantize_inference,
        "image_dir": os.path.abspath(args.image_dir),
        "query_json": os.path.abspath(args.query_json),
        "image_batch_size": args.image_batch_size,
        "top_k": args.top_k,
    }

    if args.lora_adapter and not args.quantize_inference:
        print("[NOTE] The adapter was trained against 4-bit weights but is being applied to bf16 "
              "weights. Also run with --quantize_inference for a matched comparison.")

    # ------------------------------------------------------------------ model
    print(f"Loading model: {args.model_name} ({precision_tag})")
    device_map = {"": torch.device(args.device).index or 0} if args.device else "auto"
    if args.quantize_inference:
        from transformers import BitsAndBytesConfig
        bnb_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.bfloat16,
            bnb_4bit_use_double_quant=True,
            # Keep the ColBERT projection head in full precision; if quantized
            # its weight.dtype becomes uint8 and subsequent casts can crash.
            llm_int8_skip_modules=["custom_text_proj"],
        )
        model = ColQwen2_5.from_pretrained(
            args.model_name,
            quantization_config=bnb_config,
            torch_dtype=torch.bfloat16,   # same as training; non-quantized modules in bf16
            device_map=device_map,
        )
    else:
        model = ColQwen2_5.from_pretrained(
            args.model_name,
            torch_dtype=torch.bfloat16,
            device_map=device_map,
        )
    describe_adapters(model, "after loading pretrained model")

    if args.lora_adapter:
        from peft import PeftModel
        print(f"Loading LoRA adapter from {args.lora_adapter} ...")
        model = PeftModel.from_pretrained(model, args.lora_adapter)
        describe_adapters(model, "after loading fine-tuned adapter")

    model.eval()
    processor = ColQwen2_5_Processor.from_pretrained(
        args.lora_adapter if args.lora_adapter else args.model_name
    )

    # Handles device_map="auto" multi-GPU splits without hardcoding cuda:0.
    input_device = next(model.parameters()).device
    print(f"Model input device: {input_device}")

    # ------------------------------------------------------------------ queries
    print("Loading queries...")
    queries = load_queries(args.query_json)
    print(f"  {len(queries)} queries loaded.")

    # ------------------------------------------------------------------ images
    print("Indexing images...")
    image_paths = find_images(args.image_dir)
    doc_items = canonical_doc_items(args.image_dir, image_paths)
    print(f"  {len(image_paths)} images found, {len(doc_items)} unique doc_ids indexed.")

    qrels = None
    if args.qrels:
        qrels = load_qrels(args.qrels)
        report_qrels_coverage(qrels, [d for d, _ in doc_items], label="test qrels vs. corpus ids")

    print("Encoding queries...")
    qids, query_embeddings = encode_queries(
        model, processor, queries, input_device, batch_size=args.query_batch_size
    )
    query_embeddings_gpu = query_embeddings.to(input_device)

    # ------------------------------------------------------------------ checkpoint
    ckpt_file = ckpt_path(args.output)
    saved_heaps, last_batch_idx, saved_stats = load_checkpoint(ckpt_file, run_config)

    if saved_heaps is not None:
        heaps = {qid: list(saved_heaps.get(qid, [])) for qid in qids}
        for qid in qids:
            heapq.heapify(heaps[qid])
        start_batch = last_batch_idx + 1
        print(f"  Skipping batches 0–{last_batch_idx}, resuming from batch {start_batch}.")
    else:
        heaps = {qid: [] for qid in qids}
        start_batch = 0

    stats = saved_stats or {"failed_images": 0, "skipped_batches": 0, "skipped_batch_images": 0}

    # ------------------------------------------------------------------ encode + score
    batch_size = args.image_batch_size
    n_batches  = (len(doc_items) + batch_size - 1) // batch_size
    batches_since_ckpt = 0

    for b in range(start_batch, n_batches):
        batch_items = doc_items[b * batch_size : (b + 1) * batch_size]
        pil_images, doc_ids = [], []
        for did, p in batch_items:
            img = safe_open_image(p)
            if img is not None:
                pil_images.append(img)
                doc_ids.append(did)
            else:
                stats["failed_images"] += 1

        if pil_images:
            try:
                inputs_images = processor.process_images(pil_images).to(input_device)
                if "pixel_values" in inputs_images:
                    # Match training: the vision encoder runs in bf16.
                    inputs_images["pixel_values"] = inputs_images["pixel_values"].to(torch.bfloat16)
                with torch.no_grad():
                    image_embeddings = model(**inputs_images)

                scores = processor.score_multi_vector(query_embeddings_gpu, image_embeddings)
                scores = scores.detach().cpu()

                for qi, qid in enumerate(qids):
                    heap = heaps[qid]
                    for di, doc_id in enumerate(doc_ids):
                        score = scores[qi, di].item()
                        if len(heap) < args.top_k:
                            heapq.heappush(heap, (score, doc_id))
                        elif score > heap[0][0]:
                            heapq.heapreplace(heap, (score, doc_id))

                del image_embeddings, inputs_images, scores

            except Exception as e:  # noqa: BLE001
                if not args.skip_failed_batches:
                    save_checkpoint(ckpt_file, heaps, b - 1, run_config, stats)
                    print(f"[ERROR] Batch {b} failed: {e}\n"
                          f"  Checkpoint saved up to batch {b - 1}. Fix the error and re-run "
                          f"the same command to resume, or pass --skip_failed_batches.",
                          file=sys.stderr)
                    raise
                stats["skipped_batches"] += 1
                stats["skipped_batch_images"] += len(doc_ids)
                print(f"[WARN] Skipping batch {b}: {e}", file=sys.stderr)

            finally:
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()

        batches_since_ckpt += 1

        # Periodic checkpoint
        if batches_since_ckpt >= args.checkpoint_every:
            save_checkpoint(ckpt_file, heaps, b, run_config, stats)
            images_done = min((b + 1) * batch_size, len(doc_items))
            print(f"  Checkpoint saved at batch {b} ({images_done}/{len(doc_items)} images)")
            batches_since_ckpt = 0

        if (b + 1) % 50 == 0 or (b + 1) == n_batches:
            print(f"  Processed {min((b + 1) * batch_size, len(doc_items))}/{len(doc_items)} images")

    # ------------------------------------------------------------------ summary
    print("Scan summary:")
    print(f"  images that failed to decode : {stats['failed_images']}")
    print(f"  batches skipped after errors : {stats['skipped_batches']} "
          f"({stats['skipped_batch_images']} images never scored)")
    if stats["failed_images"] or stats["skipped_batches"]:
        print("  [WARN] Some of the corpus was never scored; metrics are lower than they "
              "should be for this model.")

    # ------------------------------------------------------------------ write TREC
    print(f"Writing TREC run file to {args.output} ...")
    with open(args.output, "w", encoding="utf-8") as f:
        for qid, heap in heaps.items():
            ranked = sorted(heap, key=lambda x: x[0], reverse=True)
            for rank, (score, doc_id) in enumerate(ranked, start=1):
                f.write(f"{qid} Q0 {doc_id} {rank} {score:.6f} {args.run_tag}\n")
    print("Done.")

    # Clean up checkpoint once TREC file is written successfully
    if os.path.exists(ckpt_file):
        os.remove(ckpt_file)
        print("Checkpoint deleted (run complete).")

    # ------------------------------------------------------------------ eval
    if qrels is not None:
        print("Computing metrics against provided qrels...")
        run = {
            qid: [d for _, d in sorted(heap, key=lambda x: x[0], reverse=True)]
            for qid, heap in heaps.items()
        }
        metrics = evaluate_run(run, qrels, k_list=(10, 100))
        for name, val in metrics.items():
            print(f"  {name}: {val:.4f}")


if __name__ == "__main__":
    main()