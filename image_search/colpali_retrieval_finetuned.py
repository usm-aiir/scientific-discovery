"""
ColPali Retrieval (Finetuned) -> TREC-format run file.

This is the retrieval script to use after fine-tuning: it runs the base
model plus your fine-tuned LoRA/QLoRA adapter (as produced by
train_colpali_qlora.py). It also still works with no adapter (base model
only) if you omit --lora_adapter -- but if all you need is the plain base
model, the original standalone colpali_retrieval.py script (no eval_utils.py
dependency) does the same thing with one fewer moving part.

Requires eval_utils.py to be present in the same directory.

Given:
  - a directory of images (e.g. 100K figures)
  - a JSON file of queries (list of dicts with at least "query_id" and "query")

Produces a standard TREC run file:
  query_id Q0 doc_id rank score run_tag

doc_id = image filename without its extension (e.g. "2504.15247::F10").

Streams images in batches, scoring each batch against all (already-encoded)
queries and keeping a running top-K heap per query, so memory stays bounded
regardless of corpus size.
"""

import argparse
import heapq
import sys

import torch
from PIL import Image
from transformers import ColPaliForRetrieval, ColPaliProcessor

# This environment's cuDNN fails to initialize (CUDNN_STATUS_NOT_INITIALIZED)
# on the very first conv2d -- SigLIP's patch-embedding layer in the vision
# tower. Same workaround already applied in train_colpali_qlora.py. Falls
# back to PyTorch's non-cuDNN conv kernels; slightly slower, but works.
torch.backends.cudnn.enabled = False

from eval_utils import (
    find_images,
    canonical_doc_items,
    load_queries,
    load_qrels,
    evaluate_run,
)


def encode_queries(model, processor, queries_dict, device, batch_size=16):
    qids = list(queries_dict.keys())
    texts = list(queries_dict.values())
    all_embeddings = []
    for i in range(0, len(texts), batch_size):
        batch_texts = texts[i : i + batch_size]
        inputs = processor(text=batch_texts).to(device)
        with torch.no_grad():
            emb = model(**inputs).embeddings
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


def load_image_safe(path):
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
    parser.add_argument(
        "--lora_adapter",
        default=None,
        help="Path to a fine-tuned LoRA adapter directory (e.g. colpali_qlora_run/best_adapter). "
        "If omitted, runs the base pretrained model.",
    )
    parser.add_argument(
        "--quantize_inference",
        action="store_true",
        help="Load the base model in 4-bit for inference too (matches QLoRA training conditions, "
        "uses less memory, slightly slower). Default: load in bf16/fp32 for best quality/speed.",
    )
    parser.add_argument("--image_batch_size", type=int, default=8)
    parser.add_argument("--query_batch_size", type=int, default=16)
    parser.add_argument("--top_k", type=int, default=100)
    parser.add_argument("--run_tag", default="colpali")
    parser.add_argument(
        "--qrels",
        default=None,
        help="Optional qrels TSV file. If given, nDCG@10/MRR/Recall@10 are computed "
        "against the generated run and printed at the end (e.g. for the test set).",
    )
    parser.add_argument(
        "--device",
        default=None,
        help="Explicit device, e.g. 'cuda:0' or 'cpu'. Default: cuda:0 if available, else cpu. "
        "(device_map='auto' is deliberately NOT used -- see README note about the "
        "cpu/cuda index-mismatch bug in PaliGemma's masking code.)",
    )
    args = parser.parse_args()

    if args.device is not None:
        device = torch.device(args.device)
    else:
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    print(f"Loading model on {device} ...")
    if args.quantize_inference:
        from transformers import BitsAndBytesConfig

        bnb_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.bfloat16,
            bnb_4bit_use_double_quant=True,
            # See train_colpali_qlora.py: without this, embedding_proj_layer
            # gets 4-bit quantized too, its weight.dtype becomes torch.uint8,
            # and ColPaliForRetrieval.forward() casts activations to uint8
            # before .norm() -- crashes with "Expected a floating point ...
            # Got Byte". Keep this small projection head unquantized.
            llm_int8_skip_modules=["embedding_proj_layer"],
        )
        model = ColPaliForRetrieval.from_pretrained(
            args.model_name,
            quantization_config=bnb_config,
            torch_dtype=torch.bfloat16,
            device_map={"": device.index if device.index is not None else 0},
        )
    else:
        dtype = torch.bfloat16 if device.type == "cuda" else torch.float32
        model = ColPaliForRetrieval.from_pretrained(
            args.model_name,
            torch_dtype=dtype,
        ).to(device)

    if args.lora_adapter:
        from peft import PeftModel

        print(f"Loading LoRA adapter from {args.lora_adapter} ...")
        model = PeftModel.from_pretrained(model, args.lora_adapter)

    model.eval()
    processor = ColPaliProcessor.from_pretrained(
        args.lora_adapter if args.lora_adapter else args.model_name
    )

    print("Loading queries...")
    queries = load_queries(args.query_json)
    print(f"  {len(queries)} queries loaded.")

    print("Encoding queries...")
    qids, query_embeddings = encode_queries(
        model, processor, queries, device, batch_size=args.query_batch_size
    )
    query_embeddings_gpu = query_embeddings.to(device)

    print("Indexing images...")
    image_paths = find_images(args.image_dir)
    # canonical_doc_items() gives exactly ONE (doc_id, path) entry per physical
    # file, unlike build_doc_index() which also maps in raw-filename-stem
    # aliases for lookup purposes -- iterating that dict would embed and
    # score every image twice under two different doc_ids.
    doc_items = canonical_doc_items(args.image_dir, image_paths)
    print(f"  {len(image_paths)} images found, {len(doc_items)} unique doc_ids indexed.")

    heaps = {qid: [] for qid in qids}

    batch_size = args.image_batch_size
    n_batches = (len(doc_items) + batch_size - 1) // batch_size

    for b in range(n_batches):
        batch_items = doc_items[b * batch_size : (b + 1) * batch_size]
        pil_images, doc_ids = [], []
        for did, p in batch_items:
            img = load_image_safe(p)
            if img is not None:
                pil_images.append(img)
                doc_ids.append(did)
        if not pil_images:
            continue

        inputs_images = processor(images=pil_images).to(device)
        with torch.no_grad():
            image_embeddings = model(**inputs_images).embeddings

        scores = processor.score_retrieval(query_embeddings_gpu, image_embeddings)
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
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        if (b + 1) % 50 == 0 or (b + 1) == n_batches:
            print(f"  Processed {min((b + 1) * batch_size, len(doc_items))}/{len(doc_items)} images")

    print(f"Writing TREC run file to {args.output} ...")
    with open(args.output, "w", encoding="utf-8") as f:
        for qid, heap in heaps.items():
            ranked = sorted(heap, key=lambda x: x[0], reverse=True)
            for rank, (score, doc_id) in enumerate(ranked, start=1):
                f.write(f"{qid} Q0 {doc_id} {rank} {score:.6f} {args.run_tag}\n")
    print("Done.")

    if args.qrels:
        print("Computing metrics against provided qrels...")
        qrels = load_qrels(args.qrels)
        run = {qid: [d for _, d in sorted(heap, key=lambda x: x[0], reverse=True)] for qid, heap in heaps.items()}
        metrics = evaluate_run(run, qrels, k_list=(10, 100))
        for name, val in metrics.items():
            print(f"  {name}: {val:.4f}")


if __name__ == "__main__":
    main()