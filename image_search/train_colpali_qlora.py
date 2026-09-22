"""
python -m pip install --upgrade "torch>=2.11.0" torchvision

QLoRA fine-tuning for ColPali (vidore/colpali-v1.3-hf).

Trains on (query, positive-figure) pairs built from train.json + Train_figure_qrels.tsv,
using in-batch negatives (bidirectional ColBERT-style contrastive loss).
After every epoch, evaluates on val.json + Val_figure_qrels.tsv and keeps the
LoRA adapter with the best validation nDCG@10.

The base model is loaded in 4-bit (QLoRA); only LoRA adapters on the
language-model's attention/MLP projections are trained. The vision tower and
base weights stay frozen and quantized.

Usage:
  python train_colpali_qlora.py \
      --image_dir /mnt/netstore1_home/behrooz.mansouri/SIGIRSciDis/25_04/figures/images \
      --train_json train.json --train_qrels Train_figure_qrels.tsv \
      --val_json   val.json   --val_qrels   Val_figure_qrels.tsv \
      --output_dir ./colpali_qlora_run

Requires: transformers, torch, peft, bitsandbytes, accelerate, pillow
  pip install -U peft bitsandbytes accelerate
"""

import argparse
import os
import random

import torch
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from PIL import Image

from transformers import (
    ColPaliForRetrieval,
    ColPaliProcessor,
    BitsAndBytesConfig,
    get_cosine_schedule_with_warmup,
)
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training

from eval_utils import (
    find_images,
    build_doc_index,
    canonical_doc_items,
    load_queries,
    load_qrels,
    evaluate_run,
    safe_open_image,
)

# Fallback size used whenever an image fails to decode (corrupt/truncated
# file, oversized metadata, etc.) and gets replaced by a blank placeholder.
# Matches PaliGemma's native resolution; the processor resizes anyway.
PLACEHOLDER_IMAGE_SIZE = (448, 448)

# Restrict LoRA to the language-model's projection layers (standard ColPali
# fine-tuning recipe); leaves the SigLIP vision tower frozen and untouched.
LORA_TARGET_REGEX = r".*language_model.*\.(q_proj|k_proj|v_proj|o_proj|gate_proj|up_proj|down_proj)$"
# Add these configuration lines:
torch.backends.cudnn.enabled = False

# ---------------------------------------------------------------------------
# Dataset
# ---------------------------------------------------------------------------

class QueryFigurePairDataset(Dataset):
    """One example per (query, positive figure) pair found in the qrels file."""

    def __init__(self, queries, qrels, doc_index, min_relevance=1):
        self.pairs = []
        skipped_missing_query = 0
        skipped_missing_doc = 0
        missing_examples = []
        for qid, docs in qrels.items():
            if qid not in queries:
                skipped_missing_query += 1
                continue
            for docid, rel in docs.items():
                if rel < min_relevance:
                    continue
                if docid not in doc_index:
                    skipped_missing_doc += 1
                    if len(missing_examples) < 10:
                        missing_examples.append(docid)
                    continue
                self.pairs.append((qid, docid))
        if skipped_missing_query:
            print(f"[WARN] {skipped_missing_query} qrels queries had no matching entry in the query JSON.")
        if skipped_missing_doc:
            print(f"[WARN] {skipped_missing_doc} qrels (query, doc) pairs referenced an image not found on disk.")
        print(f"Built {len(self.pairs)} training pairs.")

        if len(self.pairs) == 0 and (skipped_missing_doc or skipped_missing_query):
            print("\n[DIAGNOSTIC] No training pairs were built -- doc_ids in the qrels file "
                  "don't match doc_ids built from the image directory. Compare these:")
            print(f"  Sample qrels doc_ids expected : {missing_examples}")
            print(f"  Sample doc_ids actually indexed: {list(doc_index.keys())[:10]}")
            print("  If these look structurally different (e.g. 'paper::fig' vs 'fig', or a "
                  "different separator), the image folder layout doesn't match what "
                  "build_doc_index() assumes. Check the actual file paths, e.g.:\n"
                  "    find <image_dir> -iname '*<some_paper_id>*'\n")

    def __len__(self):
        return len(self.pairs)

    def __getitem__(self, idx):
        return self.pairs[idx]


class Collator:
    def __init__(self, processor, queries, doc_index):
        self.processor = processor
        self.queries = queries
        self.doc_index = doc_index

    def __call__(self, batch):
        qids, docids = zip(*batch)
        texts = [self.queries[q] for q in qids]
        images = [safe_open_image(self.doc_index[d], fallback_size=PLACEHOLDER_IMAGE_SIZE) for d in docids]
        image_inputs = self.processor(images=images)
        text_inputs = self.processor(text=list(texts))
        return image_inputs, text_inputs, qids, docids


# ---------------------------------------------------------------------------
# Loss
# ---------------------------------------------------------------------------

def colbert_inbatch_loss(query_embeddings, doc_embeddings, processor, temperature=1.0):
    """Bidirectional in-batch contrastive loss (standard ColBERT/ColPali training loss)."""
    scores = processor.score_retrieval(query_embeddings, doc_embeddings)  # (B, B)
    scores = scores / temperature
    labels = torch.arange(scores.shape[0], device=scores.device)
    loss_q2d = F.cross_entropy(scores, labels)
    loss_d2q = F.cross_entropy(scores.t(), labels)
    return (loss_q2d + loss_d2q) / 2


# ---------------------------------------------------------------------------
# Validation
# ---------------------------------------------------------------------------

@torch.no_grad()
def embed_images(model, processor, doc_ids, doc_index, device, batch_size):
    embeddings = []
    for i in range(0, len(doc_ids), batch_size):
        batch_ids = doc_ids[i : i + batch_size]
        images = [safe_open_image(doc_index[d], fallback_size=PLACEHOLDER_IMAGE_SIZE) for d in batch_ids]
        inputs = processor(images=images).to(device)
        out = model(**inputs).embeddings
        embeddings.append(out.cpu())
    max_len = max(e.shape[1] for e in embeddings)
    dim = embeddings[0].shape[2]
    padded = []
    for e in embeddings:
        if e.shape[1] < max_len:
            pad = torch.zeros(e.shape[0], max_len - e.shape[1], dim, dtype=e.dtype)
            e = torch.cat([e, pad], dim=1)
        padded.append(e)
    return torch.cat(padded, dim=0)


@torch.no_grad()
def embed_queries(model, processor, qids, queries, device, batch_size):
    embeddings = []
    texts = [queries[q] for q in qids]
    for i in range(0, len(texts), batch_size):
        batch_texts = texts[i : i + batch_size]
        inputs = processor(text=batch_texts).to(device)
        out = model(**inputs).embeddings
        embeddings.append(out.cpu())
    max_len = max(e.shape[1] for e in embeddings)
    dim = embeddings[0].shape[2]
    padded = []
    for e in embeddings:
        if e.shape[1] < max_len:
            pad = torch.zeros(e.shape[0], max_len - e.shape[1], dim, dtype=e.dtype)
            e = torch.cat([e, pad], dim=1)
        padded.append(e)
    return torch.cat(padded, dim=0)


@torch.no_grad()
def run_retrieval_eval(model, processor, queries, qrels, doc_index, device,
                        image_batch_size, query_batch_size, top_k, full_corpus_ids=None):
    """
    Encodes queries + a candidate doc pool, scores them, builds a run dict,
    and returns nDCG@10 / MRR / etc. against qrels.

    candidate pool = full_corpus_ids if given, else the union of doc_ids that
    appear (as positive or otherwise judged) in the qrels file -- this keeps
    per-epoch validation fast; pass --eval_full_corpus for a true full-corpus
    evaluation (slow, matches the actual test-time deployment setting).
    """
    model.eval()

    qids = [q for q in qrels.keys() if q in queries]
    query_embeddings = embed_queries(model, processor, qids, queries, device, query_batch_size)
    query_embeddings = query_embeddings.to(device)

    if full_corpus_ids is not None:
        candidate_ids = full_corpus_ids
    else:
        candidate_ids = sorted({d for docs in qrels.values() for d in docs.keys() if d in doc_index})

    run_scores = {q: [] for q in qids}
    for i in range(0, len(candidate_ids), image_batch_size):
        batch_ids = candidate_ids[i : i + image_batch_size]
        images = [safe_open_image(doc_index[d], fallback_size=PLACEHOLDER_IMAGE_SIZE) for d in batch_ids]
        inputs = processor(images=images).to(device)
        doc_embeddings = model(**inputs).embeddings
        scores = processor.score_retrieval(query_embeddings, doc_embeddings).detach().cpu()
        for qi, qid in enumerate(qids):
            for di, docid in enumerate(batch_ids):
                run_scores[qid].append((scores[qi, di].item(), docid))
        del inputs, doc_embeddings, scores
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    run = {q: [d for _, d in sorted(s, key=lambda x: x[0], reverse=True)[:top_k]] for q, s in run_scores.items()}
    metrics = evaluate_run(run, qrels, k_list=(10,))
    model.train()
    return metrics, run_scores


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="QLoRA fine-tuning for ColPali")
    parser.add_argument(
        "--image_dir",
        default="/mnt/netstore1_home/behrooz.mansouri/SIGIRSciDis/25_04/figures/images",
    )
    parser.add_argument("--train_json", required=True)
    parser.add_argument("--train_qrels", required=True)
    parser.add_argument("--val_json", required=True)
    parser.add_argument("--val_qrels", required=True)
    parser.add_argument("--output_dir", default="./colpali_qlora_run")
    parser.add_argument("--model_name", default="vidore/colpali-v1.3-hf")

    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--train_batch_size", type=int, default=2)
    parser.add_argument("--grad_accum_steps", type=int, default=8)
    parser.add_argument("--eval_image_batch_size", type=int, default=4)
    parser.add_argument("--eval_query_batch_size", type=int, default=16)
    parser.add_argument("--lr", type=float, default=2e-4)
    parser.add_argument("--warmup_ratio", type=float, default=0.05)
    parser.add_argument("--temperature", type=float, default=1.0)
    parser.add_argument("--top_k_eval", type=int, default=10)
    parser.add_argument("--eval_full_corpus", action="store_true",
                         help="Evaluate validation nDCG@10 against the FULL image corpus "
                              "instead of just the docs referenced in the val qrels file. "
                              "Much slower but matches deployment conditions.")

    parser.add_argument("--lora_r", type=int, default=16)
    parser.add_argument("--lora_alpha", type=int, default=32)
    parser.add_argument("--lora_dropout", type=float, default=0.05)
    parser.add_argument("--gradient_checkpointing", action="store_true", default=False,
                         help="Try to enable gradient checkpointing on the inner language model to reduce "
                              "activation memory. OFF by default: it's known to be broken with some "
                              "transformers/peft/bitsandbytes/ColPali version combinations (surfaces as "
                              "'Expected floating point tensor, got Byte' regardless of use_reentrant). "
                              "If your environment doesn't hit that, this can meaningfully raise the "
                              "batch size you can afford -- worth trying, but verify a training step "
                              "actually runs before trusting it.")

    parser.add_argument("--device", default=None, help="e.g. 'cuda:0'. Default: cuda:0 if available.")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--num_workers", type=int, default=4)
    args = parser.parse_args()

    random.seed(args.seed)
    torch.manual_seed(args.seed)

    device = torch.device(args.device) if args.device else torch.device(
        "cuda:0" if torch.cuda.is_available() else "cpu"
    )
    if device.type != "cuda":
        raise SystemExit("QLoRA (4-bit) requires a CUDA GPU. No GPU detected / selected.")

    os.makedirs(args.output_dir, exist_ok=True)
    best_adapter_dir = os.path.join(args.output_dir, "best_adapter")

    print("Indexing image corpus...")
    image_paths = find_images(args.image_dir)
    doc_index = build_doc_index(args.image_dir, image_paths)  # id lookup (incl. aliases)
    canonical_items = canonical_doc_items(args.image_dir, image_paths)  # 1 entry/file, for full-corpus scans
    print(f"  {len(canonical_items)} unique images indexed ({len(doc_index)} lookup keys incl. aliases).")

    print("Loading data...")
    train_queries = load_queries(args.train_json)
    train_qrels = load_qrels(args.train_qrels)
    val_queries = load_queries(args.val_json)
    val_qrels = load_qrels(args.val_qrels)

    train_dataset = QueryFigurePairDataset(train_queries, train_qrels, doc_index)

    print("Loading base model in 4-bit (QLoRA) ...")
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16,
        bnb_4bit_use_double_quant=True,
        # ColPaliForRetrieval.forward() does:
        #   proj_dtype = self.embedding_proj_layer.weight.dtype
        #   embeddings = self.embedding_proj_layer(last_hidden_states.to(proj_dtype))
        #   embeddings = embeddings / embeddings.norm(...)
        # If embedding_proj_layer gets 4-bit quantized like everything else,
        # its weight.dtype becomes torch.uint8 (bnb's packed storage dtype),
        # so last_hidden_states gets cast to uint8 and the later .norm() call
        # crashes with "Expected a floating point ... Got Byte". This tiny
        # projection head (a few million params) buys nothing from
        # quantization anyway, so keep it in full precision.
        llm_int8_skip_modules=["embedding_proj_layer"],
    )
    # device_map pinned to a single device on purpose -- see the retrieval
    # script's note about the accelerate multi-device dispatch bug in the
    # PaliGemma causal-mask code when using device_map="auto".
    model = ColPaliForRetrieval.from_pretrained(
        args.model_name,
        quantization_config=bnb_config,
        torch_dtype=torch.bfloat16,
        device_map={"": device.index if device.index is not None else 0},
    )
    processor = ColPaliProcessor.from_pretrained(args.model_name)

    # use_gradient_checkpointing=False here: ColPaliForRetrieval's top-level
    # wrapper doesn't implement gradient_checkpointing_enable() itself, so
    # letting prepare_model_for_kbit_training call it crashes. We instead
    # enable checkpointing directly on the inner language model below, which
    # is what actually matters for activation memory.
    model = prepare_model_for_kbit_training(model, use_gradient_checkpointing=False)

    if args.gradient_checkpointing:
        # Needed for checkpointing to backprop correctly through frozen/
        # quantized layers (prepare_model_for_kbit_training would normally
        # do this itself, but we skipped its checkpointing branch above).
        if hasattr(model, "enable_input_require_grads"):
            model.enable_input_require_grads()

        if hasattr(model, "vlm") and hasattr(model.vlm, "gradient_checkpointing_enable"):
            # use_reentrant=True is deliberate: non-reentrant checkpointing
            # (use_reentrant=False) has a known bad interaction with
            # bitsandbytes 4-bit layers where it can hand back the packed
            # uint8 quantized-weight buffer instead of the real activation
            # (surfaces as "Expected floating point tensor, got Byte").
            # Reentrant checkpointing doesn't go through that code path.
            model.vlm.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": True})
            model.config.use_cache = False
            print("Gradient checkpointing enabled on the inner vlm (language model) submodule, "
                  "use_reentrant=True.")
        else:
            print("[WARN] Could not find model.vlm to enable gradient checkpointing on; "
                  "proceeding without it (higher activation memory use).")
    else:
        print("Gradient checkpointing disabled (--no_gradient_checkpointing). "
              "Reduce --train_batch_size / --eval_image_batch_size if you hit OOM.")
    lora_config = LoraConfig(
        r=args.lora_r,
        lora_alpha=args.lora_alpha,
        lora_dropout=args.lora_dropout,
        bias="none",
        target_modules=LORA_TARGET_REGEX,
        # task_type intentionally omitted: yields a generic PeftModel whose
        # forward() just passes through to the base model, preserving the
        # `.embeddings` output that ColPaliForRetrieval returns.
    )
    model = get_peft_model(model, lora_config)
    model.print_trainable_parameters()

    collator = Collator(processor, train_queries, doc_index)
    train_loader = DataLoader(
        train_dataset,
        batch_size=args.train_batch_size,
        shuffle=True,
        collate_fn=collator,
        num_workers=args.num_workers,
        drop_last=True,  # keep batches full-size for stable in-batch negatives
    )

    optimizer = torch.optim.AdamW(
        [p for p in model.parameters() if p.requires_grad], lr=args.lr
    )
    steps_per_epoch = len(train_loader) // args.grad_accum_steps
    total_steps = max(1, steps_per_epoch * args.epochs)
    scheduler = get_cosine_schedule_with_warmup(
        optimizer,
        num_warmup_steps=int(total_steps * args.warmup_ratio),
        num_training_steps=total_steps,
    )

    full_corpus_ids = [doc_id for doc_id, _ in canonical_items] if args.eval_full_corpus else None

    best_score = -1.0
    global_step = 0

    for epoch in range(args.epochs):
        model.train()
        running_loss = 0.0
        optimizer.zero_grad()

        for step, (image_inputs, text_inputs, qids, docids) in enumerate(train_loader):
            image_inputs = {k: v.to(device) for k, v in image_inputs.items()}
            if "pixel_values" in image_inputs:
                image_inputs["pixel_values"] = image_inputs["pixel_values"].to(torch.bfloat16)
            text_inputs = {k: v.to(device) for k, v in text_inputs.items()}

            doc_embeddings = model(**image_inputs).embeddings
            query_embeddings = model(**text_inputs).embeddings

            loss = colbert_inbatch_loss(query_embeddings, doc_embeddings, processor, args.temperature)
            (loss / args.grad_accum_steps).backward()
            running_loss += loss.item()

            if (step + 1) % args.grad_accum_steps == 0:
                torch.nn.utils.clip_grad_norm_(
                    [p for p in model.parameters() if p.requires_grad], 1.0
                )
                optimizer.step()
                scheduler.step()
                optimizer.zero_grad()
                global_step += 1

                if global_step % 20 == 0:
                    avg_loss = running_loss / (step + 1)
                    print(f"epoch {epoch} step {global_step} avg_loss={avg_loss:.4f}")

        print(f"Epoch {epoch} finished. Running validation...")
        val_metrics, _ = run_retrieval_eval(
            model, processor, val_queries, val_qrels, doc_index, device,
            image_batch_size=args.eval_image_batch_size,
            query_batch_size=args.eval_query_batch_size,
            top_k=args.top_k_eval,
            full_corpus_ids=full_corpus_ids,
        )
        print(f"  Validation metrics (epoch {epoch}): {val_metrics}")

        score = val_metrics["ndcg@10"]
        if score > best_score:
            best_score = score
            print(f"  New best val nDCG@10 = {best_score:.4f} -- saving adapter to {best_adapter_dir}")
            model.save_pretrained(best_adapter_dir)
            processor.save_pretrained(best_adapter_dir)

        # also keep the latest epoch's adapter, useful for resuming/inspection
        latest_dir = os.path.join(args.output_dir, f"epoch_{epoch}_adapter")
        model.save_pretrained(latest_dir)

    print(f"Training complete. Best val nDCG@10 = {best_score:.4f}")
    print(f"Best adapter saved at: {best_adapter_dir}")


if __name__ == "__main__":
    main()
