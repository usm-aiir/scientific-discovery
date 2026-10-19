"""
QLoRA fine-tuning for ColQwen2.5 (vidore/colqwen2.5-v0.2).

Trains on (query, positive-figure) pairs built from train.json + Train_figure_qrels.tsv,
using in-batch negatives (bidirectional ColBERT-style contrastive loss).
Evaluates on val.json + Val_figure_qrels.tsv BEFORE training (step-0 baseline),
after every epoch, and optionally every N optimizer steps, keeping the LoRA
adapter with the best validation nDCG@10.

The base model is loaded in 4-bit (QLoRA); only LoRA adapters on the
language-model's attention/MLP projections are trained. The vision encoder and
base weights stay frozen and quantized.

Changes vs. previous version:
  - step-0 validation (before any training) so you can tell "loading is broken"
    apart from "training made it worse"
  - adapter diagnostics: prints which PEFT adapters are attached/active
  - LoRA target regex now excludes the vision encoder (model.visual.*), and the
    script prints which modules actually got LoRA
  - in-batch false negatives masked (another item in the batch that is also
    relevant to the query is no longer treated as a negative)
  - training pairs whose image fails to decode are dropped up front instead of
    being trained as blank white placeholders
  - pixel_values cast to bf16 in validation too (matches training)
  - lower default LR (5e-5), optional --eval_every_steps
  - qrels/corpus doc_id coverage check for the validation set
  - FIX: keeps training the pretrained ColQwen adapter. The old version added a
    new adapter with the same name ('default'), which silently replaced the
    pretrained retrieval adapter, so every run started from the untrained base
  - FIX: gradient checkpointing now actually turns on (it was silently skipped)
  - --no_quantize: train the LoRA adapter on top of the full bf16 model
    (plain LoRA instead of QLoRA), to rule out 4-bit quantization as a factor
  - --val_distractors K: add K fixed random corpus figures to the validation
    candidate pool, so validation is harder and predicts full-corpus test better
  - --overfit_n N: sanity check -- train on N training queries only and
    validate on those SAME queries; a working pipeline should memorize them
  - optional same-paper hard negatives (--hard_negatives K): each query also
    competes against K other figures from its own paper, which is the hard
    part of this task (picking the right figure/panel within one paper)

Usage:
  python train_colqwen_qlora.py \
      --image_dir /mnt/netstore1_home/adah.holt/figures/images \
      --train_json train.json --train_qrels Train_figure_qrels.tsv \
      --val_json   val.json   --val_qrels   Val_figure_qrels.tsv \
      --output_dir ./colqwen_qlora_run

  Add same-paper hard negatives (2 per query here; each step then encodes
  train_batch_size * (1 + K) images, so watch GPU memory):
      ... --hard_negatives 2

Requires: transformers, torch, peft, bitsandbytes, accelerate, pillow, colpali_engine
  pip install -U peft bitsandbytes accelerate colpali_engine
"""

import argparse
import os
import random

import torch
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader

from transformers import (
    BitsAndBytesConfig,
    get_cosine_schedule_with_warmup,
)
from colpali_engine.models import ColQwen2_5, ColQwen2_5_Processor
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training

from eval_utils import (
    find_images,
    build_doc_index,
    canonical_doc_items,
    load_queries,
    load_qrels,
    evaluate_run,
    safe_open_image,
    report_qrels_coverage,
)

# Fallback size used whenever an image fails to decode at eval time and gets
# replaced by a blank placeholder. (Training pairs with undecodable images are
# dropped up front instead -- see filter_unreadable_pairs.)
PLACEHOLDER_IMAGE_SIZE = (448, 448)

# LoRA on the language model's projection layers only. The negative lookahead
# (?!.*visual) excludes the vision encoder: in recent transformers versions the
# vision blocks live under model.visual.* and their MLPs ALSO use
# gate_proj/up_proj/down_proj, so the old regex silently put LoRA on them.
# custom_text_proj (the ColBERT projection head) is left frozen.
LORA_TARGET_REGEX = (
    r"(.*(model)(?!.*visual).*\.(q_proj|k_proj|v_proj|o_proj|gate_proj|up_proj|down_proj)$)"
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
              f"base_model={getattr(c, 'base_model_name_or_path', None)}, "
              f"targets={getattr(c, 'target_modules', None)}")
    try:
        active = getattr(model, "active_adapters", None)
        active = active() if callable(active) else active
    except Exception as e:  # noqa: BLE001
        active = f"(could not query: {e})"
    print(f"  active adapters: {active}")


def report_lora_targets(model):
    """Print how many modules carry LoRA weights and whether any are in the vision encoder."""
    names = [n[: -len(".lora_A")] for n, _ in model.named_modules() if n.endswith(".lora_A")]
    vision = [n for n in names if "visual" in n]
    print(f"[LORA] {len(names)} modules carry LoRA weights ({len(vision)} inside the vision encoder).")
    for n in names[:5]:
        print(f"    e.g. {n}")
    if vision:
        print(f"  [WARN] LoRA found in the vision encoder, e.g. {vision[:3]}. "
              f"If these come from the NEW adapter, the target regex is too broad.")


def filter_unreadable_pairs(dataset, doc_index):
    """Drop training pairs whose image cannot be decoded (instead of training on blank placeholders)."""
    docids = sorted({d for _, d in dataset.pairs})
    print(f"Checking {len(docids)} training images can be decoded...")
    bad = {d for d in docids if safe_open_image(doc_index[d]) is None}
    if bad:
        before = len(dataset.pairs)
        dataset.pairs = [(q, d) for q, d in dataset.pairs if d not in bad]
        print(f"  [WARN] {len(bad)} images failed to decode; dropped "
              f"{before - len(dataset.pairs)} training pairs.")
    else:
        print("  All training images decode fine.")


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


def paper_id_of(doc_id):
    """'2504.15247_10' / '2504.15247_3_e' -> '2504.15247' (qrels use '<paper>_<fig>[_<panel>]')."""
    return doc_id.split("_")[0]


class HardNegativeSampler:
    """
    Picks negatives for a query from the SAME paper as its positive figure,
    skipping anything the qrels mark as relevant to that query. If the paper
    has too few other figures, the rest are filled with random figures from
    other papers, so every batch has the same shape.
    """

    def __init__(self, canonical_items, qrels, k, seed=0):
        self.k = k
        self.qrels = qrels
        self.rng = random.Random(seed)
        # Index by filename stem: that is the convention the qrels use, and
        # build_doc_index() maps every stem to its file.
        self.all_ids = sorted({p.stem for _, p in canonical_items})
        self.by_paper = {}
        for d in self.all_ids:
            self.by_paper.setdefault(paper_id_of(d), []).append(d)

    def report(self, pairs):
        """Print how many same-paper negatives are actually available."""
        avail = [
            sum(1 for d in self.by_paper.get(paper_id_of(pos), [])
                if self.qrels.get(qid, {}).get(d, 0) <= 0)
            for qid, pos in pairs
        ]
        enough = sum(1 for a in avail if a >= self.k)
        print(f"[HARD NEG] k={self.k}: {enough}/{len(pairs)} pairs have at least {self.k} "
              f"same-paper negatives (mean available {sum(avail) / max(1, len(avail)):.1f}); "
              f"the rest are topped up with random figures from other papers.")

    def _rng(self):
        # DataLoader workers each get a copy of this object; reseed per worker
        # (info.seed differs per worker and per epoch) so they don't all draw
        # the same sequence of negatives.
        info = torch.utils.data.get_worker_info()
        if info is not None and getattr(self, "_worker_seed", None) != info.seed:
            self._worker_seed = info.seed
            self.rng = random.Random(info.seed)
        return self.rng

    def sample(self, qid, pos_docid):
        rng = self._rng()
        relevant = self.qrels.get(qid, {})
        same = [d for d in self.by_paper.get(paper_id_of(pos_docid), [])
                if relevant.get(d, 0) <= 0]
        negs = rng.sample(same, min(self.k, len(same)))
        while len(negs) < self.k:
            d = rng.choice(self.all_ids)
            if relevant.get(d, 0) <= 0 and d not in negs:
                negs.append(d)
        return negs


class Collator:
    def __init__(self, processor, queries, doc_index, neg_sampler=None):
        self.processor = processor
        self.queries = queries
        self.doc_index = doc_index
        self.neg_sampler = neg_sampler

    def __call__(self, batch):
        qids, docids = zip(*batch)
        texts = [self.queries[q] for q in qids]
        # Image list = the B positives (column i belongs to query i), then any
        # hard negatives. Every image is a candidate for every query.
        all_docids = list(docids)
        if self.neg_sampler is not None:
            for q, d in zip(qids, docids):
                all_docids.extend(self.neg_sampler.sample(q, d))
        images = [safe_open_image(self.doc_index[d], fallback_size=PLACEHOLDER_IMAGE_SIZE)
                  for d in all_docids]
        # ColQwen2.5 uses separate process_images / process_queries methods
        image_inputs = self.processor.process_images(images)
        text_inputs = self.processor.process_queries(list(texts))
        return image_inputs, text_inputs, qids, tuple(all_docids)


# ---------------------------------------------------------------------------
# Loss
# ---------------------------------------------------------------------------

def colbert_inbatch_loss(query_embeddings, doc_embeddings, processor, qids, docids, qrels,
                         temperature=1.0):
    """
    Bidirectional in-batch contrastive loss (standard ColBERT/ColQwen training loss),
    with false negatives masked: if doc j is also relevant to query i (same figure
    sampled twice, the same query with several positives, or any other qrels hit),
    score[i, j] is excluded from the softmax instead of being pushed down.

    docids lists the B positives first (column i = positive of query i), then
    any hard negatives, so scores has shape (B, B + B*K). Query->doc uses every
    column; doc->query uses only the B positive columns, since a hard negative
    has no query of its own to be matched to.
    """
    scores = processor.score_multi_vector(query_embeddings, doc_embeddings)  # (B, N)
    scores = scores / temperature
    B, N = scores.shape
    false_neg = torch.tensor(
        [[i != j and qrels.get(qids[i], {}).get(docids[j], 0) > 0 for j in range(N)]
         for i in range(B)],
        dtype=torch.bool, device=scores.device,
    )
    scores = scores.masked_fill(false_neg, float("-inf"))
    labels = torch.arange(B, device=scores.device)
    loss_q2d = F.cross_entropy(scores, labels)
    loss_d2q = F.cross_entropy(scores[:, :B].t(), labels)
    return (loss_q2d + loss_d2q) / 2


# ---------------------------------------------------------------------------
# Validation
# ---------------------------------------------------------------------------

def _cast_pixels(inputs):
    """Match the training loop: pixel_values in bf16."""
    if "pixel_values" in inputs:
        inputs["pixel_values"] = inputs["pixel_values"].to(torch.bfloat16)
    return inputs


@torch.no_grad()
def run_retrieval_eval(model, processor, queries, qrels, doc_index, device,
                       image_batch_size, query_batch_size, top_k, full_corpus_ids=None):
    """
    Encodes queries + a candidate doc pool, scores them, builds a run dict,
    and returns nDCG@10 / MRR / etc. against qrels.

    candidate pool = full_corpus_ids if given, else the union of doc_ids that
    appear in the qrels file -- this keeps per-epoch validation fast but is a
    much EASIER task than test-time retrieval; pass --eval_full_corpus for a
    true full-corpus evaluation (slow, matches the deployment setting).
    """
    was_training = model.training
    model.eval()

    qids = [q for q in qrels.keys() if q in queries]
    texts = [queries[q] for q in qids]
    q_embs = []
    for i in range(0, len(texts), query_batch_size):
        inputs = processor.process_queries(texts[i : i + query_batch_size]).to(device)
        q_embs.append(model(**inputs).cpu())
    max_len = max(e.shape[1] for e in q_embs)
    dim = q_embs[0].shape[2]
    q_embs = [
        torch.cat([e, torch.zeros(e.shape[0], max_len - e.shape[1], dim, dtype=e.dtype)], dim=1)
        if e.shape[1] < max_len else e
        for e in q_embs
    ]
    query_embeddings = torch.cat(q_embs, dim=0).to(device)

    if full_corpus_ids is not None:
        candidate_ids = full_corpus_ids
    else:
        candidate_ids = sorted({d for docs in qrels.values() for d in docs.keys() if d in doc_index})

    run_scores = {q: [] for q in qids}
    for i in range(0, len(candidate_ids), image_batch_size):
        batch_ids = candidate_ids[i : i + image_batch_size]
        images = [safe_open_image(doc_index[d], fallback_size=PLACEHOLDER_IMAGE_SIZE) for d in batch_ids]
        inputs = _cast_pixels(processor.process_images(images).to(device))
        doc_embeddings = model(**inputs)
        scores = processor.score_multi_vector(query_embeddings, doc_embeddings).detach().cpu()
        for qi, qid in enumerate(qids):
            for di, docid in enumerate(batch_ids):
                run_scores[qid].append((scores[qi, di].item(), docid))
        del inputs, doc_embeddings, scores
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    run = {q: [d for _, d in sorted(s, key=lambda x: x[0], reverse=True)[:top_k]] for q, s in run_scores.items()}
    metrics = evaluate_run(run, qrels, k_list=(10,))
    if was_training:
        model.train()
    return metrics, run_scores


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="QLoRA fine-tuning for ColQwen2.5")
    parser.add_argument(
        "--image_dir",
        default="/mnt/netstore1_home/adah.holt/figures/images",
    )
    parser.add_argument("--train_json",  required=True)
    parser.add_argument("--train_qrels", required=True)
    parser.add_argument("--val_json",    required=True)
    parser.add_argument("--val_qrels",   required=True)
    parser.add_argument("--output_dir",  default="./colqwen_qlora_run")
    parser.add_argument("--model_name",  default="vidore/colqwen2.5-v0.2")

    parser.add_argument("--epochs",              type=int,   default=3)
    parser.add_argument("--train_batch_size",    type=int,   default=2,
                        help="Per-step batch size. This is ALSO the number of in-batch "
                             "candidates per query (batch_size - 1 negatives); gradient "
                             "accumulation does not add negatives. Raise it as far as memory allows.")
    parser.add_argument("--grad_accum_steps",    type=int,   default=8)
    parser.add_argument("--hard_negatives",      type=int,   default=0,
                        help="Extra negatives per query drawn from the SAME paper as its positive "
                             "figure (0 = off, plain in-batch negatives). Each step encodes "
                             "train_batch_size * (1 + K) images, so lower the batch size if you OOM.")
    parser.add_argument("--eval_image_batch_size", type=int, default=4)
    parser.add_argument("--eval_query_batch_size", type=int, default=16)
    parser.add_argument("--lr",                  type=float, default=5e-5,
                        help="Was 2e-4. The starting model is already a tuned retriever, "
                             "so a gentler LR is less likely to wreck it.")
    parser.add_argument("--warmup_ratio",        type=float, default=0.05)
    parser.add_argument("--temperature",         type=float, default=1.0)
    parser.add_argument("--top_k_eval",          type=int,   default=10)
    parser.add_argument("--eval_every_steps",    type=int,   default=0,
                        help="Also validate every N optimizer steps (0 = only after each epoch).")
    parser.add_argument("--skip_step0_eval", action="store_true",
                        help="Skip the validation pass before training.")
    parser.add_argument("--skip_image_check", action="store_true",
                        help="Skip the up-front decode check of training images.")
    parser.add_argument("--eval_full_corpus", action="store_true",
                        help="Evaluate validation nDCG@10 against the FULL image corpus "
                             "instead of just the docs referenced in the val qrels file. "
                             "Much slower but matches deployment conditions.")

    parser.add_argument("--val_distractors", type=int, default=0,
                        help="Add this many fixed, randomly chosen corpus figures to the validation "
                             "candidate pool (default 0 = only figures in the val qrels). E.g. 3000 makes "
                             "validation much closer to full-corpus test retrieval.")
    parser.add_argument("--overfit_n", type=int, default=0,
                        help="SANITY CHECK: train on only N training queries and validate on those same "
                             "queries. If the pipeline works, validation should climb towards 1.0.")
    parser.add_argument("--no_quantize", action="store_true",
                        help="Load the base model in bf16 instead of 4-bit (plain LoRA, not QLoRA). "
                             "Uses more GPU memory (fits on a 48 GB A6000).")

    parser.add_argument("--new_adapter", action="store_true",
                        help="Old behaviour: add a fresh LoRA adapter instead of continuing the "
                             "pretrained one. WARNING: this discards the pretrained retrieval "
                             "adapter. Only for reproducing earlier runs.")
    parser.add_argument("--lora_r",       type=int,   default=16,
                        help="Only used with --new_adapter.")
    parser.add_argument("--lora_alpha",   type=int,   default=32)
    parser.add_argument("--lora_dropout", type=float, default=0.05)
    parser.add_argument("--gradient_checkpointing", action="store_true", default=False,
                        help="Enable gradient checkpointing on the inner language model to reduce "
                             "activation memory, so you can afford a larger --train_batch_size "
                             "(= more in-batch negatives). Verify a training step runs before trusting it.")

    parser.add_argument("--device",      default=None,
                        help="e.g. 'cuda:0'. Default: cuda:0 if available.")
    parser.add_argument("--seed",        type=int, default=42)
    parser.add_argument("--num_workers", type=int, default=4)
    args = parser.parse_args()

    random.seed(args.seed)
    torch.manual_seed(args.seed)

    device = torch.device(args.device) if args.device else torch.device(
        "cuda:0" if torch.cuda.is_available() else "cpu"
    )
    if device.type != "cuda":
        raise SystemExit("Training requires a CUDA GPU. No GPU detected / selected.")

    if args.train_batch_size < 8 and args.hard_negatives == 0:
        print(f"[NOTE] train_batch_size={args.train_batch_size} gives each query only "
              f"{args.train_batch_size - 1} in-batch negative(s). This is a weak contrastive "
              f"signal; try --gradient_checkpointing and a larger batch.")

    os.makedirs(args.output_dir, exist_ok=True)
    best_adapter_dir = os.path.join(args.output_dir, "best_adapter")

    print("Indexing image corpus...")
    image_paths = find_images(args.image_dir)
    doc_index = build_doc_index(args.image_dir, image_paths)
    canonical_items = canonical_doc_items(args.image_dir, image_paths)
    print(f"  {len(canonical_items)} unique images indexed ({len(doc_index)} lookup keys incl. aliases).")

    print("Loading data...")
    train_queries = load_queries(args.train_json)
    train_qrels   = load_qrels(args.train_qrels)
    val_queries   = load_queries(args.val_json)
    val_qrels     = load_qrels(args.val_qrels)

    if args.overfit_n:
        # Sanity check: keep N training queries that have at least one image on
        # disk, train only on them, and "validate" on exactly those queries.
        usable = sorted(q for q, docs in train_qrels.items()
                        if q in train_queries and any(r > 0 and d in doc_index for d, r in docs.items()))
        chosen = sorted(random.Random(args.seed).sample(usable, min(args.overfit_n, len(usable))))
        train_qrels = {q: train_qrels[q] for q in chosen}
        val_queries, val_qrels = train_queries, train_qrels
        print(f"[OVERFIT TEST] Training AND validating on the same {len(chosen)} training queries. "
              f"If training works, validation nDCG@10 should rise towards 1.0.")

    if args.eval_full_corpus:
        candidate_ids = [doc_id for doc_id, _ in canonical_items]
    else:
        candidate_ids = sorted({d for docs in val_qrels.values() for d in docs if d in doc_index})
        if args.val_distractors:
            pool = [doc_id for doc_id, _ in canonical_items if doc_id not in set(candidate_ids)]
            extra = random.Random(args.seed + 1).sample(pool, min(args.val_distractors, len(pool)))
            candidate_ids = sorted(set(candidate_ids) | set(extra))
    print(f"Validation candidate pool: {len(candidate_ids)} figures"
          + (" (full corpus)" if args.eval_full_corpus else
             f" ({args.val_distractors} random distractors added)" if args.val_distractors else
             " (val-qrels figures only -- an easy pool; consider --val_distractors)"))
    report_qrels_coverage(val_qrels, doc_index.keys(), label="val qrels vs. doc index")

    train_dataset = QueryFigurePairDataset(train_queries, train_qrels, doc_index)
    if not args.skip_image_check:
        filter_unreadable_pairs(train_dataset, doc_index)
    if len(train_dataset) < args.train_batch_size:
        raise SystemExit("Not enough training pairs to form a single batch.")

    if args.no_quantize:
        print("Loading base model in bf16 (plain LoRA, NO 4-bit quantization)...")
        quant_kwargs = {}
    else:
        print("Loading base model in 4-bit (QLoRA)...")
        quant_kwargs = {"quantization_config": BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16,
        bnb_4bit_use_double_quant=True,
        # Keep the ColBERT projection head in full precision; if it is 4-bit
        # quantized its weight.dtype becomes uint8 and the projection crashes.
        llm_int8_skip_modules=["custom_text_proj"],
        )}
    # Pin to a single device for training; device_map="auto" (multi-GPU split)
    # can cause gradient/dispatch issues with PEFT + QLoRA.
    model = ColQwen2_5.from_pretrained(
        args.model_name,
        torch_dtype=torch.bfloat16,
        device_map={"": device.index if device.index is not None else 0},
        **quant_kwargs,
    )
    processor = ColQwen2_5_Processor.from_pretrained(args.model_name)
    describe_adapters(model, "after loading pretrained model")

    # vidore/colqwen2.5-v0.2 is itself a LoRA adapter (named "default") on top of
    # vidore/colqwen2.5-base, and transformers attaches it automatically on load.
    # The previous version of this script then called get_peft_model(), which
    # created a NEW adapter also named "default" -- re-initialising those LoRA
    # weights and silently throwing away the pretrained retrieval adapter
    # (including its LoRA on the custom_text_proj head). Training therefore
    # started from the untrained base model. Now we keep training the
    # pretrained adapter itself instead.
    has_pretrained_adapter = bool(getattr(model, "peft_config", None))

    # prepare_model_for_kbit_training freezes every parameter (and upcasts the
    # small non-quantized ones to fp32); we re-enable the LoRA weights below.
    # It is only meant for quantized models, so skip it with --no_quantize.
    if not args.no_quantize:
        model = prepare_model_for_kbit_training(model, use_gradient_checkpointing=False)

    if args.gradient_checkpointing:
        # Needed so frozen/quantized layers still pass gradients through.
        if hasattr(model, "enable_input_require_grads"):
            model.enable_input_require_grads()
        # In current colpali_engine, ColQwen2_5 IS the Qwen2.5-VL model (there
        # is no inner model.model), so enable it on whichever exists.
        inner = getattr(model, "model", None)
        target = inner if (inner is not None and hasattr(inner, "gradient_checkpointing_enable")) else model
        try:
            # use_reentrant=True: non-reentrant checkpointing has a known bad
            # interaction with bitsandbytes 4-bit layers.
            target.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": True})
            model.config.use_cache = False
            print(f"Gradient checkpointing enabled on {type(target).__name__}, use_reentrant=True.")
        except Exception as e:  # noqa: BLE001
            print(f"[WARN] Could not enable gradient checkpointing ({e}); "
                  f"proceeding without it (higher activation memory use).")
    else:
        print("Gradient checkpointing disabled. Reduce --train_batch_size / "
              "--eval_image_batch_size if you hit OOM.")

    if has_pretrained_adapter and not args.new_adapter:
        print("Continuing training of the PRETRAINED ColQwen adapter (not adding a new one).")
        for name, param in model.named_parameters():
            param.requires_grad = "lora_" in name
            # Keep trainable weights in fp32: tiny Adam updates on bf16 weights
            # get rounded away (bf16 has only ~3 significant digits). The
            # quantized path already does this inside prepare_model_for_kbit_training.
            if param.requires_grad and param.dtype != torch.float32:
                param.data = param.data.float()
    else:
        if has_pretrained_adapter:
            print("[WARN] --new_adapter: adding a fresh LoRA adapter. This REPLACES the "
                  "pretrained retrieval adapter (same name 'default'); training starts "
                  "from the untrained base model. Only use this to reproduce old runs.")
        lora_config = LoraConfig(
            r=args.lora_r,
            lora_alpha=args.lora_alpha,
            lora_dropout=args.lora_dropout,
            bias="none",
            target_modules=LORA_TARGET_REGEX,
            # task_type intentionally omitted: yields a generic PeftModel whose
            # forward() passes through to the base model, preserving the tensor
            # output that ColQwen2_5 returns directly.
        )
        model = get_peft_model(model, lora_config)

    n_train = sum(p.numel() for p in model.parameters() if p.requires_grad)
    n_total = sum(p.numel() for p in model.parameters())
    print(f"Trainable parameters: {n_train:,} / {n_total:,} ({100 * n_train / max(1, n_total):.3f}%)")
    if n_train == 0:
        raise SystemExit("[ERROR] No trainable parameters -- no LoRA weights were found to train.")
    describe_adapters(model, "model being trained")
    report_lora_targets(model)

    neg_sampler = None
    if args.hard_negatives > 0:
        neg_sampler = HardNegativeSampler(canonical_items, train_qrels, args.hard_negatives,
                                          seed=args.seed)
        neg_sampler.report(train_dataset.pairs)
        print(f"  Each step encodes {args.train_batch_size * (1 + args.hard_negatives)} images "
              f"({args.train_batch_size} positives + "
              f"{args.train_batch_size * args.hard_negatives} hard negatives).")

    collator = Collator(processor, train_queries, doc_index, neg_sampler=neg_sampler)
    train_loader = DataLoader(
        train_dataset,
        batch_size=args.train_batch_size,
        shuffle=True,
        collate_fn=collator,
        num_workers=args.num_workers,
        drop_last=True,   # keep batches full-size for stable in-batch negatives
    )

    optimizer = torch.optim.AdamW(
        [p for p in model.parameters() if p.requires_grad], lr=args.lr
    )
    steps_per_epoch = len(train_loader) // args.grad_accum_steps
    total_steps     = max(1, steps_per_epoch * args.epochs)
    scheduler = get_cosine_schedule_with_warmup(
        optimizer,
        num_warmup_steps=int(total_steps * args.warmup_ratio),
        num_training_steps=total_steps,
    )

    state = {"best_score": -1.0, "best_tag": None}

    def validate(tag):
        print(f"Running validation ({tag})...")
        metrics, _ = run_retrieval_eval(
            model, processor, val_queries, val_qrels, doc_index, device,
            image_batch_size=args.eval_image_batch_size,
            query_batch_size=args.eval_query_batch_size,
            top_k=args.top_k_eval,
            full_corpus_ids=candidate_ids,
        )
        print(f"  Validation metrics ({tag}): {metrics}")
        return metrics

    def validate_and_maybe_save(tag):
        score = validate(tag)["ndcg@10"]
        if score > state["best_score"]:
            state["best_score"], state["best_tag"] = score, tag
            print(f"  New best val nDCG@10 = {score:.4f} ({tag}) -- saving adapter to {best_adapter_dir}")
            model.save_pretrained(best_adapter_dir)
            processor.save_pretrained(best_adapter_dir)
        return score

    # Step-0 baseline: the new LoRA adapter is initialised to a no-op (lora_B = 0),
    # so this should match the pretrained model evaluated in 4-bit. If it is
    # already much worse than your pretrained baseline, the problem is loading /
    # quantization, not training.
    step0_score = None
    if not args.skip_step0_eval:
        step0_score = validate("step 0, before training")["ndcg@10"]

    global_step = 0
    for epoch in range(args.epochs):
        model.train()
        running_loss = 0.0
        optimizer.zero_grad()

        for step, (image_inputs, text_inputs, qids, docids) in enumerate(train_loader):
            image_inputs = _cast_pixels({k: v.to(device) for k, v in image_inputs.items()})
            text_inputs = {k: v.to(device) for k, v in text_inputs.items()}

            # ColQwen2_5 returns the embedding tensor directly (no .embeddings field)
            doc_embeddings   = model(**image_inputs)
            query_embeddings = model(**text_inputs)

            loss = colbert_inbatch_loss(
                query_embeddings, doc_embeddings, processor, qids, docids, train_qrels,
                args.temperature,
            )
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
                    print(f"epoch {epoch} step {global_step} avg_loss={avg_loss:.4f} "
                          f"lr={scheduler.get_last_lr()[0]:.2e}")

                if args.eval_every_steps and global_step % args.eval_every_steps == 0:
                    validate_and_maybe_save(f"epoch {epoch}, step {global_step}")

        print(f"Epoch {epoch} finished.")
        validate_and_maybe_save(f"end of epoch {epoch}")

        # Also keep per-epoch snapshot, useful for resuming / inspection
        latest_dir = os.path.join(args.output_dir, f"epoch_{epoch}_adapter")
        model.save_pretrained(latest_dir)

    print(f"Training complete. Best val nDCG@10 = {state['best_score']:.4f} ({state['best_tag']})")
    if step0_score is not None:
        print(f"Step-0 (untrained) val nDCG@10 = {step0_score:.4f}")
        if state["best_score"] <= step0_score:
            print("  [WARN] Training never beat the untrained starting point on validation.")
    print(f"Best adapter saved at: {best_adapter_dir}")


if __name__ == "__main__":
    main()