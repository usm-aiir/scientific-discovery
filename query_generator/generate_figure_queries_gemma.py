#!/usr/bin/env python3
"""
generate_figure_queries.py
==========================
Two-agent LLM pipeline for generating natural-language search queries
grounded in arXiv figures.

Pipeline
--------
1. Load captions, metadata, and reference TSVs and join them into one DataFrame.
2. Resolve each figure's image path on disk
   (figures/{yy}/{mm}/{paper_id}/{fig}{sub}.png) and drop rows whose image
   file is missing.
3. Build a diverse candidate pool:
     - at most one figure per paper
     - roughly balanced across arXiv categories (round-robin)
     - mixed across caption-length buckets (short / medium / long)
4. For each candidate figure, run a two-agent loop with a local Gemma model:
     Agent 1 ("Author")   -- drafts a scientist's information need (a query)
                             that is answerable by looking at the figure.
     Agent 2 ("Reviewer") -- checks the query; either ACCEPTs it or REJECTs
                             it with concrete feedback for Agent 1 to revise.
   Up to MAX_ROUNDS attempts per figure. Figures that are still rejected after
   all rounds are skipped and the next candidate is tried.
5. Stop once TARGET accepted queries have been collected (or the candidate
   pool runs out).
6. Write paper_id, figure_id, sub_id, and query to a TSV incrementally so
   the script is safely resumable if interrupted.

Note on image usage
-------------------
Agent 1 (the Author) receives the actual figure image alongside the
caption, abstract, and in-text references so that its queries are
grounded in what the figure *shows*, not just its textual description.
Agent 2 (the Reviewer) operates text-only — it has enough context from
the caption and abstract to judge whether a query is answerable from
the figure without needing the pixels itself.

Output TSV columns
------------------
paper_id, figure_id, sub_id, query

Usage
-----
    python generate_figure_queries.py \\
        --captions_tsv arxiv_data/captions/24_10.tsv \\
        --metadata_tsv arxiv_data/metadata_24_10.tsv \\
        --ref_tsv      arxiv_data/references/ref_24_10.tsv \\
        --figures_dir  arxiv_data/figures \\
        --output_tsv   queries_200.tsv

Example
-------
    python generate_figure_queries.py \\
        --captions_tsv arxiv_data/captions/24_10.tsv \\
        --metadata_tsv arxiv_data/metadata_24_10.tsv \\
        --ref_tsv      arxiv_data/references/ref_24_10.tsv \\
        --figures_dir  arxiv_data/figures \\
        --output_tsv   queries_200.tsv \\
        --target 200 --model_name google/gemma-4-31B-it
"""

from __future__ import annotations  # FIX #8: enables str | None etc. on Python 3.9

import argparse
import json
import logging
import os
import random
import re
import sys
from collections import defaultdict, deque

import pandas as pd

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-7s | %(message)s",
)
log = logging.getLogger("figure_query_gen")


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

DEFAULT_TARGET          = 200
DEFAULT_MAX_ROUNDS      = 3
DEFAULT_POOL_MULTIPLIER = 4
DEFAULT_SEED            = 13
DEFAULT_MODEL           = "google/gemma-4-31B-it"

# GemmaChat generation settings
MAX_NEW_TOKENS     = 150
AGENT1_TEMPERATURE = 0.8   # higher for creative query drafting
AGENT2_TEMPERATURE = 0.0   # greedy — Agent 2 is binary accept/reject, no benefit from sampling

# Figure context construction
ABSTRACT_MAX_CHARS = 600   # truncate abstracts beyond this to save tokens

# Caption-length bucket thresholds (in words)
CAPTION_SHORT_MAX  = 20
CAPTION_MEDIUM_MAX = 60

# Agent system prompts
AGENT1_SYSTEM = """You are simulating a working scientist who has just come \
across a figure in a paper. You are shown the actual figure image alongside \
its caption, sub-caption, abstract, and how other parts of the text cite it. \
Your job is to produce ONE realistic information need this scientist would \
have -- a natural-language query that can be answered by LOOKING AT THE \
FIGURE ITSELF (not by reading the rest of the paper).

Important constraints:
- Do NOT refer to the figure by its number or say "in the figure" or \
"Figure X". The query must be phrased as a standalone question that a \
researcher might ask, e.g. "How does the predicted interest rate change with \
increasing term length for each mechanism?"
- The query must be answerable using only the figure's visual content, given \
the context provided.
- It must be specific to this figure's content, not generic.
- It should be a natural, one to two sentence question, as if asked in a \
conversation or search.

Good examples:
- "What is the trend of training loss for the proposed method compared to the baseline over epochs?"
- "Which of the three mechanisms produces the highest output under low capital levels?"

Respond with ONLY a JSON object with a single key "query" whose value is your \
question. Do not include any other text, explanation, or formatting outside \
the JSON object.

Example of the required format:
{"query": "How does accuracy change as the number of training samples increases for each model?"}"""

AGENT2_SYSTEM = """You are a careful peer reviewer checking whether a proposed \
query is a good fit for a figure-grounded question-answering benchmark. ACCEPT \
the query only if ALL of the following hold:
  1. It can plausibly be answered by looking at the figure alone (given the \
caption/context you were shown), without needing the rest of the paper.
  2. It is specific to this particular figure (not a generic template question).
  3. It reads like something a real scientist would ask, and is grammatical, \
one to two sentences.
  4. It does not already state the answer/finding within the query itself.
  5. It is not simply restating the caption verbatim.
  6. It does NOT mention the figure number or say "in the figure" or "Figure X" \
-- the query must be a standalone, figure-agnostic question.

If any criterion fails, REJECT and give concise, actionable feedback describing \
exactly what to change (e.g. "too generic, ask about the specific trend for \
method X" or "the query gives away the answer, rephrase as a question instead \
of a statement" or "remove reference to Figure 3").

Respond with ONLY a JSON object:
{"verdict": "ACCEPT" or "REJECT", "feedback": "<empty string if ACCEPT, else \
the fix needed>"}"""


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------

def load_data(captions_tsv: str, metadata_tsv: str, ref_tsv: str) -> pd.DataFrame:
    """
    Load captions, metadata, and reference TSVs, join them into one DataFrame,
    and return rows that have a non-empty caption.

    Multiple reference paragraphs for the same (paper_id, figure_id) pair are
    collapsed into a single string separated by " || ".
    """
    df_cap  = pd.read_csv(captions_tsv, sep="\t", dtype=str, keep_default_na=False)
    df_meta = pd.read_csv(metadata_tsv,  sep="\t", dtype=str, keep_default_na=False)
    df_ref  = pd.read_csv(ref_tsv,       sep="\t", dtype=str, keep_default_na=False)

    # Ensure expected columns exist (fill missing ones with empty strings).
    for col in ["paper_id", "figure_id", "sub_id", "caption", "sub_caption"]:
        if col not in df_cap.columns:
            df_cap[col] = ""
    for col in ["url", "paper_id", "title", "abstract", "categories"]:
        if col not in df_meta.columns:
            df_meta[col] = ""
    for col in ["paper_id", "figure_id", "reference_text"]:
        if col not in df_ref.columns:
            df_ref[col] = ""

    df_cap  = df_cap.fillna("")
    df_meta = df_meta.fillna("")
    df_ref  = df_ref.fillna("")

    # Collapse multiple reference rows per (paper_id, figure_id) into one string.
    df_ref_grouped = (
        df_ref
        .groupby(["paper_id", "figure_id"])["reference_text"]
        .apply(lambda s: " || ".join(x for x in s if x.strip()))
        .reset_index()
    )

    df = df_cap.merge(df_meta, on="paper_id", how="left", suffixes=("", "_meta"))
    df = df.merge(df_ref_grouped, on=["paper_id", "figure_id"], how="left")
    df["reference_text"] = df["reference_text"].fillna("")

    # Drop rows with no caption -- nothing useful to build a query from.
    df = df[df["caption"].str.strip() != ""].reset_index(drop=True)
    return df


# ---------------------------------------------------------------------------
# Image path resolution
# ---------------------------------------------------------------------------

def resolve_image_path(figures_dir: str, paper_id: str,
                        figure_id: str, sub_id: str) -> str | None:
    """
    Return the path to the figure image on disk, or None if not found.

    ArXiv paper IDs are formatted as '{yymm}.{number}' (e.g. '2410.24226').
    Images are stored at figures/{yy}/{mm}/{paper_id}/{figure_id}{sub_id}.png.
    """
    if "." not in paper_id or len(paper_id.split(".")[0]) < 4:
        return None
    prefix = paper_id.split(".")[0]
    yy, mm  = prefix[:2], prefix[2:4]

    fname_core = f"{figure_id}{sub_id}" if sub_id else figure_id
    candidates = [f"{fname_core}.png", f"{fname_core}.jpg", f"{fname_core}.jpeg"]

    paper_dir = os.path.join(figures_dir, yy, mm, paper_id)
    for candidate in candidates:
        path = os.path.join(paper_dir, candidate)
        if os.path.isfile(path):
            return path
    return None


def attach_image_paths(df: pd.DataFrame, figures_dir: str) -> pd.DataFrame:
    """
    Add an 'image_path' column to df and drop rows whose image is not on disk.
    Logs how many rows were kept vs. dropped.
    """
    df = df.copy()
    df["image_path"] = df.apply(
        lambda row: resolve_image_path(
            figures_dir, row["paper_id"], row["figure_id"], row["sub_id"]
        ),
        axis=1,
    )
    before = len(df)
    df = df[df["image_path"].notna()].reset_index(drop=True)
    log.info("Resolved images: %d / %d rows have a matching file on disk.", len(df), before)
    return df


# ---------------------------------------------------------------------------
# Diverse candidate selection
# ---------------------------------------------------------------------------

def primary_category(categories_str: str) -> str:
    """Return the first arXiv category listed, or 'unknown' if none."""
    categories_str = (categories_str or "").strip()
    return categories_str.split()[0] if categories_str else "unknown"


def caption_len_bucket(caption: str, sub_caption: str) -> str:
    """Bin a figure's combined caption length into 'short', 'medium', or 'long'."""
    n_words = len(f"{caption} {sub_caption}".split())
    if n_words < CAPTION_SHORT_MAX:
        return "short"
    elif n_words <= CAPTION_MEDIUM_MAX:
        return "medium"
    else:
        return "long"


def build_diverse_candidate_order(df: pd.DataFrame, target: int,
                                   pool_multiplier: int = DEFAULT_POOL_MULTIPLIER,
                                   seed: int = DEFAULT_SEED) -> list[int]:
    """
    Return a list of row indices (into df) in the order they should be
    attempted, satisfying:
      - at most one figure per paper
      - round-robin across arXiv categories (keeps the list balanced)
      - within each category, captions alternate across length buckets

    The returned list is capped at target * pool_multiplier so there are
    enough backup candidates to absorb skips from the review loop without
    processing the entire dataset.
    """
    rng = random.Random(seed)

    df = df.copy()
    df["primary_category"] = df["categories"].apply(primary_category)
    df["len_bucket"] = df.apply(
        lambda r: caption_len_bucket(r["caption"], r["sub_caption"]), axis=1
    )

    # One randomly chosen figure per paper.
    one_per_paper = []
    for _, group in df.groupby("paper_id"):
        idxs = list(group.index)
        rng.shuffle(idxs)
        one_per_paper.append(idxs[0])
    rng.shuffle(one_per_paper)

    picked_df = df.loc[one_per_paper]

    # Within each category, interleave figures across length buckets.
    # FIX #9: replaced iterrows() with .items() on the category Series.
    by_category: dict[str, list] = defaultdict(list)
    for idx, cat in picked_df["primary_category"].items():
        by_category[cat].append(idx)

    for cat, idxs in by_category.items():
        by_len: dict[str, list] = defaultdict(list)
        for idx in idxs:
            by_len[picked_df.loc[idx, "len_bucket"]].append(idx)
        for lb in by_len:
            rng.shuffle(by_len[lb])
        interleaved = []
        queues = [deque(by_len[lb]) for lb in ("short", "medium", "long") if by_len.get(lb)]
        while any(queues):
            for q in queues:
                if q:
                    interleaved.append(q.popleft())
        by_category[cat] = interleaved

    # Round-robin across categories so no single category dominates the front.
    cat_names = list(by_category.keys())
    rng.shuffle(cat_names)
    cat_queues = {c: deque(by_category[c]) for c in cat_names}

    order = []
    while any(cat_queues.values()):
        for c in cat_names:
            if cat_queues[c]:
                order.append(cat_queues[c].popleft())

    # Cap to pool_multiplier × target so we don't iterate the full dataset.
    cap = target * pool_multiplier
    order = order[:cap]

    log.info(
        "Candidate pool built: %d figures across %d categories (capped at %d).",
        len(order), len(cat_names), cap,
    )
    return order


# ---------------------------------------------------------------------------
# Model wrapper
# ---------------------------------------------------------------------------

def _normalize_messages_for_template(messages: list[dict]) -> list[dict]:
    """
    Ensure every message's content is in the typed-dict list format that
    Gemma 4's chat template expects, i.e.:
        {"role": "...", "content": [{"type": "text", "text": "..."}]}
    """
    normalized = []
    for msg in messages:
        content = msg["content"]
        if isinstance(content, str):
            content = [{"type": "text", "text": content}]
        normalized.append({"role": msg["role"], "content": content})
    return normalized


class GemmaChat:
    """
    Thin wrapper around a local HuggingFace vision-language model (Gemma 4).

    Uses AutoProcessor (which bundles the tokenizer and image processor) and
    AutoModelForImageTextToText so that figure images can be passed directly
    to Agent 1 alongside the textual context.
    """

    def __init__(self, model_name: str, load_in_4bit: bool = False,
                 max_new_tokens: int = MAX_NEW_TOKENS):
        import torch
        from transformers import AutoModelForImageTextToText, AutoProcessor

        self.max_new_tokens = max_new_tokens
        log.info("Loading model %s ...", model_name)

        quant_kwargs: dict = {}
        if load_in_4bit:
            from transformers import BitsAndBytesConfig
            quant_kwargs["quantization_config"] = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_compute_dtype=torch.float16,
            )

        self.processor = AutoProcessor.from_pretrained(model_name)
        self.model = AutoModelForImageTextToText.from_pretrained(
            model_name,
            device_map="auto",
            torch_dtype=torch.float16,  # FIX #21: torch_dtype (not dtype) is the correct kwarg;
                                        # 'dtype' was silently ignored, leaving non-quantized params
                                        # (vision encoder, layer norms) in bfloat16, which triggers
                                        # a CUDA device-side assert on Turing GPUs (no bf16 support).
            **quant_kwargs,
        )

        # FIX #21 (continued): After loading, cast any remaining bfloat16 tensors to float16.
        # Some model configs set torch_dtype=bfloat16 internally; this guarantees no bf16 survives.
        _n_bf16_cast = 0
        for _p in self.model.parameters():
            if _p.dtype == torch.bfloat16:
                _p.data = _p.data.to(torch.float16)
                _n_bf16_cast += 1
        for _name, _buf in self.model.named_buffers():
            if _buf.dtype == torch.bfloat16:
                _buf.data = _buf.data.to(torch.float16)
                _n_bf16_cast += 1
        if _n_bf16_cast:
            log.info(
                "FIX #21: cast %d bfloat16 parameter/buffer tensors to float16 "
                "(Turing GPU has no bfloat16 hardware support).", _n_bf16_cast
            )
        else:
            log.info("FIX #21: model loaded cleanly in float16 — no bfloat16 tensors found.")

        # FIX #20: after 4-bit loading, convert vision encoder layers back to
        # float16 in-place.  bitsandbytes quantizes ALL nn.Linear layers by
        # default, including the SigLIP vision encoder.  On Turing GPUs (RTX
        # 2080 Ti, compute 7.5) the quantized vision kernels trigger a CUDA
        # device-side assert on the first image forward pass, killing the GPU
        # context for every subsequent figure.  Dequantizing the vision layers
        # here leaves them in float16 (minimal VRAM cost — encoder is small)
        # while the text model stays 4-bit quantized.
        #
        # This is done AFTER from_pretrained because the transformers version
        # installed on this server doesn't accept modules_to_not_convert as a
        # from_pretrained kwarg.
        if load_in_4bit:
            _VISION_PREFIXES = (
                "vision_tower",          # Gemma 3 / LLaVA
                "vision_model",          # PaliGemma variants
                "image_encoder",         # Mistral-VL
                "multi_modal_projector", # cross-modal connector
                "mm_projector",
                "image_newline",
            )
            try:
                import bitsandbytes as bnb
                # Log every 4-bit module name so we can verify vision prefixes.
                all_4bit = [n for n, m in self.model.named_modules()
                            if isinstance(m, bnb.nn.Linear4bit)]
                log.info("Total 4-bit quantized Linear layers in model: %d", len(all_4bit))
                if all_4bit:
                    log.info("First 10 quantized layer names: %s", all_4bit[:10])

                n_dequant = 0
                for full_name, module in list(self.model.named_modules()):
                    if not isinstance(module, bnb.nn.Linear4bit):
                        continue
                    # Use 'in' not 'startswith' — the model may wrap modules
                    # under an extra 'model.' prefix (e.g. model.vision_tower.xxx)
                    if not any(p in full_name for p in _VISION_PREFIXES):
                        continue
                    # Dequantize the 4-bit weight back to float16.
                    try:
                        w16 = module.weight.dequantize().to(torch.float16)
                    except Exception:
                        try:
                            w16 = bnb.functional.dequantize_4bit(
                                module.weight.data, module.weight.quant_state
                            ).to(torch.float16)
                        except Exception as inner:
                            log.warning(
                                "Cannot dequantize vision layer %s: %s — skipping.",
                                full_name, inner,
                            )
                            continue
                    parent_name, child_name = full_name.rsplit(".", 1)
                    parent_mod = self.model.get_submodule(parent_name)
                    new_linear = torch.nn.Linear(
                        w16.shape[1], w16.shape[0],
                        bias=module.bias is not None,
                        device=w16.device,
                        dtype=torch.float16,
                    )
                    new_linear.weight = torch.nn.Parameter(w16)
                    if module.bias is not None:
                        new_linear.bias = torch.nn.Parameter(
                            module.bias.to(device=w16.device, dtype=torch.float16)
                        )
                    setattr(parent_mod, child_name, new_linear)
                    n_dequant += 1

                if n_dequant:
                    log.info(
                        "Dequantized %d vision encoder layers to float16 "
                        "(Turing GPU / 4-bit vision kernel workaround).",
                        n_dequant,
                    )
                else:
                    log.warning(
                        "No 4-bit vision layers found to dequantize — checked "
                        "prefixes: %s.  The vision encoder may already be float16 "
                        "or use different module names.", _VISION_PREFIXES,
                    )
            except Exception as dq_err:
                log.warning(
                    "Vision encoder dequantization failed (%s) — GPU may still "
                    "crash on image input.  If errors persist, re-run without "
                    "--load_in_4bit and use google/gemma-3-4b-it instead.",
                    dq_err,
                )
        self.model.eval()

        # With device_map="auto" the model may be split across multiple GPUs.
        # Find the embedding layer's device so input_ids land on the right one.
        self._input_device = None
        for module in self.model.modules():
            if isinstance(module, torch.nn.Embedding):
                d = module.weight.device
                if d.type != "meta":
                    self._input_device = d
                    break
        if self._input_device is None:
            for p in self.model.parameters():
                if p.device.type not in ("meta", "cpu") and p.is_floating_point():
                    self._input_device = p.device
                    break
        if self._input_device is None:
            self._input_device = torch.device(
                "cuda:0" if torch.cuda.is_available() else "cpu"
            )
        log.info("Input device (embedding layer): %s", self._input_device)

        # Flush any async CUDA errors that occurred during model loading.
        if torch.cuda.is_available():
            try:
                torch.cuda.synchronize(self._input_device)
                log.info("CUDA context healthy after model load.")
            except RuntimeError as cuda_err:
                log.error(
                    "CUDA error detected right after model.from_pretrained — "
                    "likely a bitsandbytes / Turing incompatibility: %s", cuda_err
                )
                raise

        # Cast LayerNorm inputs to float16 for Turing GPU (compute 7.5) compatibility.
        # Accelerate wraps each module's forward as new_forward; patching _old_forward
        # puts the cast after accelerate's device management and before F.layer_norm.
        # FIX #10: define _ln_pre_hook once outside the loop — it uses `mod` (the
        # hook's own argument), not the loop variable, so one definition is correct
        # for all LayerNorm modules.
        def _ln_pre_hook(mod, args):
            return tuple(
                a.to(mod.weight.dtype) if isinstance(a, torch.Tensor) else a
                for a in args
            )

        def _make_ln_forward(original_forward, ln_module):
            def _forward(x):
                return original_forward(x.to(ln_module.weight.dtype))
            return _forward

        n_patched = 0
        for module in self.model.modules():
            if not isinstance(module, torch.nn.LayerNorm):
                continue
            if hasattr(module, '_old_forward'):
                module._old_forward = _make_ln_forward(module._old_forward, module)
            else:
                module.register_forward_pre_hook(_ln_pre_hook)
            n_patched += 1
        log.info(
            "Applied dtype-cast fix to %d LayerNorm layers (Turing GPU compatibility).",
            n_patched,
        )

        # FIX #19: warm-up forward pass to flush deferred bitsandbytes 4-bit
        # initialization.  When load_in_4bit=True, bitsandbytes defers some
        # CUDA kernel compilation to the first actual computation; this causes
        # the CUDA context to die at the first .to(device=...) call inside
        # chat() rather than during model loading where it would be easier to
        # diagnose.  Running a minimal embedding lookup here forces that
        # initialization to happen now, while we still have good error context.
        if torch.cuda.is_available():
            log.info("Running GPU warm-up pass (flushes deferred bitsandbytes init) ...")
            try:
                dummy = torch.zeros((1, 4), dtype=torch.long, device=self._input_device)
                with torch.no_grad():
                    _ = self.model.get_input_embeddings()(dummy)
                torch.cuda.synchronize(self._input_device)
                log.info("GPU warm-up complete — CUDA context confirmed healthy.")
            except RuntimeError as warmup_err:
                log.error(
                    "GPU warm-up FAILED.  This almost certainly means that "
                    "bitsandbytes 4-bit quantization is not compatible with "
                    "this GPU (RTX 2080 Ti / Turing compute 7.5) on the "
                    "installed CUDA/bitsandbytes versions.  "
                    "Try running without --load_in_4bit and a smaller model "
                    "(e.g. google/gemma-3-4b-it fits in ~8 GB float16): %s",
                    warmup_err,
                )
                raise

        log.info("Model loaded.")

    def chat(self, messages: list[dict], temperature: float = 0.7,
             image=None) -> str:
        """
        Run one forward pass with the given chat messages and return the response text.

        Parameters
        ----------
        messages : list[dict]
            Standard chat messages with "role" and "content" keys.  Content may
            be a plain string or a list of typed-dict parts.
        temperature : float
            Sampling temperature.  Pass 0.0 for greedy decoding.
        image : PIL.Image.Image or None
            When provided, the image is prepended to the first user turn as a
            visual input so the model can reason from the figure pixels.
        """
        import torch

        # FIX #19b: confirm CUDA context is alive at the top of every chat()
        # call so we know exactly which figure killed it.
        if torch.cuda.is_available():
            try:
                torch.cuda.synchronize(self._input_device)
            except RuntimeError as _ctx_err:
                log.error(
                    "CUDA context already dead at start of chat() — "
                    "GPU was killed by a previous operation: %s", _ctx_err
                )
                raise

        normalized = _normalize_messages_for_template(messages)

        if image is not None:
            # FIX #18: embed the PIL image directly in the message content so
            # that apply_chat_template knows the image dimensions when computing
            # how many image tokens to insert into input_ids.
            #
            # The original two-step approach (template→string, then processor
            # called separately with images=[image]) can produce a mismatch: the
            # template inserts a fixed placeholder count while the processor may
            # compute a DIFFERENT count based on the actual image resolution.
            # That input_ids / pixel_values shape disagreement is the most common
            # cause of "CUDA device-side assert triggered" on vision-language
            # models — the model indexes into pixel_values at a position that
            # doesn't exist and the CUDA kernel aborts.
            #
            # The fix is to use the one-step path where apply_chat_template
            # receives both the messages and the image at the same time, so the
            # processor can compute the correct token count before tokenising.
            structured: list[dict] = []
            first_user_done = False
            for msg in normalized:
                if msg["role"] == "user" and not first_user_done:
                    structured.append({
                        "role": "user",
                        "content": [{"type": "image", "image": image}] + msg["content"],
                    })
                    first_user_done = True
                else:
                    structured.append(msg)

            try:
                # One-step path (transformers >= 4.49): processor receives the
                # image in the message dict and generates input_ids + pixel_values
                # with guaranteed matching shapes.
                raw_inputs = self.processor.apply_chat_template(
                    structured,
                    add_generation_prompt=True,
                    tokenize=True,
                    return_dict=True,
                    return_tensors="pt",
                )
            except TypeError:
                # Fallback for older transformers that don't support
                # tokenize=True + return_dict in apply_chat_template.
                # Use two-step but keep the image in the content dict so the
                # template at least has access to it when building the string.
                prompt_text = self.processor.apply_chat_template(
                    structured,
                    add_generation_prompt=True,
                    tokenize=False,
                )
                raw_inputs = self.processor(
                    text=prompt_text,
                    images=[image],
                    return_tensors="pt",
                )
        else:
            prompt_text = self.processor.apply_chat_template(
                normalized,
                add_generation_prompt=True,
                tokenize=False,
            )
            raw_inputs = self.processor(
                text=prompt_text,
                return_tensors="pt",
            )

        # Move tensors to the embedding layer's device; cast floats to float16
        # (Turing GPUs have no bfloat16 CUDA kernels).
        inputs = {}
        for k, v in raw_inputs.items():
            if not isinstance(v, torch.Tensor):
                inputs[k] = v
                continue
            try:
                if v.is_floating_point() and v.dtype != torch.float16:
                    v = v.to(dtype=torch.float16)
                v = v.to(device=self._input_device)
            except RuntimeError as _e:
                log.error(
                    "Failed moving input '%s' (shape=%s dtype=%s) to %s: %s",
                    k, tuple(v.shape), v.dtype, self._input_device, _e,
                )
                raise
            inputs[k] = v

        input_len = inputs["input_ids"].shape[1]

        # FIX #16: validate token IDs before they reach the GPU.
        # A processor/model version mismatch (common with Gemma 3 image tokens)
        # can produce token IDs >= vocab_size, causing an unrecoverable CUDA
        # device-side assert that kills the entire GPU context.  Catching this
        # on the CPU first means the figure is skipped cleanly and the GPU
        # context survives for subsequent figures.
        # Gemma3Config nests vocab_size inside text_config; fall back to reading
        # the embedding table shape directly so this works for any model.
        vocab_size = (
            getattr(self.model.config, "vocab_size", None)
            or getattr(getattr(self.model.config, "text_config", None), "vocab_size", None)
            or self.model.get_input_embeddings().weight.shape[0]
        )
        max_token_id = int(inputs["input_ids"].max().item())
        if max_token_id >= vocab_size:
            raise ValueError(
                f"Processor generated token ID {max_token_id} which exceeds "
                f"model vocab size {vocab_size}. This is usually a "
                f"processor/model version mismatch — check that both were "
                f"loaded from the same model name."
            )

        # FIX #17: synchronize after generate() so CUDA errors surface
        # immediately at the right call rather than propagating silently to
        # the next figure and breaking the GPU context there instead.

        # FIX #11: only pass temperature/top_p when actually sampling.
        # Passing them with do_sample=False raises ValueError in transformers >= 4.46.
        do_sample = temperature > 0
        gen_kwargs: dict = dict(
            max_new_tokens=self.max_new_tokens,
            do_sample=do_sample,
        )
        if do_sample:
            gen_kwargs["temperature"] = temperature
            gen_kwargs["top_p"] = 0.9

        with torch.no_grad():
            out = self.model.generate(**inputs, **gen_kwargs)
            # FIX #17: force synchronization so any CUDA error surfaces here
            # rather than silently propagating to the next figure's .to() call.
            if torch.cuda.is_available():
                torch.cuda.synchronize(self._input_device)

        gen_tokens = out[0][input_len:]
        return self.processor.decode(gen_tokens, skip_special_tokens=True).strip()


# ---------------------------------------------------------------------------
# Agents
# ---------------------------------------------------------------------------

def extract_json(text: str) -> dict | None:
    """
    Pull the first {...} JSON object out of a model response, or return None.
    """
    text = re.sub(r"^```(json)?", "", text.strip()).strip()
    text = re.sub(r"```$", "", text).strip()

    start = text.find("{")
    if start == -1:
        return None
    try:
        obj, _ = json.JSONDecoder().raw_decode(text, start)
        if isinstance(obj, dict):
            return obj
    except json.JSONDecodeError:
        pass
    return None


def figure_context_block(row: pd.Series) -> str:
    """Format a figure's metadata into the context block shown to both agents."""
    abstract = row.get("abstract", "")
    if len(abstract) > ABSTRACT_MAX_CHARS:
        cut = abstract.rfind(" ", 0, ABSTRACT_MAX_CHARS)
        abstract = abstract[: cut if cut != -1 else ABSTRACT_MAX_CHARS] + " ..."

    sub_label = f" (sub-figure {row['sub_id']})" if row["sub_id"] else ""
    ref_text  = row["reference_text"] or "(none found)"

    return (
        f"Paper title: {row['title']}\n"
        f"Paper abstract: {abstract}\n"
        f"Paper categories: {row['categories']}\n"
        f"Figure id: {row['figure_id']}{sub_label}\n"
        f"Figure caption: {row['caption']}\n"
        f"Sub-caption: {row['sub_caption']}\n"
        f"How the figure is referenced elsewhere in the text: {ref_text}"
    )


def run_agent1(chat: GemmaChat, row: pd.Series,
               prior_query: str | None = None, feedback: str | None = None) -> str | None:
    """
    Agent 1 (Author): draft a query grounded in the figure image and its metadata.

    The figure image is loaded from disk and passed to the model so it can
    reason from the actual visual content, not just the caption text.

    If feedback is provided (whether from a rejection or a parse failure on
    the previous attempt), it is included in the prompt so Agent 1 can revise.

    Returns the query string, or None if the model response could not be parsed.
    """
    from PIL import Image as PILImage

    image = None
    image_path = row.get("image_path", "")
    if image_path and os.path.isfile(str(image_path)):
        try:
            image = PILImage.open(image_path).convert("RGB")
        except Exception as exc:
            log.warning("Could not open image %s: %s — proceeding text-only.", image_path, exc)

    context  = figure_context_block(row)
    user_msg = f"Figure information:\n{context}\n\nProduce the JSON now."

    if feedback:
        prior_note = f'Your previous attempt: "{prior_query}"\n' if prior_query else ""
        user_msg = (
            f"Figure information:\n{context}\n\n"
            f"{prior_note}"
            f"Feedback: {feedback}\n"
            f"Revise the query to address this feedback. Produce the JSON now."
        )

    messages = [
        {"role": "system", "content": AGENT1_SYSTEM},
        {"role": "user",   "content": user_msg},
    ]
    raw    = chat.chat(messages, temperature=AGENT1_TEMPERATURE, image=image)
    parsed = extract_json(raw)
    if not parsed or "query" not in parsed or not str(parsed["query"]).strip():
        return None

    query = str(parsed["query"]).strip()

    if re.search(r"<[^>]{1,60}>", query):
        log.warning(
            "Agent 1 produced a placeholder-style query %r — treating as parse failure.", query
        )
        return None

    query = re.sub(r"(?i)\b(fig(ure)?\.?\s*[0-9a-z]+)\b", "", query)
    query = re.sub(r"\s+", " ", query).strip()
    return query or None


def run_agent2(chat: GemmaChat, row: pd.Series,
               query: str) -> tuple[str, str]:
    """
    Agent 2 (Reviewer): accept or reject a proposed query.

    Returns a tuple of (verdict, feedback) where verdict is "ACCEPT" or "REJECT"
    and feedback is an empty string on acceptance or a revision note on rejection.
    """
    context  = figure_context_block(row)
    user_msg = (
        f"Figure information:\n{context}\n\n"
        f"Proposed query: \"{query}\"\n\n"
        f"Produce the JSON now."
    )
    messages = [
        {"role": "system", "content": AGENT2_SYSTEM},
        {"role": "user",   "content": user_msg},
    ]
    raw    = chat.chat(messages, temperature=AGENT2_TEMPERATURE)
    parsed = extract_json(raw)
    if not parsed or "verdict" not in parsed:
        return "REJECT", "Reviewer response could not be parsed; treating as rejection."

    verdict  = str(parsed["verdict"]).strip().upper()
    feedback = str(parsed.get("feedback", "")).strip()
    if verdict not in ("ACCEPT", "REJECT"):
        verdict  = "REJECT"
        feedback = feedback or "Malformed verdict; treating as rejection."
    return verdict, feedback


def generate_query_for_figure(chat: GemmaChat, row: pd.Series,
                               max_rounds: int = DEFAULT_MAX_ROUNDS) -> str | None:
    """
    Run the two-agent loop for a single figure.

    Agent 1 drafts a query; Agent 2 accepts or rejects it with feedback.
    Feedback from both parse failures and rejections is passed back to
    Agent 1 on the next attempt. Repeats up to max_rounds times.
    Returns the accepted query string, or None if no query was accepted.
    """
    query, feedback = None, None
    for attempt in range(1, max_rounds + 1):
        query = run_agent1(chat, row, prior_query=query, feedback=feedback)
        if query is None:
            log.info("  attempt %d: Agent 1 produced no parseable query.", attempt)
            feedback = (
                "Your previous response was not valid JSON with a 'query' field. "
                "Return only the JSON object."
            )
            continue
        verdict, feedback = run_agent2(chat, row, query)
        log.info("  attempt %d: verdict=%s  query=%r", attempt, verdict, query[:80])
        if verdict == "ACCEPT":
            return query
    return None


# ---------------------------------------------------------------------------
# Resumption helper
# ---------------------------------------------------------------------------

def load_already_done(output_tsv: str) -> set[str]:
    """
    Return the set of paper IDs that already have an accepted query in the
    output TSV (used to safely resume an interrupted run).
    """
    done: set[str] = set()
    if os.path.isfile(output_tsv):
        try:
            existing = pd.read_csv(output_tsv, sep="\t", dtype=str, keep_default_na=False)
            done = set(existing["paper_id"].tolist())
        except Exception as exc:
            log.warning(
                "Could not read existing output %s: %s — starting fresh.",
                output_tsv, exc,
            )
    return done


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(
        description="Generate grounded search queries for arXiv figures using a two-agent LLM loop.",
        epilog=(
            "Example:\n"
            "  python generate_figure_queries.py \\\n"
            "      --captions_tsv arxiv_data/captions/24_10.tsv \\\n"
            "      --metadata_tsv arxiv_data/metadata_24_10.tsv \\\n"
            "      --ref_tsv      arxiv_data/references/ref_24_10.tsv \\\n"
            "      --figures_dir  arxiv_data/figures \\\n"
            "      --output_tsv   queries_200.tsv"
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    ap.add_argument("--captions_tsv",    required=True,
                    help="TSV of figure captions from the scraper.")
    ap.add_argument("--metadata_tsv",    required=True,
                    help="TSV of per-paper metadata (title, abstract, categories, url).")
    ap.add_argument("--ref_tsv",         required=True,
                    help="TSV of in-text figure references from the scraper.")
    ap.add_argument("--figures_dir",     required=True,
                    help="Root directory of downloaded figure images.")
    ap.add_argument("--output_tsv",      default="queries_output.tsv",
                    help="Path for the output TSV (default: queries_output.tsv).")
    ap.add_argument("--target",          type=int, default=DEFAULT_TARGET,
                    help=f"Number of accepted queries to collect (default: {DEFAULT_TARGET}).")
    ap.add_argument("--max_rounds",      type=int, default=DEFAULT_MAX_ROUNDS,
                    help=f"Max review rounds per figure before skipping (default: {DEFAULT_MAX_ROUNDS}).")
    ap.add_argument("--pool_multiplier", type=int, default=DEFAULT_POOL_MULTIPLIER,
                    help=f"Candidate pool size as a multiple of --target (default: {DEFAULT_POOL_MULTIPLIER}).")
    ap.add_argument("--model_name",      default=DEFAULT_MODEL,
                    help=f"HuggingFace model name or local path (default: {DEFAULT_MODEL}).")
    ap.add_argument("--load_in_4bit",    action="store_true",
                    help="Load the model in 4-bit quantization (requires bitsandbytes).")
    ap.add_argument("--seed",            type=int, default=DEFAULT_SEED,
                    help=f"Random seed for candidate selection (default: {DEFAULT_SEED}).")
    return ap.parse_args()


def main() -> None:
    args = parse_args()

    # FIX #12: validate figures_dir early — a wrong path silently drops all
    # candidates and exits with a confusing "no images found" message.
    if not os.path.isdir(args.figures_dir):
        log.error(
            "figures_dir does not exist or is not a directory: %s", args.figures_dir
        )
        sys.exit(1)

    # FIX #13: create the output directory if it doesn't exist, so paths like
    # 'results/queries_200.tsv' don't crash at the open() call below.
    output_dir = os.path.dirname(args.output_tsv)
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)

    log.info("Loading and joining TSVs ...")
    df = load_data(args.captions_tsv, args.metadata_tsv, args.ref_tsv)
    log.info("Joined rows with non-empty captions: %d", len(df))

    log.info("Resolving image paths under %s ...", args.figures_dir)
    df = attach_image_paths(df, args.figures_dir)

    if df.empty:
        log.error("No candidate figures with images found on disk. Exiting.")
        sys.exit(1)

    order = build_diverse_candidate_order(
        df,
        target=args.target,
        pool_multiplier=args.pool_multiplier,
        seed=args.seed,
    )

    already_done = load_already_done(args.output_tsv)
    if already_done:
        log.info(
            "Resuming: %d papers already have accepted queries in %s",
            len(already_done), args.output_tsv,
        )

    # FIX #14: treat a zero-byte file as having no header (e.g. the script
    # crashed after creating the file but before writing the header line).
    write_header = (
        not os.path.isfile(args.output_tsv)
        or os.path.getsize(args.output_tsv) == 0
    )

    n_accepted  = len(already_done)
    n_attempted = 0
    n_skipped   = 0

    chat = GemmaChat(args.model_name, load_in_4bit=args.load_in_4bit)

    with open(args.output_tsv, "a", encoding="utf-8") as out_f:
        if write_header:
            out_f.write("paper_id\tfigure_id\tsub_id\tquery\n")
            out_f.flush()

        for idx in order:
            if n_accepted >= args.target:
                break
            row = df.loc[idx]
            if row["paper_id"] in already_done:
                continue

            n_attempted += 1
            log.info(
                "[%d/%d accepted | %d attempted | %d skipped] paper=%s figure=%s sub=%s",
                n_accepted, args.target, n_attempted, n_skipped,
                row["paper_id"], row["figure_id"], row["sub_id"],
            )

            try:
                query = generate_query_for_figure(chat, row, max_rounds=args.max_rounds)
            except Exception as e:
                # FIX #15: flush GPU cache after any error so subsequent figures
                # don't inherit a corrupted CUDA state from a prior OOM.
                try:
                    import torch
                    if torch.cuda.is_available():
                        torch.cuda.empty_cache()
                except Exception:
                    pass
                log.exception(
                    "Error processing paper=%s figure=%s: %s",
                    row["paper_id"], row["figure_id"], e,
                )
                query = None

            if query is None:
                n_skipped += 1
                log.info("  -> SKIPPED after %d rounds.", args.max_rounds)
                continue

            clean_query = re.sub(r"\s+", " ", query).strip().replace("\t", " ")
            out_f.write(f"{row['paper_id']}\t{row['figure_id']}\t{row['sub_id']}\t{clean_query}\n")
            out_f.flush()
            already_done.add(row["paper_id"])
            n_accepted += 1

    if n_accepted < args.target:
        log.warning(
            "Candidate pool exhausted: only reached %d / %d accepted queries. "
            "Increase --pool_multiplier or relax filters if you need more.",
            n_accepted, args.target,
        )
    else:
        log.info("Done. %d queries written to %s", n_accepted, args.output_tsv)


if __name__ == "__main__":
    main()