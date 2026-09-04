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
MAX_NEW_TOKENS     = 400
AGENT1_TEMPERATURE = 0.8   # higher for creative query drafting
AGENT2_TEMPERATURE = 0.3   # lower for consistent accept/reject decisions
 
# Figure context construction
ABSTRACT_MAX_CHARS = 1500  # truncate abstracts beyond this to save tokens
 
# Caption-length bucket thresholds (in words)
CAPTION_SHORT_MAX  = 20
CAPTION_MEDIUM_MAX = 60
 
# Agent system prompts
AGENT1_SYSTEM = """You are simulating a working scientist who has just come \
across a figure in a paper (via its caption, sub-caption, abstract, and how \
other papers cite it). Your job is to produce ONE realistic information need \
this scientist would have -- a natural-language query that can be answered by \
LOOKING AT THE FIGURE ITSELF (not by reading the rest of the paper).
 
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
- "What is the trend of training loss for the proposed method compared to the \
baseline over epochs?"
- "Which of the three mechanisms produces the highest output under low capital \
levels?"
 
Respond with ONLY a JSON object: {"query": "<the query text>"}"""
 
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
    df["image_path"] = [
        resolve_image_path(figures_dir, row["paper_id"], row["figure_id"], row["sub_id"])
        for _, row in df.iterrows()
    ]
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
    by_category: dict[str, list] = defaultdict(list)
    for idx, row in picked_df.iterrows():
        by_category[row["primary_category"]].append(idx)
 
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
 
class GemmaChat:
    """Thin wrapper around a local HuggingFace causal LM used as a chat model."""
 
    def __init__(self, model_name: str, load_in_4bit: bool = False,
                 max_new_tokens: int = MAX_NEW_TOKENS):
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer
 
        self.max_new_tokens = max_new_tokens
        log.info("Loading model %s ...", model_name)
 
        quant_kwargs = {}
        if load_in_4bit:
            from transformers import BitsAndBytesConfig
            quant_kwargs["quantization_config"] = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_compute_dtype=torch.bfloat16,
            )
 
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForCausalLM.from_pretrained(
            model_name,
            device_map="auto",
            torch_dtype=torch.bfloat16,
            **quant_kwargs,
        )
        self.model.eval()
        log.info("Model loaded.")
 
    def chat(self, messages: list[dict], temperature: float = 0.7) -> str:
        """Run one forward pass with the given chat messages and return the response text."""
        import torch
 
        prompt = self.tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
        inputs = self.tokenizer(prompt, return_tensors="pt").to(self.model.device)
 
        with torch.no_grad():
            out = self.model.generate(
                **inputs,
                max_new_tokens=self.max_new_tokens,
                do_sample=temperature > 0,
                temperature=max(temperature, 1e-4),
                top_p=0.9,
                pad_token_id=self.tokenizer.eos_token_id,
            )
        gen_tokens = out[0][inputs["input_ids"].shape[1]:]
        return self.tokenizer.decode(gen_tokens, skip_special_tokens=True).strip()
 
 
# ---------------------------------------------------------------------------
# Agents
# ---------------------------------------------------------------------------
 
def extract_json(text: str) -> dict | None:
    """Pull the first {...} JSON object out of a model response, or return None."""
    text = re.sub(r"^```(json)?", "", text.strip()).strip()
    text = re.sub(r"```$", "", text).strip()
    match = re.search(r"\{.*\}", text, re.DOTALL)
    if not match:
        return None
    try:
        return json.loads(match.group(0))
    except json.JSONDecodeError:
        return None
 
 
def figure_context_block(row: pd.Series) -> str:
    """Format a figure's metadata into the context block shown to both agents."""
    abstract = row.get("abstract", "")
    if len(abstract) > ABSTRACT_MAX_CHARS:
        abstract = abstract[:ABSTRACT_MAX_CHARS] + " ..."
 
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
    Agent 1 (Author): draft a query from the figure context.
 
    If prior_query and feedback are provided, revise the previous attempt
    based on the reviewer's feedback.
 
    Returns the query string, or None if the model response could not be parsed.
    """
    context  = figure_context_block(row)
    user_msg = f"Figure information:\n{context}\n\nProduce the JSON now."
    if prior_query is not None and feedback:
        user_msg = (
            f"Figure information:\n{context}\n\n"
            f"Your previous attempt: \"{prior_query}\"\n"
            f"Reviewer feedback: {feedback}\n"
            f"Revise the query to address this feedback. Produce the JSON now."
        )
 
    messages = [
        {"role": "system", "content": AGENT1_SYSTEM},
        {"role": "user",   "content": user_msg},
    ]
    raw    = chat.chat(messages, temperature=AGENT1_TEMPERATURE)
    parsed = extract_json(raw)
    if not parsed or "query" not in parsed or not str(parsed["query"]).strip():
        return None
 
    query = str(parsed["query"]).strip()
    # Remove any accidental "Figure X" references that would violate the benchmark rules.
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
    Repeats up to max_rounds times. Returns the accepted query string, or
    None if no query was accepted within the round limit.
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
        except Exception:
            pass
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
 
    write_header = not os.path.isfile(args.output_tsv)
    out_f = open(args.output_tsv, "a", encoding="utf-8")
    if write_header:
        out_f.write("paper_id\tfigure_id\tsub_id\tquery\n")
        out_f.flush()
 
    n_accepted  = len(already_done)
    n_attempted = 0
    n_skipped   = 0
 
    chat = GemmaChat(args.model_name, load_in_4bit=args.load_in_4bit)
 
    for idx in order:
        if n_accepted >= args.target:
            break
        row = df.loc[idx]
        if row["paper_id"] in already_done:
            continue
 
        n_attempted += 1
        log.info(
            "[%d/%d accepted | %d attempted | %d skipped] paper=%s figure=%s%s",
            n_accepted, args.target, n_attempted, n_skipped,
            row["paper_id"], row["figure_id"], row["sub_id"],
        )
 
        try:
            query = generate_query_for_figure(chat, row, max_rounds=args.max_rounds)
        except Exception as e:
            log.exception(
                "Error processing paper=%s figure=%s: %s",
                row["paper_id"], row["figure_id"], e,
            )
            query = None
 
        if query is None:
            n_skipped += 1
            log.info("  -> SKIPPED after %d rounds.", args.max_rounds)
            continue
 
        # Strip tabs and newlines so the TSV stays well-formed.
        clean_query = re.sub(r"\s+", " ", query).strip().replace("\t", " ")
        out_f.write(f"{row['paper_id']}\t{row['figure_id']}\t{row['sub_id']}\t{clean_query}\n")
        out_f.flush()
        already_done.add(row["paper_id"])
        n_accepted += 1
 
    out_f.close()
 
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
 