#!/usr/bin/env python3
import argparse
import json
import logging
import os
import random
import re
import sys
from collections import defaultdict, deque
import pandas as pd

logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)-7s | %(message)s")
log = logging.getLogger("figure_query_gen")

def load_data(captions_tsv, metadata_tsv, ref_tsv):
    df_cap = pd.read_csv(captions_tsv, sep="\t", dtype=str, keep_default_na=False)
    df_meta = pd.read_csv(metadata_tsv, sep="\t", dtype=str, keep_default_na=False)
    df_ref = pd.read_csv(ref_tsv, sep="\t", dtype=str, keep_default_na=False)
    for col in ["paper_id", "figure_id", "sub_id", "caption", "sub_caption"]:
        if col not in df_cap.columns:
            df_cap[col] = ""
    for col in ["url", "paper_id", "title", "abstract", "categories"]:
        if col not in df_meta.columns:
            df_meta[col] = ""
    for col in ["paper_id", "figure_id", "reference_text"]:
        if col not in df_ref.columns:
            df_ref[col] = ""
    df_cap = df_cap.fillna("")
    df_meta = df_meta.fillna("")
    df_ref = df_ref.fillna("")
    df_ref_grouped = (
        df_ref.groupby(["paper_id", "figure_id"])["reference_text"]
        .apply(lambda s: " || ".join([x for x in s if x.strip()]))
        .reset_index()
    )
    df = df_cap.merge(df_meta, on="paper_id", how="left", suffixes=("", "_meta"))
    df = df.merge(df_ref_grouped, on=["paper_id", "figure_id"], how="left")
    df["reference_text"] = df["reference_text"].fillna("")
    df = df[df["caption"].str.strip() != ""].reset_index(drop=True)
    return df

def resolve_image_path(figures_dir, paper_id, figure_id, sub_id):
    if "." not in paper_id or len(paper_id.split(".")[0]) < 4:
        return None
    prefix = paper_id.split(".")[0]
    yy, mm = prefix[:2], prefix[2:4]
    fname_core = f"{figure_id}{sub_id}" if sub_id else f"{figure_id}"
    candidates = [f"{fname_core}.png", f"{fname_core}.jpg", f"{fname_core}.jpeg"]
    paper_dir = os.path.join(figures_dir, yy, mm, paper_id)
    for c in candidates:
        p = os.path.join(paper_dir, c)
        if os.path.isfile(p):
            return p
    return None

def attach_image_paths(df, figures_dir):
    paths = []
    for _, row in df.iterrows():
        p = resolve_image_path(figures_dir, row["paper_id"], row["figure_id"], row["sub_id"])
        paths.append(p)
    df = df.copy()
    df["image_path"] = paths
    before = len(df)
    df = df[df["image_path"].notna()].reset_index(drop=True)
    log.info("Resolved images: %d / %d rows have a matching file on disk.", len(df), before)
    return df

def primary_category(categories_str):
    if pd.isna(categories_str):
        return "unknown"
    categories_str = str(categories_str).strip()
    if not categories_str:
        return "unknown"
    return categories_str.split()[0]

def caption_len_bucket(caption, sub_caption):
    text = f"{caption} {sub_caption}".strip()
    n_words = len(text.split())
    if n_words < 20:
        return "short"
    elif n_words <= 60:
        return "medium"
    else:
        return "long"

def build_diverse_candidate_order(df, pool_multiplier=4, seed=13):
    rng = random.Random(seed)
    df = df.copy()
    df["primary_category"] = df["categories"].apply(primary_category)
    df["len_bucket"] = df.apply(lambda r: caption_len_bucket(r["caption"], r["sub_caption"]), axis=1)
    one_per_paper = []
    for paper_id, group in df.groupby("paper_id"):
        idxs = list(group.index)
        rng.shuffle(idxs)
        one_per_paper.append(idxs[0])
    rng.shuffle(one_per_paper)
    picked_df = df.loc[one_per_paper]
    by_category = defaultdict(list)
    for idx, row in picked_df.iterrows():
        by_category[row["primary_category"]].append(idx)
    for cat, idxs in by_category.items():
        by_len = defaultdict(list)
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
    cat_names = list(by_category.keys())
    rng.shuffle(cat_names)
    cat_queues = {c: deque(by_category[c]) for c in cat_names}
    order = []
    while any(cat_queues.values()):
        for c in cat_names:
            if cat_queues[c]:
                order.append(cat_queues[c].popleft())
    log.info("Candidate pool built: %d papers across %d categories.", len(order), len(cat_names))
    return order

class GemmaChat:
    def __init__(self, model_name, load_in_4bit=False, max_new_tokens=400):
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer
        self.max_new_tokens = max_new_tokens
        log.info("Loading model %s ...", model_name)
        quant_kwargs = {}
        if load_in_4bit:
            from transformers import BitsAndBytesConfig
            quant_kwargs["quantization_config"] = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_compute_dtype=torch.bfloat16)
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForCausalLM.from_pretrained(model_name, device_map="auto", torch_dtype=torch.bfloat16, **quant_kwargs)
        self.model.eval()
        log.info("Model loaded.")

    def chat(self, messages, temperature=0.7):
        import torch
        parts = []
        for m in messages:
            if m["role"] == "system":
                parts.append(m["content"])
            elif m["role"] == "user":
                parts.append("User: " + m["content"])
            elif m["role"] == "assistant":
                parts.append("Assistant: " + m["content"])
        parts.append("Assistant:")
        prompt = "\n\n".join(parts)
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
        text = self.tokenizer.decode(gen_tokens, skip_special_tokens=True)
        return text.strip()

def extract_json(text):
    text = text.strip()
    text = re.sub(r"^```(json)?", "", text).strip()
    text = re.sub(r"```$", "", text).strip()
    match = re.search(r"\{.*\}", text, re.DOTALL)
    if not match:
        return None
    try:
        return json.loads(match.group(0))
    except json.JSONDecodeError:
        return None

AGENT1_SYSTEM = """You are simulating a working scientist who has just come across a figure in a paper. Your job is to produce ONE realistic information need - a natural-language query answerable by LOOKING AT THE FIGURE ITSELF.
Constraints:
- Do NOT say "Figure X" or reference the figure number.
- Must be answerable from the figure alone.
- Must be specific, not generic.
- One to two sentences.
Respond with ONLY: {"query": "<the query text>"}"""

AGENT2_SYSTEM = """You are a peer reviewer checking a query for a figure-grounded QA benchmark. ACCEPT only if ALL hold:
1. Answerable by the figure alone.
2. Specific to this figure.
3. Grammatical, one to two sentences.
4. Does not give away the answer.
5. Not a verbatim caption restatement.
6. Does not say "Figure X".
Respond with ONLY: {"verdict": "ACCEPT" or "REJECT", "feedback": "<empty if ACCEPT, else fix needed>"}"""

def figure_context_block(row):
    abstract = row.get("abstract", "")
    if len(abstract) > 1500:
        abstract = abstract[:1500] + " ..."
    return (
        f"Paper title: {row['title']}\n"
        f"Paper abstract: {abstract}\n"
        f"Paper categories: {row['categories']}\n"
        f"Figure id: {row['figure_id']}" + (f" (sub-figure {row['sub_id']})" if row["sub_id"] else "") + "\n"
        f"Figure caption: {row['caption']}\n"
        f"Sub-caption: {row['sub_caption']}\n"
        f"Referenced in text: {row['reference_text'] or '(none found)'}"
    )

def run_agent1(chat, row, prior_query=None, feedback=None):
    context = figure_context_block(row)
    user_msg = f"Figure information:\n{context}\n\nProduce the JSON now."
    if prior_query is not None and feedback:
        user_msg = f"Figure information:\n{context}\n\nPrevious attempt: \"{prior_query}\"\nFeedback: {feedback}\nRevise and produce the JSON now."
    messages = [{"role": "system", "content": AGENT1_SYSTEM}, {"role": "user", "content": user_msg}]
    raw = chat.chat(messages, temperature=0.8)
    parsed = extract_json(raw)
    if not parsed or "query" not in parsed or not str(parsed["query"]).strip():
        return None
    query = str(parsed["query"]).strip()
    query = re.sub(r"(?i)\b(fig(ure)?\.?\s*[0-9a-z]+)\b", "", query)
    query = re.sub(r"\s+", " ", query).strip()
    return query if query else None

def run_agent2(chat, row, query):
    context = figure_context_block(row)
    user_msg = f"Figure information:\n{context}\n\nProposed query: \"{query}\"\n\nProduce the JSON now."
    messages = [{"role": "system", "content": AGENT2_SYSTEM}, {"role": "user", "content": user_msg}]
    raw = chat.chat(messages, temperature=0.3)
    parsed = extract_json(raw)
    if not parsed or "verdict" not in parsed:
        return "REJECT", "Could not parse reviewer response."
    verdict = str(parsed["verdict"]).strip().upper()
    feedback = str(parsed.get("feedback", "")).strip()
    if verdict not in ("ACCEPT", "REJECT"):
        verdict = "REJECT"
        feedback = feedback or "Malformed verdict."
    return verdict, feedback

def generate_query_for_figure(chat, row, max_rounds=3):
    query, feedback = None, None
    for attempt in range(1, max_rounds + 1):
        query = run_agent1(chat, row, prior_query=query, feedback=feedback)
        if query is None:
            log.info("  attempt %d: Agent 1 produced no parseable query.", attempt)
            feedback = "Return only the JSON object with a query field."
            continue
        verdict, feedback = run_agent2(chat, row, query)
        log.info("  attempt %d: verdict=%s query=%r", attempt, verdict, query[:80])
        if verdict == "ACCEPT":
            return query
    return None

def load_already_done(output_tsv):
    done = set()
    if os.path.isfile(output_tsv):
        try:
            existing = pd.read_csv(output_tsv, sep="\t", dtype=str, keep_default_na=False)
            for _, r in existing.iterrows():
                done.add(r["paper_id"])
        except Exception:
            pass
    return done

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--captions_tsv", required=True)
    ap.add_argument("--metadata_tsv", required=True)
    ap.add_argument("--ref_tsv", required=True)
    ap.add_argument("--figures_dir", required=True)
    ap.add_argument("--output_tsv", default="queries_output.tsv")
    ap.add_argument("--target", type=int, default=200)
    ap.add_argument("--max_rounds", type=int, default=3)
    ap.add_argument("--pool_multiplier", type=int, default=4)
    ap.add_argument("--model_name", default="google/gemma-4-E2B")
    ap.add_argument("--load_in_4bit", action="store_true")
    ap.add_argument("--seed", type=int, default=13)
    args = ap.parse_args()
    log.info("Loading and joining TSVs ...")
    df = load_data(args.captions_tsv, args.metadata_tsv, args.ref_tsv)
    log.info("Joined rows with non-empty captions: %d", len(df))
    log.info("Resolving image paths under %s ...", args.figures_dir)
    df = attach_image_paths(df, args.figures_dir)
    if df.empty:
        log.error("No candidate figures with images found on disk. Exiting.")
        sys.exit(1)
    order = build_diverse_candidate_order(df, pool_multiplier=args.pool_multiplier, seed=args.seed)
    already_done_papers = load_already_done(args.output_tsv)
    if already_done_papers:
        log.info("Resuming: %d papers already done.", len(already_done_papers))
    write_header = not os.path.isfile(args.output_tsv)
    out_f = open(args.output_tsv, "a", encoding="utf-8")
    if write_header:
        out_f.write("paper_id\tfigure_id\tsub_id\tquery\n")
        out_f.flush()
    n_accepted = len(already_done_papers)
    n_attempted = 0
    n_skipped = 0
    chat = GemmaChat(args.model_name, load_in_4bit=args.load_in_4bit)
    for idx in order:
        if n_accepted >= args.target:
            break
        row = df.loc[idx]
        if row["paper_id"] in already_done_papers:
            continue
        n_attempted += 1
        log.info("[%d accepted / %d target | %d attempted | %d skipped] paper=%s figure=%s%s",
            n_accepted, args.target, n_attempted, n_skipped, row["paper_id"], row["figure_id"], row["sub_id"])
        try:
            query = generate_query_for_figure(chat, row, max_rounds=args.max_rounds)
        except Exception as e:
            log.exception("Error: %s", e)
            query = None
        if query is None:
            n_skipped += 1
            log.info("  -> SKIPPED after %d rounds.", args.max_rounds)
            continue
        clean_query = re.sub(r"\s+", " ", query).strip().replace("\t", " ")
        out_f.write(f"{row['paper_id']}\t{row['figure_id']}\t{row['sub_id']}\t{clean_query}\n")
        out_f.flush()
        already_done_papers.add(row["paper_id"])
        n_accepted += 1
    out_f.close()
    if n_accepted < args.target:
        log.warning("Candidate pool exhausted: only reached %d / %d.", n_accepted, args.target)
    else:
        log.info("Done. %d queries written to %s", n_accepted, args.output_tsv)

if __name__ == "__main__":
    main()
