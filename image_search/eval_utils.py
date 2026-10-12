"""
Shared utilities for the ColPali training / retrieval pipeline:
  - indexing the image corpus (doc_id -> path)
  - loading query JSON files and qrels TSV files
  - computing nDCG@k / MRR / Recall@k for a run against qrels
  - sanity-checking that qrels doc_ids actually exist in the indexed corpus

Kept dependency-free (no pytrec_eval / ranx) so it runs in restricted /
offline conda environments.
"""

import csv
import json
import math

from pathlib import Path

from PIL import Image, ImageFile, PngImagePlugin

IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp"}

# Pillow's decompression-bomb guards (pixel-count cap + max decompressed size
# for PNG text chunks) exist to protect servers from malicious uploads by
# strangers. This is a trusted local research corpus of large scientific
# figures (often with big embedded PNG metadata from PDF->PNG conversion
# tools), so the defaults are both too strict and, once exceeded, fatal
# (ValueError) rather than just a warning. Raise them process-wide, once,
# here -- every training/retrieval script imports this module.
Image.MAX_IMAGE_PIXELS = None  # disable the pixel-count DecompressionBombWarning/error
PngImagePlugin.MAX_TEXT_CHUNK = 200 * 1024 * 1024  # was 1 MB; some figures embed larger zTXt metadata
ImageFile.LOAD_TRUNCATED_IMAGES = True  # tolerate partially-written/truncated files too


def safe_open_image(path, fallback_size=None):
    """
    Open `path` as an RGB PIL Image, tolerating decode failures.

    On failure, logs a warning and:
      - returns a blank white placeholder of `fallback_size` (a (width,
        height) tuple) if one is given -- only for places where a fixed-size,
        index-aligned list of images is required; or
      - returns None if `fallback_size` is omitted -- the caller filters the
        failed item out. Prefer this wherever possible: a blank placeholder
        paired with a real query is a corrupted training/eval example.
    """
    try:
        img = Image.open(path)
        img.load()
        return img.convert("RGB")
    except Exception as e:
        print(f"[WARN] Failed to load image {path}: {e}")
        if fallback_size is not None:
            return Image.new("RGB", fallback_size, color=(255, 255, 255))
        return None


# ---------------------------------------------------------------------------
# Corpus indexing
# ---------------------------------------------------------------------------

def find_images(image_dir):
    """Recursively find image files under image_dir, return sorted list of Paths."""
    image_dir = Path(image_dir)
    paths = [
        p for p in sorted(image_dir.rglob("*"))
        if p.suffix.lower() in IMAGE_EXTS and p.is_file()
    ]
    return paths


def doc_id_from_path(path: Path) -> str:
    """Figure/doc id = filename without extension."""
    return Path(path).stem


def filename_to_qrels_id(stem):
    """
    Hook for converting an on-disk filename stem into the id convention used
    by the qrels files.

    Currently a no-op: it returns the stem unchanged, because the qrels doc_ids
    in this project are plain filename stems (e.g. '2504.15247_10'). If a qrels
    file ever uses a different convention, implement the conversion here so
    training, validation and retrieval all pick it up consistently.
    """
    return stem


def canonical_doc_items(image_dir, image_paths=None):
    """
    Returns a list of (doc_id, Path), EXACTLY ONE entry per physical image
    file -- safe to iterate for a full-corpus scan (no duplicate embedding).

    doc_id is:
      1. filename_to_qrels_id(stem), if that changes the stem (currently never);
      2. else, if the image sits in a subfolder of image_dir, the relative path
         with its parts joined by '_' (e.g. 'sub/2504.15247_10.png' ->
         'sub_2504.15247_10');
      3. else, the bare filename stem.

    NOTE: case 2 means corpora with subfolders get doc_ids that are NOT plain
    filename stems. If your qrels use plain stems, retrieval run files will
    not match them. Use report_qrels_coverage() to check this.
    """
    image_dir = Path(image_dir)
    if image_paths is None:
        image_paths = find_images(image_dir)

    items = []
    seen = {}
    dupes = 0
    for p in image_paths:
        stem = p.stem
        doc_id = filename_to_qrels_id(stem)
        if doc_id == stem:
            rel = p.relative_to(image_dir).with_suffix("")
            if len(rel.parts) > 1:
                doc_id = "_".join(rel.parts)
        if doc_id in seen:
            dupes += 1
            continue
        seen[doc_id] = p
        items.append((doc_id, p))

    if dupes:
        print(f"[WARN] {dupes} duplicate canonical doc_id(s) while indexing images; "
              f"kept the first occurrence of each.")
    return items


def build_doc_index(image_dir, image_paths=None):
    """
    Returns a dict doc_id -> Path for MEMBERSHIP / PATH-LOOKUP use. Includes
    both the canonical id (see canonical_doc_items) and, as a fallback alias,
    the raw filename stem -- so a lookup succeeds whichever convention the
    qrels file uses.

    Do NOT iterate this dict to scan the corpus -- the same physical file may
    appear under two keys. Use canonical_doc_items() for a full-corpus scan.
    """
    items = canonical_doc_items(image_dir, image_paths)
    lookup = {doc_id: p for doc_id, p in items}
    for doc_id, p in items:
        stem = p.stem
        lookup.setdefault(stem, p)
    return lookup


def report_qrels_coverage(qrels, corpus_ids, label="qrels"):
    """
    Print how many relevant doc_ids in `qrels` exist among `corpus_ids`
    (the ids a retrieval run can actually return). A low number means the
    run file and qrels use different doc_id conventions, which silently
    drives every metric toward zero.
    Returns (n_found, n_relevant).
    """
    corpus_ids = set(corpus_ids)
    relevant = {d for docs in qrels.values() for d, r in docs.items() if r > 0}
    found = relevant & corpus_ids
    missing = sorted(relevant - corpus_ids)
    pct = 100.0 * len(found) / max(1, len(relevant))
    print(f"[COVERAGE] {label}: {len(found)}/{len(relevant)} relevant doc_ids "
          f"({pct:.1f}%) are retrievable ids in the corpus.")
    if missing:
        print(f"  [WARN] {len(missing)} relevant doc_ids can never be retrieved. Examples: {missing[:5]}")
        print(f"  Example corpus ids: {sorted(corpus_ids)[:5]}")
    return len(found), len(relevant)


# ---------------------------------------------------------------------------
# Query / qrels loading
# ---------------------------------------------------------------------------

def load_queries(json_path):
    """
    Load a queries JSON file (list of dicts, or {"queries": [...]}), returning
    an ordered dict: str(query_id) -> query text.
    """
    with open(json_path, "r", encoding="utf-8") as f:
        data = json.load(f)
    if isinstance(data, dict):
        data = data.get("queries", list(data.values()))
    queries = {}
    for item in data:
        qid = str(item["query_id"])
        queries[qid] = item["query"]
    return queries


def load_qrels(tsv_path):
    """
    Load a TREC-style qrels file. Accepts either:
      qid  docid  relevance          (3 columns)
      qid  0      docid  relevance   (4 columns, standard TREC)
    Whitespace or tab separated. Returns dict: qid -> {docid: relevance(int)}.
    """
    qrels = {}
    with open(tsv_path, "r", encoding="utf-8") as f:
        sniffed = f.readline()
        f.seek(0)
        delimiter = "\t" if "\t" in sniffed else None  # None => split on whitespace
        reader = csv.reader(f, delimiter=delimiter) if delimiter else (
            line.split() for line in f
        )
        for row in reader:
            row = [c for c in row if c != ""]
            if not row:
                continue
            if len(row) == 3:
                qid, docid, rel = row
            elif len(row) == 4:
                qid, _iter, docid, rel = row
            else:
                # tolerate stray header / malformed lines
                continue
            try:
                rel = int(float(rel))
            except ValueError:
                continue  # likely a header row
            qid = str(qid)
            qrels.setdefault(qid, {})[docid] = rel
    return qrels


# ---------------------------------------------------------------------------
# Metrics
# ---------------------------------------------------------------------------

def dcg_at_k(relevances, k):
    dcg = 0.0
    for i, rel in enumerate(relevances[:k], start=1):
        if rel > 0:
            dcg += (2 ** rel - 1) / math.log2(i + 1)
    return dcg


def ndcg_at_k(ranked_docids, qrel_for_query, k):
    rels = [qrel_for_query.get(d, 0) for d in ranked_docids[:k]]
    dcg = dcg_at_k(rels, k)
    ideal_rels = sorted(qrel_for_query.values(), reverse=True)
    idcg = dcg_at_k(ideal_rels, k)
    if idcg == 0:
        return 0.0
    return dcg / idcg


def reciprocal_rank(ranked_docids, qrel_for_query):
    for i, d in enumerate(ranked_docids, start=1):
        if qrel_for_query.get(d, 0) > 0:
            return 1.0 / i
    return 0.0


def recall_at_k(ranked_docids, qrel_for_query, k):
    n_relevant = sum(1 for r in qrel_for_query.values() if r > 0)
    if n_relevant == 0:
        return 0.0
    n_found = sum(1 for d in ranked_docids[:k] if qrel_for_query.get(d, 0) > 0)
    return n_found / n_relevant


def evaluate_run(run, qrels, k_list=(10,)):
    """
    run:   dict qid -> list of docids, ranked best-first
    qrels: dict qid -> {docid: relevance}
    Returns dict of averaged metrics across all queries present in qrels
    (and also present in run).
    """
    metrics = {f"ndcg@{k}": [] for k in k_list}
    metrics.update({f"recall@{k}": [] for k in k_list})
    metrics["mrr"] = []

    missing_from_run = 0
    for qid, qrel_for_query in qrels.items():
        if qid not in run:
            missing_from_run += 1
            continue
        ranked = run[qid]
        for k in k_list:
            metrics[f"ndcg@{k}"].append(ndcg_at_k(ranked, qrel_for_query, k))
            metrics[f"recall@{k}"].append(recall_at_k(ranked, qrel_for_query, k))
        metrics["mrr"].append(reciprocal_rank(ranked, qrel_for_query))

    if missing_from_run:
        print(f"[WARN] {missing_from_run} qrels queries have no entry in the run (excluded from averages).")

    return {name: (sum(vals) / len(vals) if vals else 0.0) for name, vals in metrics.items()}


def write_trec_run(run_scores, output_path, run_tag="colpali", top_k=100):
    """
    run_scores: dict qid -> list of (score, docid), NOT necessarily sorted.
    Writes standard TREC format: qid Q0 docid rank score run_tag
    """
    with open(output_path, "w", encoding="utf-8") as f:
        for qid, scored in run_scores.items():
            ranked = sorted(scored, key=lambda x: x[0], reverse=True)[:top_k]
            for rank, (score, docid) in enumerate(ranked, start=1):
                f.write(f"{qid} Q0 {docid} {rank} {score:.6f} {run_tag}\n")