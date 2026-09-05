#!/usr/bin/env python3
"""
table_sampler.py
================
Reads output files produced by table_scraper.py and arxiv_scraper.py,
merges them, and draws a reproducible stratified random sample of exactly
200 unique tables while maintaining category diversity.

Input files (auto-discovered under OUTPUT_DIR):
  tables/<year>_<month>.jsonl    -- one JSON record per table (table_scraper.py)
  table_metadata_<year>_<month>.tsv   -- one row per paper  (table_scraper.py)

Both scrapers may produce multiple files (one per month); this script
globs for all of them and concatenates before sampling.

Output:
  sampled_tables.tsv -- 200-row TSV with the columns defined in OUTPUT_COLUMNS

Usage
-----
  python table_sampler.py
  python table_sampler.py /path/to/arxiv_data
  python table_sampler.py /path/to/arxiv_data /path/to/sampled_tables.tsv

Example
-------
  python table_sampler.py arxiv_data sampled_tables.tsv
"""

from __future__ import annotations

import argparse
import json
import logging
import math
import random
import sys
from pathlib import Path
from typing import Dict, List

import pandas as pd

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

# Root directory that the scrapers wrote into (contains tables/ and table_metadata_*.tsv)
OUTPUT_DIR = Path("arxiv_data")

OUTPUT_TSV  = Path("sampled_tables.tsv")

SAMPLE_SIZE = 200
RANDOM_SEED = 42

# Separator used to join multiple in-text reference sentences for one table.
REFERENCE_SEPARATOR = " | "

# Base URL used to construct paper and table URLs from paper_id / html_id.
ARXIV_HTML_BASE = "https://ar5iv.labs.arxiv.org/html/"

# Exact column order in the output file.
OUTPUT_COLUMNS = [
    "paper_id",
    "table_id",
    "category",
    "paper_url",
    "table_url",
    "caption",
    "reference_text",
    "title",
    "abstract",
]

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# 1. Loading
# ---------------------------------------------------------------------------

def load_tables(output_dir: Path) -> pd.DataFrame:
    """
    Load every tables/<year>_<month>.jsonl file under output_dir and
    return a single concatenated DataFrame.

    Each JSONL line is one table record written by table_scraper.py.
    The references field is a list of paragraph strings; we join them
    here into a single pipe-separated string (reference_text).

    The JSONL records do not carry an explicit paper_id column -- we
    derive it by stripping the trailing _T<num>[<letter>] suffix from
    table_id (e.g. "2410.00004_T1" -> "2410.00004").
    """
    jsonl_files = sorted((output_dir / "tables").glob("*.jsonl"))
    if not jsonl_files:
        raise FileNotFoundError(
            f"No JSONL files found under {output_dir / 'tables'}.\n"
            "Check that table_scraper.py has finished writing output there."
        )

    frames: List[pd.DataFrame] = []
    for path in jsonl_files:
        log.info("Loading tables from %s ...", path)
        df = pd.read_json(path, lines=True, dtype=False)
        frames.append(df)

    tables_df = pd.concat(frames, ignore_index=True)

    # Derive paper_id from table_id.
    # table_id format: "<paper_id>_T<num>" or "<paper_id>_T<num><letter>"
    tables_df["paper_id"] = tables_df["table_id"].str.extract(r"^(.+?)_T\d+", expand=False)

    # Flatten the references list into a single string.
    def _join_refs(val) -> str:
        if isinstance(val, list):
            return REFERENCE_SEPARATOR.join(s for s in val if s)
        if pd.isna(val):
            return ""
        return str(val)

    tables_df["reference_text"] = tables_df["references"].apply(_join_refs)

    # Guard: caption column may be absent from JSONL records.
    if "caption" not in tables_df.columns:
        tables_df["caption"] = ""
    else:
        tables_df["caption"] = tables_df["caption"].fillna("")

    log.info(
        "Tables loaded: %d records from %d file(s).",
        len(tables_df), len(jsonl_files),
    )
    return tables_df


def load_metadata(output_dir: Path) -> pd.DataFrame:
    """
    Load every table_metadata_<year>_<month>.tsv file under output_dir and
    return a single concatenated DataFrame.

    Columns written by arxiv_scraper.py: url, paper_id, title, abstract, categories.
    The categories field uses "; " (semicolon) as the separator between
    multiple category labels (e.g. "cs.LG; cs.CL").
    """
    tsv_files = sorted(output_dir.glob("table_metadata_*.tsv"))
    if not tsv_files:
        raise FileNotFoundError(
            f"No table_metadata TSV files found under {output_dir}.\n"
            "Check that table_scraper.py has finished writing output there."
        )

    frames: List[pd.DataFrame] = []
    for path in tsv_files:
        log.info("Loading metadata from %s ...", path)
        df = pd.read_csv(path, sep="\t", dtype=str)
        frames.append(df)

    metadata_df = pd.concat(frames, ignore_index=True)

    # Normalise column names.
    metadata_df.columns = (
        metadata_df.columns
        .str.strip()
        .str.lower()
        .str.replace(" ", "_", regex=False)
    )

    # Rename url -> paper_url to match OUTPUT_COLUMNS.
    if "url" in metadata_df.columns and "paper_url" not in metadata_df.columns:
        metadata_df = metadata_df.rename(columns={"url": "paper_url"})

    # Drop duplicate paper_id rows (can happen if scraper was restarted).
    before = len(metadata_df)
    metadata_df = metadata_df.drop_duplicates(subset=["paper_id"], keep="first")
    if len(metadata_df) < before:
        log.warning(
            "Dropped %d duplicate paper rows in metadata (kept first occurrence).",
            before - len(metadata_df),
        )

    for col in ["title", "abstract", "categories", "paper_url"]:
        if col in metadata_df.columns:
            metadata_df[col] = metadata_df[col].fillna("").str.strip()
        else:
            metadata_df[col] = ""

    log.info(
        "Metadata loaded: %d papers from %d file(s).",
        len(metadata_df), len(tsv_files),
    )
    return metadata_df


# ---------------------------------------------------------------------------
# 2. Merging
# ---------------------------------------------------------------------------

def merge_dataframes(
    tables_df: pd.DataFrame,
    metadata_df: pd.DataFrame,
) -> pd.DataFrame:
    """
    Join the tables DataFrame with paper metadata on paper_id.

    Uses a LEFT join so tables whose paper is absent from the metadata TSV
    are still kept in the pool with blank title / abstract / category fields.
    """
    merged = tables_df.merge(metadata_df, on="paper_id", how="left")

    for col in ["title", "abstract", "categories", "paper_url", "caption", "reference_text"]:
        if col in merged.columns:
            merged[col] = merged[col].fillna("").astype(str).str.strip()
        else:
            merged[col] = ""

    n_missing_meta = (merged["title"] == "").sum()
    if n_missing_meta:
        log.warning(
            "%d table(s) have no matching paper metadata "
            "(title / abstract / url will be blank in the output).",
            n_missing_meta,
        )

    log.info("Merged DataFrame: %d rows, %d columns.", *merged.shape)
    return merged


# ---------------------------------------------------------------------------
# 3. Category index
# ---------------------------------------------------------------------------

def build_category_index(merged_df: pd.DataFrame) -> Dict[str, List[str]]:
    """
    Build a mapping of {category: [table_id, ...]}.

    Each table is placed into every category its paper belongs to (semicolon-
    separated in the categories field), so one table_id can appear under
    multiple keys. Tables with no category go into an _uncategorised bucket.
    Within each bucket, table_id values are deduplicated preserving first-seen order.
    """
    index: Dict[str, List[str]] = {}

    for _, row in merged_df.iterrows():
        table_id = row["table_id"]
        cats_raw = str(row.get("categories", "")).strip()

        if not cats_raw:
            index.setdefault("_uncategorised", []).append(table_id)
            continue

        for cat in [c.strip() for c in cats_raw.split(";") if c.strip()]:
            index.setdefault(cat, []).append(table_id)

    # Deduplicate within each bucket (preserve first-seen order).
    index = {k: list(dict.fromkeys(v)) for k, v in index.items()}

    sizes = [len(v) for v in index.values()]
    log.info(
        "Category index: %d categories | pool sizes min=%d, max=%d, total_unique=%d.",
        len(index), min(sizes), max(sizes),
        len({tid for ids in index.values() for tid in ids}),
    )
    return index


# ---------------------------------------------------------------------------
# 4. Slot allocation (Hamilton / largest-remainder method)
# ---------------------------------------------------------------------------

def _allocate_slots(
    category_index: Dict[str, List[str]],
    total: int,
) -> Dict[str, int]:
    """
    Proportionally divide total sampling slots across categories, guaranteeing
    at least 1 slot per category.

    Uses the Hamilton (largest-remainder) method so allocations sum to exactly
    total without systematic bias toward large or small categories. If there are
    more categories than slots, the largest buckets take priority.
    """
    categories  = list(category_index.keys())
    sizes       = [len(category_index[c]) for c in categories]
    grand_total = sum(sizes)

    if grand_total == 0:
        raise ValueError("Category index is empty -- nothing to sample.")

    raw     = [total * s / grand_total for s in sizes]
    floored = [max(1, math.floor(r)) for r in raw]

    # Edge case: more categories than slots.
    if sum(floored) > total:
        log.warning(
            "More categories (%d) than sample slots (%d). "
            "Largest %d categories each get 1 slot; others get 0.",
            len(categories), total, total,
        )
        order   = sorted(range(len(categories)), key=lambda i: sizes[i], reverse=True)
        floored = [0] * len(categories)
        for i in order[:total]:
            floored[i] = 1
        return dict(zip(categories, floored))

    # Distribute remaining slots to categories with the largest fractional remainders.
    remainder  = total - sum(floored)
    remainders = [r - math.floor(r) for r in raw]
    order      = sorted(range(len(categories)), key=lambda i: remainders[i], reverse=True)
    for i in order[:remainder]:
        floored[i] += 1

    assert sum(floored) == total, "Allocation arithmetic error -- this should never happen."
    return dict(zip(categories, floored))


# ---------------------------------------------------------------------------
# 5. Stratified sampling
# ---------------------------------------------------------------------------

def stratified_sample(
    category_index: Dict[str, List[str]],
    sample_size: int,
    seed: int,
) -> List[str]:
    """
    Draw sample_size unique table IDs using proportional stratified sampling.

    Phase 1 -- Proportional draw:
        Allocate slots across categories proportionally using Hamilton's method,
        then shuffle each bucket with a seeded RNG and take the first n_slots entries.

    Phase 2 -- Gap fill:
        Because a single table can belong to multiple categories, Phase 1 may
        produce duplicates that reduce the unique count below sample_size. Any
        shortfall is filled by sampling uniformly from the remaining pool.

    Both phases share the same seeded RNG instance, so the full procedure is
    reproducible given the same seed and input data.
    """
    rng = random.Random(seed)

    all_ids = list(dict.fromkeys(
        tid for ids in category_index.values() for tid in ids
    ))

    if len(all_ids) < sample_size:
        log.warning(
            "Total unique pool (%d tables) is smaller than the requested "
            "sample size (%d). Returning the entire pool.",
            len(all_ids), sample_size,
        )
        rng.shuffle(all_ids)
        return all_ids

    # Phase 1: proportional draw.
    allocation = _allocate_slots(category_index, sample_size)
    selected: set = set()

    for cat, n_slots in allocation.items():
        if n_slots == 0:
            continue
        bucket = list(category_index[cat])
        rng.shuffle(bucket)
        selected.update(bucket[:n_slots])

    n_phase1 = len(selected)
    log.info(
        "Phase 1: drew from %d categories -> %d unique tables "
        "(target %d; %d cross-category duplicates removed).",
        len(allocation), n_phase1, sample_size, max(0, sample_size - n_phase1),
    )

    # Phase 2: fill any gap.
    if len(selected) < sample_size:
        remaining = [tid for tid in all_ids if tid not in selected]
        rng.shuffle(remaining)
        gap = sample_size - len(selected)
        log.info(
            "Phase 2 gap-fill: adding %d tables from %d remaining in pool.",
            gap, len(remaining),
        )
        selected.update(remaining[:gap])

    result = list(selected)[:sample_size]
    log.info("Final sample: %d unique tables.", len(result))
    return result


# ---------------------------------------------------------------------------
# 6. Build output DataFrame
# ---------------------------------------------------------------------------

def build_output(
    merged_df: pd.DataFrame,
    sampled_ids: List[str],
    category_index: Dict[str, List[str]],
) -> pd.DataFrame:
    """
    Subset the merged DataFrame to the sampled table IDs and attach a
    category column listing every category the table's paper belongs to
    (sorted, comma-separated) so no rows are duplicated.
    """
    table_to_cats: Dict[str, List[str]] = {}
    for cat, ids in category_index.items():
        for tid in ids:
            table_to_cats.setdefault(tid, []).append(cat)
    cat_lookup = {
        tid: ", ".join(sorted(c for c in cats if c != "_uncategorised"))
        for tid, cats in table_to_cats.items()
    }

    subset = merged_df[merged_df["table_id"].isin(set(sampled_ids))].copy()

    before = len(subset)
    subset = subset.drop_duplicates(subset=["table_id"], keep="first")
    if len(subset) < before:
        log.warning(
            "Dropped %d duplicate rows after merge (kept first per table_id).",
            before - len(subset),
        )

    subset["category"] = subset["table_id"].map(cat_lookup).fillna("")

    # Build paper_url from paper_id so it's always populated, regardless of
    # whether the metadata TSV covers that paper's month.
    subset["paper_url"] = ARXIV_HTML_BASE + subset["paper_id"].astype(str)

    # Build table_url by appending the in-page anchor from html_id.
    # Falls back to paper_url when html_id is absent or blank.
    if "html_id" in subset.columns:
        subset["table_url"] = subset.apply(
            lambda row: (
                row["paper_url"] + "#" + str(row["html_id"])
                if pd.notna(row.get("html_id")) and str(row.get("html_id", "")).strip()
                else row["paper_url"]
            ),
            axis=1,
        )
    else:
        subset["table_url"] = subset["paper_url"]

    for col in OUTPUT_COLUMNS:
        if col not in subset.columns:
            log.warning("Column '%s' missing from merged data; filling with empty string.", col)
            subset[col] = ""

    return subset[OUTPUT_COLUMNS].reset_index(drop=True)


# ---------------------------------------------------------------------------
# 7. Write output
# ---------------------------------------------------------------------------

def write_output(output_df: pd.DataFrame, path: Path) -> None:
    """Write output_df as a UTF-8 TSV, creating parent directories as needed."""
    path.parent.mkdir(parents=True, exist_ok=True)
    output_df.to_csv(path, sep="\t", index=False, encoding="utf-8")
    log.info("Output written: %s (%d rows x %d columns).", path, *output_df.shape)


# ---------------------------------------------------------------------------
# 8. Main pipeline
# ---------------------------------------------------------------------------

def main(
    output_dir: Path = OUTPUT_DIR,
    output_path: Path = OUTPUT_TSV,
    sample_size: int  = SAMPLE_SIZE,
    seed: int         = RANDOM_SEED,
) -> pd.DataFrame:
    """
    End-to-end pipeline:

      load tables (JSONL) + load metadata (TSV)
        -> merge on paper_id
        -> build category index
        -> stratified sample
        -> build output DataFrame
        -> write TSV

    Returns the output DataFrame so callers can inspect it without re-reading
    the file (useful in notebooks or when importing this module).
    """
    log.info(
        "=== table_sampler.py  output_dir=%s  seed=%d  target=%d tables ===",
        output_dir, seed, sample_size,
    )

    tables_df      = load_tables(output_dir)
    metadata_df    = load_metadata(output_dir)
    merged_df      = merge_dataframes(tables_df, metadata_df)
    category_index = build_category_index(merged_df)
    sampled_ids    = stratified_sample(category_index, sample_size, seed)
    output_df      = build_output(merged_df, sampled_ids, category_index)

    write_output(output_df, output_path)
    log.info("=== Done. %d unique tables written to %s. ===", len(output_df), output_path)
    return output_df


# ---------------------------------------------------------------------------
# Argument parsing
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(
        description="Draw a stratified random sample of 200 tables from scraper output.",
        epilog="Example:\n  python table_sampler.py arxiv_data sampled_tables.tsv",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    ap.add_argument(
        "output_dir", nargs="?", default=str(OUTPUT_DIR),
        help=f"Root directory of scraper output (default: {OUTPUT_DIR})",
    )
    ap.add_argument(
        "output_tsv", nargs="?", default=str(OUTPUT_TSV),
        help=f"Path for the output TSV (default: {OUTPUT_TSV})",
    )
    return ap.parse_args()


if __name__ == "__main__":
    args = parse_args()
    main(output_dir=Path(args.output_dir), output_path=Path(args.output_tsv))