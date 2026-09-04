"""
sample_figures.py
=================
Samples up to 200 figures from a scraped arXiv month, stratified by
subject category, and writes the result to a TSV ready for query generation.
 
Strategy
--------
1. Load captions, references, and metadata for the given year/month.
2. Group figures by their arXiv subject categories.
3. Pick one random figure from each category (ensures diversity).
4. Deduplicate, then fill any remaining slots (up to 200) with random
   figures not already selected.
5. Write the output TSV with empty Query 1/2/3 columns for later annotation.
 
Output TSV columns
------------------
Paper ID, Figure ID, Sampling Category, Categories, Paper URL, Figure URL,
Caption, Reference in Text, Paper Title, Paper Abstract, Query 1, Query 2, Query 3
 
Usage
-----
    python sample_figures.py <year> <month> [--data_dir DIR] [--output_tsv FILE]
 
Example
-------
    python sample_figures.py 24 10
    python sample_figures.py 24 10 --output_tsv results/sampled.tsv
"""
 
import argparse
import random
 
import pandas as pd
 
# Fixed seed for reproducibility
random.seed(42)
 
TARGET_SAMPLE_SIZE = 200
 
OUTPUT_COLUMNS = [
    "Paper ID",
    "Figure ID",
    "Sampling Category",
    "Categories",
    "Paper URL",
    "Figure URL",
    "Caption",
    "Reference in Text",
    "Paper Title",
    "Paper Abstract",
    "Query 1",
    "Query 2",
    "Query 3",
]
 
TEXT_COLUMNS = ["Caption", "Reference in Text", "Paper Title", "Paper Abstract"]
 
 
# ---------------------------------------------------------------------------
# Argument parsing
# ---------------------------------------------------------------------------
 
def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Sample figures across arXiv categories for query generation.",
        epilog="Example: python sample_figures.py 24 10",
    )
    parser.add_argument("year",  help="Two-digit year  (e.g. 24 for 2024)")
    parser.add_argument("month", help="Two-digit month (e.g. 10 for October)")
    parser.add_argument(
        "--data_dir", default="arxiv_data",
        help="Root directory of scraper output (default: arxiv_data)",
    )
    parser.add_argument(
        "--output_tsv", default="200_sampled_figures.tsv",
        help="Output file path (default: 200_sampled_figures.tsv)",
    )
    return parser.parse_args()
 
 
# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------
 
def load_data(data_dir: str, year: str, month: str) -> pd.DataFrame:
    """
    Load captions, references, and metadata for the given year/month,
    merge them into a single DataFrame, and return it.
    """
    captions = pd.read_csv(
        f"{data_dir}/captions/{year}_{month}.tsv", sep="\t"
    )
    references = pd.read_csv(
        f"{data_dir}/references/ref_{year}_{month}.tsv", sep="\t"
    )
    metadata = pd.read_csv(
        f"{data_dir}/metadata_{year}_{month}.tsv", sep="\t"
    )
 
    # One metadata row per paper
    metadata = metadata.drop_duplicates(subset="paper_id")
 
    # Collapse multiple reference paragraphs for the same figure into one string
    references_grouped = (
        references
        .groupby(["paper_id", "figure_id"])["reference_text"]
        .apply(lambda x: " | ".join(x.dropna().unique()))
        .reset_index()
    )
 
    merged = (
        captions
        .merge(references_grouped, on=["paper_id", "figure_id"], how="left")
        .merge(metadata, on="paper_id", how="left")
    )
 
    print(f"Loaded {len(merged):,} figure rows across "
          f"{merged['paper_id'].nunique():,} papers")
    return merged
 
 
# ---------------------------------------------------------------------------
# Sampling
# ---------------------------------------------------------------------------
 
def _figure_dict(row: pd.Series, sampling_category: str = "") -> dict:
    """Build a figure record dict from a DataFrame row."""
    return {
        "paper_id":          row["paper_id"],
        "figure_id":         row["figure_id"],
        "paper_url":         row["url"],
        "figure_url":        "",        # filled later by a helper script
        "caption":           row["caption"],
        "reference_text":    row.get("reference_text", ""),
        "title":             row["title"],
        "abstract":          row["abstract"],
        "all_categories":    row["categories"],
        "sampling_category": sampling_category,
    }
 
 
def sample_figures(merged: pd.DataFrame) -> list[dict]:
    """
    Sample up to TARGET_SAMPLE_SIZE figures, stratified by subject category.
 
    Step 1 — one random figure per category (ensures breadth).
    Step 2 — deduplicate (a figure may appear in multiple categories).
    Step 3 — fill remaining slots with random figures not yet selected.
    Step 4 — trim to exactly TARGET_SAMPLE_SIZE if we somehow exceed it.
    """
    # Step 1: one figure per category
    category_dict: dict[str, list] = {}
    for _, row in merged.iterrows():
        if pd.isna(row["categories"]):
            continue
        for cat in [c.strip() for c in str(row["categories"]).split(";")]:
            category_dict.setdefault(cat, []).append(row)
 
    print(f"Found {len(category_dict)} distinct categories")
 
    sampled = []
    for category, rows in category_dict.items():
        row = random.choice(rows)
        sampled.append(_figure_dict(row, sampling_category=category))
 
    print(f"After category sampling: {len(sampled)} figures")
 
    # Step 2: deduplicate
    seen: set[tuple] = set()
    unique: list[dict] = []
    for fig in sampled:
        key = (fig["paper_id"], fig["figure_id"])
        if key not in seen:
            seen.add(key)
            unique.append(fig)
 
    print(f"After deduplication: {len(unique)} figures")
 
    # Step 3: fill remaining slots
    needed = TARGET_SAMPLE_SIZE - len(unique)
    if needed > 0:
        remaining = [
            _figure_dict(row)
            for _, row in merged.iterrows()
            if not pd.isna(row["categories"])
            and (row["paper_id"], row["figure_id"]) not in seen
        ]
        print(f"Filling {needed} remaining slots from {len(remaining):,} candidates")
        extra = random.sample(remaining, min(needed, len(remaining)))
        unique.extend(extra)
 
    # Step 4: trim
    if len(unique) > TARGET_SAMPLE_SIZE:
        unique = random.sample(unique, TARGET_SAMPLE_SIZE)
 
    print(f"Final sample size: {len(unique)}")
    return unique
 
 
# ---------------------------------------------------------------------------
# Output
# ---------------------------------------------------------------------------
 
def build_output_df(figures: list[dict]) -> pd.DataFrame:
    """
    Convert the list of figure dicts to a clean DataFrame with the expected
    column names, cleaned text fields, and empty query columns.
    """
    df = pd.DataFrame(figures)
 
    df.rename(columns={
        "paper_id":          "Paper ID",
        "figure_id":         "Figure ID",
        "all_categories":    "Categories",
        "sampling_category": "Sampling Category",
        "paper_url":         "Paper URL",
        "figure_url":        "Figure URL",
        "caption":           "Caption",
        "reference_text":    "Reference in Text",
        "title":             "Paper Title",
        "abstract":          "Paper Abstract",
    }, inplace=True)
 
    # Add empty query columns for later annotation
    df["Query 1"] = ""
    df["Query 2"] = ""
    df["Query 3"] = ""
 
    # Normalize whitespace in all text fields
    for col in TEXT_COLUMNS:
        df[col] = (
            df[col]
            .fillna("")
            .astype(str)
            .str.replace(r"\s+", " ", regex=True)
            .str.strip()
        )
 
    return df[OUTPUT_COLUMNS]
 
 
# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
 
def main() -> None:
    args = parse_args()
 
    merged  = load_data(args.data_dir, args.year, args.month)
    figures = sample_figures(merged)
    df      = build_output_df(figures)
 
    df.to_csv(args.output_tsv, sep="\t", index=False)
 
    print(f"\nSaved {len(df)} figures to {args.output_tsv}")
    print(df.head())
 
 
if __name__ == "__main__":
    main()
