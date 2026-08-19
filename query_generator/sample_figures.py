import pandas as pd
import random
import os
import argparse

random.seed(42)

# -------------------------------------------------
# Parse command-line arguments
# -------------------------------------------------

parser = argparse.ArgumentParser(description="Sample 200 figures across arXiv categories.")
parser.add_argument("year", help="Two-digit year (e.g. 24 for 2024)")
parser.add_argument("month", help="Two-digit month (e.g. 10 for October)")
parser.add_argument("--data_dir", default="arxiv_data", help="Root directory of scraper output (default: arxiv_data)")
parser.add_argument("--output_tsv", default="200_sampled_figures.tsv", help="Output file path (default: 200_sampled_figures.tsv)")
args = parser.parse_args()

# -------------------------------------------------
# Read TSV files
# -------------------------------------------------

captions = pd.read_csv(
    f"{args.data_dir}/captions/{args.year}_{args.month}.tsv",
    sep="\t"
)

references = pd.read_csv(
    f"{args.data_dir}/references/ref_{args.year}_{args.month}.tsv",
    sep="\t"
)

metadata = pd.read_csv(
    f"{args.data_dir}/metadata_{args.year}_{args.month}.tsv",
    sep="\t"
)

# -------------------------------------------------
# Remove duplicate metadata rows
# -------------------------------------------------

metadata = metadata.drop_duplicates(subset="paper_id")

# -------------------------------------------------
# Combine all references for each figure
# -------------------------------------------------

references_grouped = (
    references
    .groupby(["paper_id", "figure_id"])["reference_text"]
    .apply(lambda x: " | ".join(x.dropna().unique()))
    .reset_index()
)

# -------------------------------------------------
# Merge captions + references
# -------------------------------------------------

merged = captions.merge(
    references_grouped,
    on=["paper_id", "figure_id"],
    how="left"
)

# -------------------------------------------------
# Merge metadata
# -------------------------------------------------

merged = merged.merge(
    metadata,
    on="paper_id",
    how="left"
)

print("Merged rows:", len(merged))

# -------------------------------------------------
# Build category dictionary
# -------------------------------------------------

category_dict = {}

for _, row in merged.iterrows():

    if pd.isna(row["categories"]):
        continue

    categories = [
        c.strip()
        for c in str(row["categories"]).split(";")
    ]

    figure = {

        "paper_id": row["paper_id"],
        "figure_id": row["figure_id"],

        "paper_url": row["url"],

        # will be filled later
        "figure_url": "",

        "caption": row["caption"],

        "reference_text": row["reference_text"],

        "title": row["title"],

        "abstract": row["abstract"],

        "all_categories": row["categories"]
    }

    for cat in categories:

        category_dict.setdefault(cat, []).append(figure)

print("Categories:", len(category_dict))

# -------------------------------------------------
# Pick one random figure from every category
# -------------------------------------------------

sampled = []

for category, figures in category_dict.items():

    figure = random.choice(figures).copy()

    figure["sampling_category"] = category

    sampled.append(figure)

print("Before dedup:", len(sampled))

# -------------------------------------------------
# Remove duplicate figures
# -------------------------------------------------

unique = []

seen = set()

for figure in sampled:

    key = (
        figure["paper_id"],
        figure["figure_id"]
    )

    if key not in seen:

        seen.add(key)

        unique.append(figure)

print("After dedup:", len(unique))

# -------------------------------------------------
# If we have fewer than 200, randomly fill the rest
# -------------------------------------------------

needed = 200 - len(unique)

if needed > 0:

    # Build a set of figures we've already chosen
    chosen = {
        (fig["paper_id"], fig["figure_id"])
        for fig in unique
    }

    # Remaining figures that weren't already selected
    remaining = []

    for _, row in merged.iterrows():

        if pd.isna(row["categories"]):
            continue

        key = (row["paper_id"], row["figure_id"])

        if key in chosen:
            continue

        remaining.append({

            "paper_id": row["paper_id"],
            "figure_id": row["figure_id"],

            "paper_url": row["url"],

            # Filled later by helper script
            "figure_url": "",

            "caption": row["caption"],

            "reference_text": row["reference_text"],

            "title": row["title"],

            "abstract": row["abstract"],

            "category": row["categories"]
        })

    print("Remaining figures:", len(remaining))

    extra = random.sample(
        remaining,
        min(needed, len(remaining))
    )

    unique.extend(extra)

# -------------------------------------------------
# If we somehow have more than 200, trim back
# -------------------------------------------------

if len(unique) > 200:

    unique = random.sample(unique, 200)

print("Final sample size:", len(unique))

# -------------------------------------------------
# Convert to DataFrame
# -------------------------------------------------

sample_df = pd.DataFrame(unique)

# -------------------------------------------------
# Rename columns
# -------------------------------------------------

sample_df.rename(
    columns={

        "paper_id": "Paper ID",

        "figure_id": "Figure ID",

        "all_categories": "Categories",

        "sampling_category": "Sampling Category",

        "paper_url": "Paper URL",

        "figure_url": "Figure URL",

        "caption": "Caption",

        "reference_text": "Reference in Text",

        "title": "Paper Title",

        "abstract": "Paper Abstract"

    },
    inplace=True
)

# -------------------------------------------------
# Reorder columns
# -------------------------------------------------

sample_df = sample_df[

    [

        "Paper ID",

        "Figure ID",

        "Sampling Category",

        "Categories",

        "Paper URL",

        "Figure URL",

        "Caption",

        "Reference in Text",

        "Paper Title",

        "Paper Abstract"

    ]

]

# -------------------------------------------------
# Clean text
# -------------------------------------------------

text_columns = [

    "Caption",

    "Reference in Text",

    "Paper Title",

    "Paper Abstract"

]

for col in text_columns:

    sample_df[col] = (

        sample_df[col]

        .fillna("")

        .astype(str)

        .str.replace(r"\s+", " ", regex=True)

        .str.strip()

    )

# -------------------------------------------------
# Empty AI query columns
# -------------------------------------------------

sample_df["Query 1"] = ""

sample_df["Query 2"] = ""

sample_df["Query 3"] = ""

# -------------------------------------------------
# Save
# -------------------------------------------------

sample_df.to_csv(
    args.output_tsv,
    sep="\t",
    index=False
)

print()

print("Saved 200_sampled_figures.tsv")

print("Rows:", len(sample_df))

print()

print(sample_df.head())

