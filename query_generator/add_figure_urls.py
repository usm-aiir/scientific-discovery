import pandas as pd
import requests
from bs4 import BeautifulSoup
import time
import re
from urllib.parse import urljoin

# -------------------------------------------------
# Load sampled dataset
# -------------------------------------------------

df = pd.read_csv("200_sampled_figures.tsv", sep="\t")

# Ensure column exists
if "Figure URL" not in df.columns:
    df["Figure URL"] = df["Figure URL"].fillna("").astype(str)

# -------------------------------------------------
# Helper: resolve image URL (same logic as scraper)
# -------------------------------------------------

def resolve_image_url(src, paper_url):
    if not src:
        return None
    return urljoin(paper_url, src)

# -------------------------------------------------
# Cache so we don't download same paper twice
# -------------------------------------------------

paper_cache = {}

def extract_figures_from_paper(paper_url):
    """Returns list of (figure_id, image_url)"""

    if paper_url in paper_cache:
        return paper_cache[paper_url]

    try:
        html = requests.get(paper_url, timeout=20).text
        soup = BeautifulSoup(html, "html.parser")

        figures = []

        for fig in soup.find_all("figure"):

            img = fig.find("img")
            if not img:
                continue

            src = resolve_image_url(img.get("src"), paper_url)

            # Try to extract figure number/id from caption
            caption = fig.get_text(" ", strip=True)

            match = re.search(r"Figure\s*(\d+)", caption)

            if match:
                fig_id = int(match.group(1))
                figures.append((fig_id, src))

        paper_cache[paper_url] = figures
        return figures

    except Exception as e:
        print(f"Error fetching {paper_url}: {e}")
        return []

# -------------------------------------------------
# Fill Figure URLs
# -------------------------------------------------

for i, row in df.iterrows():

    if pd.notna(row["Figure URL"]) and row["Figure URL"] != "":
        continue

    paper_url = row["Paper URL"]
    figure_id = int(row["Figure ID"])

    figures = extract_figures_from_paper(paper_url)

    # find matching figure
    for fid, url in figures:
        if fid == figure_id:
            df.at[i, "Figure URL"] = url
            break

    time.sleep(0.2)  # be polite to server

# -------------------------------------------------
# Save output
# -------------------------------------------------

df.to_csv("200_sampled_figures_with_urls.tsv", sep="\t", index=False)

print("Done!")
print(df.head())

#test
print((df["Figure URL"] != "").sum())