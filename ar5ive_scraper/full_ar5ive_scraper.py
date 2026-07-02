"""
arxiv_scraper.py
================
Scrapes arXiv HTML pages for paper metadata, figures, captions, and
in-text figure references, then writes everything to a structured
file hierarchy.

Output layout
-------------
<output_dir>/
├── figures/
│   └── <year>/
│       └── <month>/
│           └── <paper_id>/
│               ├── 1.png
│               ├── 2.png
│               └── ...
├── captions/
│   └── <year>_<month>.tsv        columns: paper_id, figure_id, caption
├── references/
│   └── ref_<year>_<month>.tsv    columns: paper_id, figure_id, reference_text
└── metadata.tsv                  columns: url, paper_id, title, abstract, categories
"""

from __future__ import annotations
from urllib.parse import urljoin
import csv
import logging
import os
import re
import time
from pathlib import Path
from typing import Optional

import requests
from bs4 import BeautifulSoup, NavigableString

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

# We may switch to this in case formulas are not extracted correctly
ARXIV_HTML_BASE = "https://ar5iv.labs.arxiv.org/html/"

Old_HTML_BASE = "https://arxiv.org/html/"

# How long to wait between requests so we don't hammer the server
REQUEST_DELAY_SECONDS = 1.0

# Retry settings for transient network errors
MAX_RETRIES = 3
RETRY_BACKOFF_SECONDS = 5.0

# Set up a module-level logger; callers can configure handlers as needed
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Networking helpers
# ---------------------------------------------------------------------------

def fetch_soup(url: str) -> Optional[BeautifulSoup]:
    """
    Download *url* and return a BeautifulSoup parse tree.

    Retries up to MAX_RETRIES times on transient HTTP errors (5xx) or
    connection problems.  Returns None if every attempt fails.
    """
    for attempt in range(1, MAX_RETRIES + 1):
        try:
            response = requests.get(url, timeout=30)
            response.raise_for_status()
            return BeautifulSoup(response.text, "html.parser")

        except requests.exceptions.HTTPError as exc:
            status = exc.response.status_code if exc.response else "?"
            if status == 404:
                log.debug("404 – paper does not exist: %s", url)
                return None          # not a transient error; stop retrying
            log.warning("HTTP %s on attempt %d/%d for %s", status, attempt, MAX_RETRIES, url)

        except requests.exceptions.RequestException as exc:
            log.warning("Request error on attempt %d/%d for %s: %s", attempt, MAX_RETRIES, url, exc)

        if attempt < MAX_RETRIES:
            time.sleep(RETRY_BACKOFF_SECONDS)

    log.error("All %d attempts failed for %s", MAX_RETRIES, url)
    return None


def download_image(url: str, dest_path: Path) -> bool:
    """
    Download a binary resource (image) from *url* and save it to *dest_path*.

    Returns True on success, False on failure.
    """
    dest_path.parent.mkdir(parents=True, exist_ok=True)
    try:
        response = requests.get(url, timeout=30, stream=True)
        response.raise_for_status()
        with open(dest_path, "wb") as fh:
            for chunk in response.iter_content(chunk_size=8192):
                fh.write(chunk)
        return True
    except requests.exceptions.RequestException as exc:
        log.warning("Could not download image %s: %s", url, exc)
        return False


# ---------------------------------------------------------------------------
# Parsing helpers
# ---------------------------------------------------------------------------

def parse_title_abstract(soup: BeautifulSoup) -> tuple[str, str]:
    """
    Extract the paper title and abstract text from a parsed arXiv HTML page.

    Returns a ``(title, abstract)`` tuple.  Either value is an empty string
    when the expected element cannot be found.
    """
    title_tag = soup.find("h1", class_="ltx_title_document")
    title = ""
    if title_tag:
        title = ' '.join(extract_text_with_math(title_tag).split())

    abstract_div = soup.find("div", class_="ltx_abstract")
    abstract = ""
    if abstract_div:
        abstract_p = abstract_div.find("p", class_="ltx_p")
        if abstract_p:
            abstract = ' '.join(extract_text_with_math(abstract_p).split())

    return title, abstract


def _resolve_image_url(raw_src: str, paper_url: str) -> str:
    """
     Return an absolute URL for *raw_src*, resolving relative paths against *paper_url*.
     Works correctly for ar5iv's absolute paths (starting with '/html/...').
    """
    return urljoin(paper_url, raw_src)


# --- Revised caption extraction ---
def _outer_caption(fig_tag: BeautifulSoup) -> str:
    for tag in fig_tag.children:
        if tag.name == "figcaption" and "ltx_caption" in tag.get("class", []):
            raw = extract_text_with_math(tag)
            return ' '.join(raw.split())
    return ""

# --- Math‑aware text extraction ---
def extract_text_with_math(element) -> str:
    if isinstance(element, NavigableString):
        return str(element)
    if element.name == 'math':
        alt = element.get('alttext', '')
        return f'${alt}$' if alt else element.get_text()
    return ''.join(extract_text_with_math(child) for child in element.children)


def parse_figures(soup: BeautifulSoup, paper_url: str) -> list[dict]:
    rows: list[dict] = []
    sequential_idx = 0

    top_level_figures = [
        fig for fig in soup.find_all("figure", class_="ltx_figure")
        if "ltx_figure_panel" not in fig.get("class", [])
    ]

    for fig in top_level_figures:
        sequential_idx += 1
        outer_caption = _outer_caption(fig)

        num_match = re.search(r"\bFigure\s+(\d+)", outer_caption, re.IGNORECASE)
        fig_id = int(num_match.group(1)) if num_match else sequential_idx

        panels = fig.find_all("figure", class_="ltx_figure_panel")

        if panels:
            # First, try to extract individual sub‑captions from each panel
            panel_captions = []
            for panel in panels:
                sub = _outer_caption(panel)
                panel_captions.append(sub)

            # If all panel captions are empty, fall back to splitting the outer caption
            if all(c == "" for c in panel_captions):
                # Parse the outer caption for "(a) ... (b) ..."
                sub_parts = split_subcaptions(outer_caption)
                # If we found sub‑labels, use them; otherwise, fall back to simple letters
                if sub_parts:
                    # Ensure we have at least as many parts as panels; if not, pad with empty texts
                    while len(sub_parts) < len(panels):
                        sub_parts.append(("", ""))
                    # Use the sub‑captions from the split
                    panel_captions = [text for _, text in sub_parts[:len(panels)]]
                    # Also set sub_ids from the letters
                    panel_sub_ids = [letter for letter, _ in sub_parts[:len(panels)]]
                    # If some letters are missing, use order-based letters
                    for i, (label, text) in enumerate(sub_parts):
                        if not label:
                            panel_sub_ids[i] = chr(ord('a') + i)
                else:
                    # No labels found; just use order-based letters and no sub‑captions
                    panel_sub_ids = [chr(ord('a') + i) for i in range(len(panels))]
                    panel_captions = ["" for _ in panels]
            else:
                # We have individual captions; extract sub_ids from each
                panel_sub_ids = []
                for sub in panel_captions:
                    sub_match = re.search(r"\(([a-z])\)", sub, re.IGNORECASE)
                    panel_sub_ids.append(sub_match.group(1).lower() if sub_match else None)

            # Now iterate over panels and build rows
            for idx, panel in enumerate(panels):
                img_tag = panel.find("img")
                img_src = (
                    _resolve_image_url(img_tag["src"], paper_url)
                    if img_tag and img_tag.get("src")
                    else None
                )
                sub_caption = panel_captions[idx] if idx < len(panel_captions) else ""
                sub_id = panel_sub_ids[idx] if idx < len(panel_sub_ids) else None

                rows.append({
                    "figure_id": fig_id,
                    "sub_id": sub_id,
                    "source": img_src,
                    "caption": outer_caption,
                    "sub_caption": sub_caption,
                })
        else:
            # Simple figure
            img_tag = fig.find("img")
            img_src = (
                _resolve_image_url(img_tag["src"], paper_url)
                if img_tag and img_tag.get("src")
                else None
            )
            rows.append({
                "figure_id": fig_id,
                "sub_id": None,
                "source": img_src,
                "caption": outer_caption,
                "sub_caption": None,
            })

    return rows
# --- Revised reference extraction ---
def parse_figure_references(soup: BeautifulSoup) -> dict[str, list[str]]:
    figure_numbers: list[str] = []
    references: dict[str, list[str]] = {}
    top_level_figures = [
        fig for fig in soup.find_all("figure", class_="ltx_figure")
        if "ltx_figure_panel" not in fig.get("class", [])
    ]
    for seq_idx, fig in enumerate(top_level_figures, start=1):
        caption = _outer_caption(fig)
        match = re.search(r"\bFigure\s+(\d+)", caption, re.IGNORECASE)
        fig_num = match.group(1) if match else str(seq_idx)
        figure_numbers.append(fig_num)
        references[f"Figure {fig_num}"] = []
    all_paragraphs = soup.find_all("p", class_="ltx_p")
    for fig_num in figure_numbers:
        key = f"Figure {fig_num}"
        pattern = re.compile(
            rf"\bfig(?:ure)?\.?\s*{re.escape(fig_num)}(?:[a-z]|\([a-z]\)|-[a-z])?\b",
            re.IGNORECASE,
        )
        for para in all_paragraphs:
            if para.find_parent("figure") is not None:
                continue
            text = extract_text_with_math(para)
            text = ' '.join(text.split())  # normalise whitespace
            if pattern.search(text) and text not in references[key]:
                references[key].append(text)
    return references

import xml.etree.ElementTree as ET


def parse_categories(soup: BeautifulSoup, paper_id) -> list[str]:
    """
    Extract arXiv subject categories. Extracts the paper ID from the HTML header
    and cross-references the official arXiv API since categories are missing
    from the raw HTML text body.
    """
    categories = []

    try:
        log.info("Fetching categories via arXiv API for ID: %s", paper_id)
        api_url = f"https://export.arxiv.org/api/query?id_list={paper_id}"
        api_resp = requests.get(api_url, timeout=10)

        if api_resp.status_code == 200:
            root = ET.fromstring(api_resp.content)
            # Parse all <category term="..."/> tags from the Atom XML feed
            for category_tag in root.findall(".//{http://www.w3.org/2005/Atom}category"):
                term = category_tag.get("term")
                # Keeps valid primary/secondary categories (e.g., cs.HC, stat.ML)
                if term and "." in term and term not in categories:
                    categories.append(term)
    except Exception as e:
        log.warning("arXiv API metadata fallback query failed: %s", e)

    return categories

# ---------------------------------------------------------------------------
# Output helpers
# ---------------------------------------------------------------------------

def _tsv_writer(path: Path, fieldnames: list[str]) -> tuple:
    """
    Open *path* for appending in TSV mode and return ``(file_handle, writer)``.

    Creates parent directories and writes the header row only when the file
    does not yet exist.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    is_new = not path.exists()
    fh = open(path, "a", newline="", encoding="utf-8")
    writer = csv.DictWriter(fh, fieldnames=fieldnames, delimiter="\t")
    if is_new:
        writer.writeheader()
    return fh, writer


# ---------------------------------------------------------------------------
# Per-paper processing
# ---------------------------------------------------------------------------

def process_paper(
    paper_id: str,
    year: str,
    month: str,
    output_dir: Path,
) -> bool:
    """
    Scrape a single arXiv paper and persist all extracted data.

    Parameters
    ----------
    paper_id:
        The five-digit arXiv sequence number, e.g. ``"12325"``.
        Combined with *year* and *month* it forms the full ID ``YYMM.NNNNN``
        (e.g. ``"2510.12325"``).
    year:
        Two-digit year string, e.g. ``"25"``.
    month:
        Two-digit month string, e.g. ``"10"``.
    output_dir:
        Root directory for all output files.

    Returns
    -------
    bool
        True if the paper was found and processed, False otherwise.
    """
    # arXiv IDs follow the format YYMM.NNNNN  (e.g. 2510.12325)
    full_id = f"{year}{month}.{paper_id}"
    paper_url = ARXIV_HTML_BASE + full_id
    log.info("Processing %s …", paper_url)

    soup = fetch_soup(paper_url)

    # print(soup)
    # input("wait")
    if soup is None:
        return False

    # ------------------------------------------------------------------ parse
    title, abstract = parse_title_abstract(soup)
    figures = parse_figures(soup, paper_url)
    fig_references = parse_figure_references(soup)
    categories = parse_categories(soup, full_id)

    # --------------------------------------------------- save figure images
    fig_dir = output_dir / "figures" / year / month / full_id
    fig_dir.mkdir(parents=True, exist_ok=True)

    for row in figures:
        if not row["source"]:
            continue
        # Use "3a.png", "3b.png" for panels; "3.png" for simple figures
        stem = str(row["figure_id"]) + (row["sub_id"] or "")
        dest = fig_dir / f"{stem}.png"
        download_image(row["source"], dest)

    # ----------------------------------------------- captions TSV (append)
    # Columns: paper_id, figure_id, sub_id, caption, sub_caption
    # sub_id and sub_caption are empty strings for simple (non-panel) figures.
    captions_path = output_dir / "captions" / f"{year}_{month}.tsv"
    cap_fh, cap_writer = _tsv_writer(
        captions_path,
        fieldnames=["paper_id", "figure_id", "sub_id", "caption", "sub_caption"],
    )
    try:
        for row in figures:
            cap_writer.writerow(
                {
                    "paper_id": full_id,
                    "figure_id": row["figure_id"],
                    "sub_id": row["sub_id"] or "",
                    "caption": row["caption"],
                    "sub_caption": row["sub_caption"] or "",
                }
            )
    finally:
        cap_fh.close()

    # ------------------------------------------ references TSV (append)
    refs_path = output_dir / "references" / f"ref_{year}_{month}.tsv"
    ref_fh, ref_writer = _tsv_writer(
        refs_path,
        fieldnames=["paper_id", "figure_id", "reference_text"],
    )
    try:
        for label, paragraphs in fig_references.items():
            # Resolve the label back to a plain integer figure_id
            id_match = re.search(r"\d+", label)
            fig_id = int(id_match.group()) if id_match else label
            for para_text in paragraphs:
                ref_writer.writerow(
                    {
                        "paper_id": full_id,
                        "figure_id": fig_id,
                        "reference_text": para_text,
                    }
                )
    finally:
        ref_fh.close()

    # ----------------------------------------------- metadata TSV (append)
    meta_path = output_dir / f"metadata_{year}_{month}.tsv"
    meta_fh, meta_writer = _tsv_writer(
        meta_path,
        fieldnames=["url", "paper_id", "title", "abstract", "categories"],
    )
    try:
        meta_writer.writerow(
            {
                "url": paper_url,
                "paper_id": full_id,
                "title": title,
                "abstract": abstract,
                "categories": "; ".join(categories),
            }
        )
    finally:
        meta_fh.close()

    log.info(
        "  ✓ %s – %d figure row(s), %d category/ies",
        full_id,
        len(figures),
        len(categories),
    )
    return True


# ---------------------------------------------------------------------------
# Batch scraping
# ---------------------------------------------------------------------------

def scrape_month(
    year: str,
    month: str,
    output_dir: Path,
    max_papers: Optional[int] = None,
    start_id: int = 1,
) -> None:
    """
    Iterate over arXiv paper IDs for a given *year* / *month* and scrape each
    one until a 404 is returned (indicating no more papers exist for that
    month/year combination).

    Parameters
    ----------
    year:
        Two-digit year, e.g. ``"25"``.
    month:
        Two-digit month, e.g. ``"10"``.
    output_dir:
        Root directory for all output files.
    max_papers:
        Stop after processing this many papers (useful for testing). Pass
        None to scrape until arXiv returns a 404.
    start_id:
        The numeric ID to begin from (default 1). Useful for resuming an
        interrupted run.
    """
    log.info("=== Scraping %s/%s (starting at %s.%05d) ===", year, month, year + month, start_id)
    processed = 0

    for numeric_id in range(start_id, 100_000):
        paper_id = f"{numeric_id:05d}"
        found = process_paper(paper_id, year, month, output_dir)

        if not found:
            log.info("No paper found for id %s – assuming end of %s/%s.", paper_id, year, month)
            continue

        processed += 1
        if max_papers is not None and processed >= max_papers:
            log.info("Reached max_papers limit (%d). Stopping.", max_papers)
            break

        time.sleep(REQUEST_DELAY_SECONDS)

    log.info("Done. Processed %d paper(s) for %s/%s.", processed, year, month)


def scrape_range(
    years: list[str],
    months: list[str],
    output_dir: Path,
    max_papers_per_month: Optional[int] = None,
) -> None:
    """
    Scrape every (year, month) combination in *years* × *months*.

    Parameters
    ----------
    years:
        List of two-digit year strings, e.g. ``["24", "25"]``.
    months:
        List of two-digit month strings, e.g. ``["01", "02", …, "12"]``.
    output_dir:
        Root directory for all output files.
    max_papers_per_month:
        Cap on papers scraped per month (useful for testing).
    """
    for year in years:
        for month in months:
            scrape_month(year, month, output_dir, max_papers=max_papers_per_month)

from typing import List, Tuple

def split_subcaptions(caption: str) -> List[Tuple[str, str]]:
    """
    Split a compound figure caption like
    "Figure 1: (a) First panel (b) Second panel."
    into a list of (label, text) for each sub‑panel.

    Returns a list of (letter, subcaption_text) pairs.
    """
    # Pattern: (a) ... followed by another (letter) or end of string
    pattern = re.compile(r'\(([a-z])\)\s*([^)]*?)(?=\s*\([a-z]\)|$)')
    matches = pattern.findall(caption)
    # Clean up: remove leading/trailing whitespace from each text
    return [(letter, text.strip()) for letter, text in matches]
# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    # ------------------------------------------------------------------
    # Quick single-paper test
    # ------------------------------------------------------------------
    OUTPUT_ROOT = Path("arxiv_data")

    ############## SAMPLE 1
    # test_paper_id = "00003"   # becomes 2410.12325
    # test_year = "24"
    # test_month = "10"
    # #
    # process_paper(test_paper_id, test_year, test_month, OUTPUT_ROOT)

    # SAMPLE 2
    # ------------------------------------------------------------------
    # Uncomment to run a full month scrape (e.g. October 2024)
    # ------------------------------------------------------------------
    scrape_month("24", "10", OUTPUT_ROOT)


    # SAMPLE 3
    # ------------------------------------------------------------------
    # Uncomment to scrape multiple years/months (limited to 5 papers each
    # here for safety during development)
    # ------------------------------------------------------------------
    # scrape_range(
    #     years=["24", "25"],
    #     months=["01", "02", "03", "04", "05", "06", "07", "08", "09", "10", "11", "12"],
    #     output_dir=OUTPUT_ROOT,
    #     max_papers_per_month=None,
    # )