
"""
arxiv_scraper.py
================
Scrapes arXiv HTML pages (via ar5iv) for paper metadata, figures, captions,
and in-text figure references, then writes everything to a structured
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
│   └── <year>_<month>.tsv        columns: paper_id, figure_id, sub_id, caption, sub_caption
├── references/
│   └── ref_<year>_<month>.tsv    columns: paper_id, figure_id, reference_text
└── metadata_<year>_<month>.tsv   columns: url, paper_id, title, abstract, categories
 
Usage
-----
    python arxiv_scraper.py <year> <month>
 
Example
-------
    python arxiv_scraper.py 24 10   # scrapes all of October 2024
"""
 
from __future__ import annotations
 
import csv
import logging
import os
import re
import time
import xml.etree.ElementTree as ET
from pathlib import Path
from typing import Optional, List, Tuple
from urllib.parse import urljoin
 
import requests
from bs4 import BeautifulSoup, NavigableString
 
 
# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
 
ARXIV_HTML_BASE = "https://ar5iv.labs.arxiv.org/html/"
 
# Delay between requests to avoid overloading the server
REQUEST_DELAY_SECONDS = 1.0
 
# Retry settings for transient network errors
MAX_RETRIES = 3
RETRY_BACKOFF_SECONDS = 5.0
 
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
    connection problems. Returns None if the page does not exist (404)
    or if every retry attempt fails.
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
                return None
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
 
    Creates parent directories as needed. Returns True on success, False on failure.
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
 
def extract_text_with_math(element) -> str:
    """
    Recursively extract text from a BeautifulSoup element, rendering
    LaTeX math tags as inline ``$...$`` expressions rather than discarding them.
    """
    if isinstance(element, NavigableString):
        return str(element)
    if element.name == "math":
        alt = element.get("alttext", "")
        return f"${alt}$" if alt else element.get_text()
    return "".join(extract_text_with_math(child) for child in element.children)
 
 
def _resolve_image_url(raw_src: str, paper_url: str) -> str:
    """Return an absolute URL for *raw_src*, resolving relative paths against *paper_url*."""
    return urljoin(paper_url, raw_src)
 
 
def _outer_caption(fig_tag: BeautifulSoup) -> str:
    """
    Extract the top-level caption text from a figure element.
 
    Looks only at direct figcaption children with class ``ltx_caption``
    to avoid picking up sub-panel captions.
    """
    for tag in fig_tag.children:
        if tag.name == "figcaption" and "ltx_caption" in tag.get("class", []):
            raw = extract_text_with_math(tag)
            return " ".join(raw.split())
    return ""
 
 
def split_subcaptions(caption: str) -> List[Tuple[str, str]]:
    """
    Split a compound figure caption into labeled sub-panel entries.
 
    Example input:  "Figure 1: (a) First panel. (b) Second panel."
    Example output: [("a", "First panel."), ("b", "Second panel.")]
 
    Returns an empty list if no sub-panel labels are found.
    """
    pattern = re.compile(r"\(([a-z])\)\s*([^)]*?)(?=\s*\([a-z]\)|$)")
    matches = pattern.findall(caption)
    return [(letter, text.strip()) for letter, text in matches]
 
 
def is_equation_figure(fig_tag) -> bool:
    """
    Return True if this ``ltx_figure`` element represents a math equation
    rather than an actual figure.
 
    arXiv HTML uses the same ``ltx_figure`` class for both equations and
    figures, so we filter equations out by checking for equation-specific
    class names, the absence of any ``<img>`` tag, or a caption that begins
    with the word "Equation".
    """
    if fig_tag.find(class_=re.compile(r"ltx_eqn|ltx_equation|ltx_Math")):
        return True
    if not fig_tag.find("img"):
        return True
    caption = _outer_caption(fig_tag)
    if re.match(r"^\s*equation\b", caption, re.IGNORECASE):
        return True
    return False
 
 
# Matches non-title metadata that ar5iv sometimes appends to the title element:
# journal names, conference classifications (CCS:), footnotes (Thanks:, Note:),
# and paper notes (This is a preprint..., This material is based...).
_TITLE_NOISE = re.compile(
    r"\s*(?:Journal|Conference|CCS|Thanks|Note|Price|DOI|ISBN)\s*:"
    r"|This\s+(?:is\s+a\s+preprint|material\s+is\s+based|work\s+was\s+supported)",
    re.IGNORECASE,
)
 
 
def _clean_title(title: str) -> str:
    """Strip journal names, conference metadata, and footnotes from a raw title string."""
    match = _TITLE_NOISE.search(title)
    if match:
        title = title[: match.start()].strip()
    return title
 
 
def parse_title_abstract(soup: BeautifulSoup) -> tuple[str, str]:
    """
    Extract the paper title and abstract from a parsed arXiv HTML page.
 
    The raw title from ar5iv sometimes contains journal names, conference
    classification tags, or footnotes appended directly to the title text.
    ``_clean_title`` strips these before returning.
 
    Returns a ``(title, abstract)`` tuple. Either value is an empty string
    when the expected element cannot be found.
    """
    title_tag = soup.find("h1", class_="ltx_title_document")
    title = ""
    if title_tag:
        raw = " ".join(extract_text_with_math(title_tag).split())
        title = _clean_title(raw)
 
    abstract_div = soup.find("div", class_="ltx_abstract")
    abstract = ""
    if abstract_div:
        abstract_p = abstract_div.find("p", class_="ltx_p")
        if abstract_p:
            abstract = " ".join(extract_text_with_math(abstract_p).split())
 
    return title, abstract
 
 
def parse_figures(soup: BeautifulSoup, paper_url: str) -> list[dict]:
    """
    Extract all figures from a parsed arXiv HTML page.
 
    Handles both simple figures and multi-panel figures (subfigures labeled
    (a), (b), etc.). Equation figures are skipped.
 
    Returns a list of dicts with keys:
        figure_id   – integer figure number
        sub_id      – sub-panel letter ("a", "b", …) or None
        source      – absolute URL of the figure image, or None
        caption     – outer (top-level) caption text
        sub_caption – individual panel caption text, or None
    """
    rows: list[dict] = []
    sequential_idx = 0
 
    top_level_figures = [
        fig for fig in soup.find_all("figure", class_="ltx_figure")
        if "ltx_figure_panel" not in fig.get("class", [])
    ]
 
    for fig in top_level_figures:
        if is_equation_figure(fig):
            continue
 
        sequential_idx += 1
        outer_caption = _outer_caption(fig)
 
        num_match = re.search(r"\bFigure\s+(\d+)", outer_caption, re.IGNORECASE)
        fig_id = int(num_match.group(1)) if num_match else sequential_idx
 
        panels = fig.find_all("figure", class_="ltx_figure_panel")
 
        if panels:
            panel_captions = [_outer_caption(panel) for panel in panels]
 
            if all(c == "" for c in panel_captions):
                # No individual panel captions — try splitting the outer caption
                sub_parts = split_subcaptions(outer_caption)
                if sub_parts:
                    while len(sub_parts) < len(panels):
                        sub_parts.append(("", ""))
                    panel_captions = [text for _, text in sub_parts[: len(panels)]]
                    panel_sub_ids = [letter for letter, _ in sub_parts[: len(panels)]]
                    for i, (label, _) in enumerate(sub_parts):
                        if not label:
                            panel_sub_ids[i] = chr(ord("a") + i)
                else:
                    panel_sub_ids = [chr(ord("a") + i) for i in range(len(panels))]
                    panel_captions = ["" for _ in panels]
            else:
                panel_sub_ids = []
                for sub in panel_captions:
                    sub_match = re.search(r"\(([a-z])\)", sub, re.IGNORECASE)
                    panel_sub_ids.append(sub_match.group(1).lower() if sub_match else None)
 
            for idx, panel in enumerate(panels):
                img_tag = panel.find("img")
                img_src = (
                    _resolve_image_url(img_tag["src"], paper_url)
                    if img_tag and img_tag.get("src")
                    else None
                )
                rows.append({
                    "figure_id":   fig_id,
                    "sub_id":      panel_sub_ids[idx] if idx < len(panel_sub_ids) else None,
                    "source":      img_src,
                    "caption":     outer_caption,
                    "sub_caption": panel_captions[idx] if idx < len(panel_captions) else "",
                })
        else:
            img_tag = fig.find("img")
            img_src = (
                _resolve_image_url(img_tag["src"], paper_url)
                if img_tag and img_tag.get("src")
                else None
            )
            rows.append({
                "figure_id":   fig_id,
                "sub_id":      None,
                "source":      img_src,
                "caption":     outer_caption,
                "sub_caption": None,
            })
 
    return rows
 
 
def parse_figure_references(soup: BeautifulSoup) -> dict[str, list[str]]:
    """
    Find every paragraph in the paper body that mentions a figure by number,
    and return them grouped by figure label.
 
    Returns a dict mapping ``"Figure N"`` to a list of paragraph strings that
    contain a reference to that figure.
    """
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
            text = " ".join(extract_text_with_math(para).split())
            if pattern.search(text) and text not in references[key]:
                references[key].append(text)
 
    return references
 
 
def parse_categories(soup: BeautifulSoup, paper_id: str) -> list[str]:
    """
    Fetch arXiv subject categories for *paper_id* via the official arXiv API.
 
    Categories are not embedded in the ar5iv HTML body, so we query
    ``export.arxiv.org/api/query`` and parse the Atom XML response.
 
    Returns a list of category strings (e.g. ``["cs.HC", "cs.AI"]``), or an
    empty list if the API call fails.
    """
    categories = []
    try:
        log.info("Fetching categories via arXiv API for ID: %s", paper_id)
        api_url = f"https://export.arxiv.org/api/query?id_list={paper_id}"
        api_resp = requests.get(api_url, timeout=10)
        if api_resp.status_code == 200:
            root = ET.fromstring(api_resp.content)
            for category_tag in root.findall(
                ".//{http://www.w3.org/2005/Atom}category"
            ):
                term = category_tag.get("term")
                if term and "." in term and term not in categories:
                    categories.append(term)
    except Exception as exc:
        log.warning("arXiv API category lookup failed: %s", exc)
    return categories
 
 
# ---------------------------------------------------------------------------
# Output helpers
# ---------------------------------------------------------------------------
 
def _tsv_writer(path: Path, fieldnames: list[str]) -> tuple:
    """
    Open *path* for appending in TSV mode and return ``(file_handle, writer)``.
 
    Creates parent directories as needed. Writes the header row only when the
    file does not yet exist (i.e. on the very first append).
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
        Zero-padded five-digit sequence number, e.g. ``"12325"``.
        Combined with *year* and *month* it forms the full ID ``YYMM.NNNNN``
        (e.g. ``"2410.12325"``).
    year:
        Two-digit year string, e.g. ``"24"``.
    month:
        Two-digit month string, e.g. ``"10"``.
    output_dir:
        Root directory for all output files.
 
    Returns
    -------
    bool
        True if the paper was found and processed successfully, False otherwise.
    """
    full_id   = f"{year}{month}.{paper_id}"
    paper_url = ARXIV_HTML_BASE + full_id
    log.info("Processing %s …", paper_url)
 
    soup = fetch_soup(paper_url)
    if soup is None:
        return False
 
    title, abstract = parse_title_abstract(soup)
    figures         = parse_figures(soup, paper_url)
    fig_references  = parse_figure_references(soup)
    categories      = parse_categories(soup, full_id)
 
    # Skip papers where ar5iv returned a page but has no parseable content.
    # This avoids writing empty rows to the metadata/captions/references TSVs.
    if not title and not abstract and not figures:
        log.info("  ✗ %s – page found but no parseable content, skipping.", full_id)
        return True
 
    # Save figure images
    fig_dir = output_dir / "figures" / year / month / full_id
    fig_dir.mkdir(parents=True, exist_ok=True)
    for row in figures:
        if not row["source"]:
            continue
        stem = str(row["figure_id"]) + (row["sub_id"] or "")
        download_image(row["source"], fig_dir / f"{stem}.png")
 
    # Append captions
    cap_fh, cap_writer = _tsv_writer(
        output_dir / "captions" / f"{year}_{month}.tsv",
        fieldnames=["paper_id", "figure_id", "sub_id", "caption", "sub_caption"],
    )
    try:
        for row in figures:
            cap_writer.writerow({
                "paper_id":    full_id,
                "figure_id":   row["figure_id"],
                "sub_id":      row["sub_id"] or "",
                "caption":     row["caption"],
                "sub_caption": row["sub_caption"] or "",
            })
    finally:
        cap_fh.close()
 
    # Append in-text figure references
    ref_fh, ref_writer = _tsv_writer(
        output_dir / "references" / f"ref_{year}_{month}.tsv",
        fieldnames=["paper_id", "figure_id", "reference_text"],
    )
    try:
        for label, paragraphs in fig_references.items():
            id_match = re.search(r"\d+", label)
            fig_id = int(id_match.group()) if id_match else label
            for para_text in paragraphs:
                ref_writer.writerow({
                    "paper_id":       full_id,
                    "figure_id":      fig_id,
                    "reference_text": para_text,
                })
    finally:
        ref_fh.close()
 
    # Append paper metadata
    meta_fh, meta_writer = _tsv_writer(
        output_dir / f"metadata_{year}_{month}.tsv",
        fieldnames=["url", "paper_id", "title", "abstract", "categories"],
    )
    try:
        meta_writer.writerow({
            "url":        paper_url,
            "paper_id":   full_id,
            "title":      title,
            "abstract":   abstract,
            "categories": "; ".join(categories),
        })
    finally:
        meta_fh.close()
 
    log.info("  ✓ %s – %d figure row(s), %d category/ies", full_id, len(figures), len(categories))
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
    Scrape all papers for a given *year* / *month* in sequence.
 
    Iterates over paper IDs starting from *start_id* and stops when a paper
    is not found (indicating the end of that month's submissions), or when
    *max_papers* papers have been processed.
 
    Parameters
    ----------
    year:
        Two-digit year, e.g. ``"24"``.
    month:
        Two-digit month, e.g. ``"10"``.
    output_dir:
        Root directory for all output files.
    max_papers:
        Stop after this many successful papers. Pass None to scrape all.
    start_id:
        Numeric ID to start from (default 1). Useful for resuming an
        interrupted run.
    """
    log.info("=== Scraping %s/%s (starting at %s.%05d) ===", year, month, year + month, start_id)
    processed = 0
 
    for numeric_id in range(start_id, 100_000):
        paper_id = f"{numeric_id:05d}"
        found = process_paper(paper_id, year, month, output_dir)
 
        if not found:
            log.info("No paper at id %s – continuing to next.", paper_id)
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
        List of two-digit month strings, e.g. ``["01", "02", ..., "12"]``.
    output_dir:
        Root directory for all output files.
    max_papers_per_month:
        Cap on papers scraped per month. Useful for testing.
    """
    for year in years:
        for month in months:
            scrape_month(year, month, output_dir, max_papers=max_papers_per_month)
 
 
# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------
 
if __name__ == "__main__":
    import argparse
 
    parser = argparse.ArgumentParser(
        description="Scrape arXiv figures and metadata for a given month.",
        epilog="Example: python arxiv_scraper.py 24 10",
    )
    parser.add_argument("year",  help="Two-digit year  (e.g. 24 for 2024)")
    parser.add_argument("month", help="Two-digit month (e.g. 10 for October)")
    parser.add_argument(
        "--max-papers", type=int, default=None,
        help="Stop after this many papers (useful for testing).",
    )
    parser.add_argument(
        "--start-id", type=int, default=1,
        help="Paper sequence number to start from (default 1).",
    )
    args = parser.parse_args()
 
    OUTPUT_ROOT = Path("arxiv_data")
    scrape_month(args.year, args.month, OUTPUT_ROOT,
                 max_papers=args.max_papers, start_id=args.start_id)