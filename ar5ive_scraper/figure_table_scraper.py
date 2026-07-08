"""
arxiv_scraper.py
=================
Scrapes ar5iv (arXiv) HTML pages for a paper's metadata, figures
(including sub-figures/panels), tables (including sub-tables/panels),
captions, and in-text references to each figure/table — then writes
everything to a clean, separated file hierarchy.

This file merges the previous `full_ar5ive_scraper.py` (figures +
metadata) and `table_scraper.py` (tables) into a single pipeline that
fetches each paper's HTML exactly once and fans the results out into
five clearly separated outputs: metadata, figures (images), table data,
captions, and references.

Output layout
-------------
<output_dir>/
├── figures/
│   └── <year>/
│       └── <month>/
│           └── <paper_id>/
│               ├── 1.png          # simple figure
│               ├── 3a.png         # sub-figure / panel "a" of figure 3
│               └── ...
├── tables/
│   └── <year>_<month>.jsonl       # one JSON object per table (see below)
├── captions/
│   └── <year>_<month>.tsv         # columns: paper_id, item_type, item_id,
│                                   #   sub_id, caption, sub_caption
├── references/
│   └── <year>_<month>.tsv         # columns: paper_id, item_type, item_id,
│                                   #   reference_text
└── metadata/
    └── <year>_<month>.tsv         # columns: url, paper_id, title,
                                    #   abstract, categories

`item_type` in captions/references is either "figure" or "table", so
both kinds of items live in the same two files but stay easy to filter.

Table JSONL record shape (one line per table / sub-table panel)
-----------------------------------------------------------------
{
  "paper_id": "2410.00004",
  "table_id": "2410.00004_T1",       # <paper_id>_T<table number>[<sub_id>]
  "html_id": "S3.T1",                # raw id attribute from the HTML
  "table_num": 1,
  "sub_id": null,                    # "a" / "b" / ... for multi-panel tables
  "n_rows": 7,
  "n_cols": 3,
  "cells": [
      {"row": 0, "col": 0, "text": "# Neighbors", "is_header": true},
      ...
      {"row": 3, "col": 0, "text": "0", "is_header": true, "rowspan": 4},
  ],
  "footnotes": [
      {"marker": "7", "text": "The vocabulary size of the original ..."}
  ]
}
Table captions and references live in captions/*.tsv and
references/*.tsv (item_type == "table"), keyed by table_id, so the
table_id links a JSONL row back to its caption/reference rows.
"""

from __future__ import annotations

import csv
import json
import logging
import re
import time
import xml.etree.ElementTree as ET
from pathlib import Path
from typing import Dict, List, Optional, Tuple
from urllib.parse import urljoin

import requests
from bs4 import BeautifulSoup, NavigableString, Tag

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

ARXIV_HTML_BASE = "https://ar5iv.labs.arxiv.org/html/"
Old_HTML_BASE = "https://arxiv.org/html/"

# How long to wait between requests so we don't hammer the server
REQUEST_DELAY_SECONDS = 1.0

# Retry settings for transient network errors
MAX_RETRIES = 1
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
    connection problems. A 404 is treated as "paper does not exist" and
    returns None immediately without retrying.
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
    """Download a binary resource (image) from *url* and save it to *dest_path*."""
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


def _resolve_image_url(raw_src: str, paper_url: str) -> str:
    """Absolute-ize *raw_src* against *paper_url* (handles ar5iv's absolute paths too)."""
    return urljoin(paper_url, raw_src)


# ---------------------------------------------------------------------------
# Text + math + footnote extraction (shared by figures and tables)
# ---------------------------------------------------------------------------

def _flatten_nested_table(table_tag: Tag) -> str:
    """Ar5iv sometimes puts a tiny <table> inside a <th>/<td> to render a
    multi-line header. Flatten its cell text into one readable string
    instead of treating it as a nested table."""
    parts = []
    for cell in table_tag.find_all(["td", "th"]):
        txt = " ".join(cell.get_text().split())
        if txt:
            parts.append(txt)
    return " ".join(parts)


def extract_text_and_footnotes(element) -> Tuple[str, List[dict]]:
    """
    Walk *element*, returning (clean_text, footnotes) where:
      - clean_text has math converted to '$...$' and footnote markers
        replaced with a lightweight '[^N]' inline marker
      - footnotes is [{"marker": "N", "text": "..."}] pulled out of any
        ltx_note/ltx_role_footnote spans found inside *element*

    Used for both figure captions and table captions/cells so footnote
    handling and math rendering stay consistent across the whole scraper.
    """
    footnotes: List[dict] = []

    def _walk(el) -> str:
        if isinstance(el, NavigableString):
            return str(el)

        classes = el.get("class", []) if isinstance(el, Tag) else []

        if el.name == "math":
            alt = el.get("alttext", "")
            return f"${alt}$" if alt else el.get_text()

        if el.name == "table":
            return _flatten_nested_table(el)

        if "ltx_note" in classes and "ltx_role_footnote" in classes:
            mark_sup = el.find("sup", class_="ltx_note_mark")
            marker = mark_sup.get_text(strip=True) if mark_sup else "?"

            content_span = el.find("span", class_="ltx_note_content")
            note_text = ""
            if content_span is not None:
                inner = []
                for child in content_span.children:
                    if isinstance(child, NavigableString):
                        inner.append(str(child))
                        continue
                    child_classes = child.get("class", []) if isinstance(child, Tag) else []
                    if child.name == "sup" and "ltx_note_mark" in child_classes:
                        continue
                    if "ltx_tag_note" in child_classes:
                        continue
                    inner.append(_walk(child))
                note_text = " ".join("".join(inner).split())

            footnotes.append({"marker": marker, "text": note_text})
            return f"[^{marker}]"

        return "".join(_walk(child) for child in el.children)

    raw_text = _walk(element)
    clean_text = " ".join(raw_text.split())
    return clean_text, footnotes


def extract_text_with_math(element) -> str:
    """Lightweight text extraction (math-aware, no footnote pulling)."""
    text, _ = extract_text_and_footnotes(element)
    return text


def split_subcaptions(caption: str) -> List[Tuple[str, str]]:
    """
    Split a compound caption like "Figure 1: (a) First panel (b) Second panel."
    into [(letter, text), ...] for each sub-panel.
    """
    pattern = re.compile(r"\(([a-z])\)\s*([^)]*?)(?=\s*\([a-z]\)|$)")
    matches = pattern.findall(caption)
    return [(letter, text.strip()) for letter, text in matches]


# ---------------------------------------------------------------------------
# Metadata parsing
# ---------------------------------------------------------------------------

def parse_title_abstract(soup: BeautifulSoup) -> Tuple[str, str]:
    """Extract the paper title and abstract text. Returns (title, abstract)."""
    title_tag = soup.find("h1", class_="ltx_title_document")
    title = ""
    if title_tag:
        title = " ".join(extract_text_with_math(title_tag).split())

    abstract_div = soup.find("div", class_="ltx_abstract")
    abstract = ""
    if abstract_div:
        abstract_p = abstract_div.find("p", class_="ltx_p")
        if abstract_p:
            abstract = " ".join(extract_text_with_math(abstract_p).split())

    return title, abstract


def parse_categories(paper_id: str) -> List[str]:
    """
    Extract arXiv subject categories via the official arXiv API, since
    categories are missing from the raw ar5iv HTML body.
    """
    categories: List[str] = []
    try:
        log.info("Fetching categories via arXiv API for ID: %s", paper_id)
        api_url = f"https://export.arxiv.org/api/query?id_list={paper_id}"
        api_resp = requests.get(api_url, timeout=10)

        if api_resp.status_code == 200:
            root = ET.fromstring(api_resp.content)
            for category_tag in root.findall(".//{http://www.w3.org/2005/Atom}category"):
                term = category_tag.get("term")
                if term and "." in term and term not in categories:
                    categories.append(term)
    except Exception as exc:
        log.warning("arXiv API metadata fallback query failed: %s", exc)

    return categories


# ---------------------------------------------------------------------------
# Figure parsing
# ---------------------------------------------------------------------------

def _outer_caption(fig_tag: Tag) -> Tuple[str, List[dict]]:
    for tag in fig_tag.children:
        if isinstance(tag, Tag) and tag.name == "figcaption" and "ltx_caption" in tag.get("class", []):
            return extract_text_and_footnotes(tag)
    return "", []


def parse_figures(soup: BeautifulSoup, paper_url: str) -> List[dict]:
    """
    Find every top-level figure and return one record per figure (or per
    sub-panel, for multi-part figures like Figure 3(a)/(b)).

    Each record: {figure_id, sub_id, source, caption, sub_caption, footnotes}
    """
    rows: List[dict] = []
    sequential_idx = 0

    top_level_figures = [
        fig for fig in soup.find_all("figure", class_="ltx_figure")
        if "ltx_figure_panel" not in fig.get("class", [])
    ]

    for fig in top_level_figures:
        sequential_idx += 1
        outer_caption, outer_footnotes = _outer_caption(fig)

        num_match = re.search(r"\bFigure\s+(\d+)", outer_caption, re.IGNORECASE)
        fig_id = int(num_match.group(1)) if num_match else sequential_idx

        panels = fig.find_all("figure", class_="ltx_figure_panel")

        if panels:
            panel_captions = []
            panel_footnotes_list = []
            for panel in panels:
                sub_cap, sub_fns = _outer_caption(panel)
                panel_captions.append(sub_cap)
                panel_footnotes_list.append(sub_fns)

            if all(c == "" for c in panel_captions):
                sub_parts = split_subcaptions(outer_caption)
                if sub_parts:
                    while len(sub_parts) < len(panels):
                        sub_parts.append(("", ""))
                    panel_captions = [text for _, text in sub_parts[:len(panels)]]
                    panel_sub_ids = [letter for letter, _ in sub_parts[:len(panels)]]
                    for i, (label, _text) in enumerate(sub_parts):
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
                for i, sid in enumerate(panel_sub_ids):
                    if sid is None:
                        panel_sub_ids[i] = chr(ord("a") + i)

            for idx, panel in enumerate(panels):
                img_tag = panel.find("img")
                img_src = (
                    _resolve_image_url(img_tag["src"], paper_url)
                    if img_tag and img_tag.get("src")
                    else None
                )
                sub_caption = panel_captions[idx] if idx < len(panel_captions) else ""
                sub_id = panel_sub_ids[idx] if idx < len(panel_sub_ids) else chr(ord("a") + idx)
                fns = list(outer_footnotes)
                fns.extend(panel_footnotes_list[idx] if idx < len(panel_footnotes_list) else [])

                rows.append({
                    "figure_id": fig_id,
                    "sub_id": sub_id,
                    "source": img_src,
                    "caption": outer_caption,
                    "sub_caption": sub_caption,
                    "footnotes": fns,
                })
        else:
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
                "footnotes": outer_footnotes,
            })

    return rows


def parse_figure_references(soup: BeautifulSoup) -> Dict[str, List[str]]:
    """Return {"Figure N": [paragraph_text, ...]} scanning body paragraphs
    for mentions like 'Figure 3', 'Fig. 3', 'Fig 3a'."""
    figure_numbers: List[str] = []
    references: Dict[str, List[str]] = {}
    top_level_figures = [
        fig for fig in soup.find_all("figure", class_="ltx_figure")
        if "ltx_figure_panel" not in fig.get("class", [])
    ]
    for seq_idx, fig in enumerate(top_level_figures, start=1):
        caption, _ = _outer_caption(fig)
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
            text = " ".join(text.split())
            if pattern.search(text) and text not in references[key]:
                references[key].append(text)
    return references


# ---------------------------------------------------------------------------
# Table parsing
# ---------------------------------------------------------------------------

def _iter_top_level_rows(table_tag: Tag) -> List[Tag]:
    """This table's own <tr> elements in document order, without descending
    into any nested <table> that might live inside a cell."""
    rows: List[Tag] = []
    for child in table_tag.find_all(["thead", "tbody", "tfoot", "tr"], recursive=False):
        if child.name == "tr":
            rows.append(child)
        else:
            rows.extend(child.find_all("tr", recursive=False))
    return rows


def extract_table_grid(table_tag: Tag) -> Tuple[List[dict], int, int, List[dict]]:
    """
    Build the full cell grid for *table_tag*, expanding rowspan/colspan
    bookkeeping so each cell gets correct (row, col) coordinates.
    Returns (cells, n_rows, n_cols, footnotes). Only the top-left cell of
    a span is emitted, carrying its span size.
    """
    cells: List[dict] = []
    footnotes: List[dict] = []
    occupied: dict = {}

    row_idx = 0
    for tr in _iter_top_level_rows(table_tag):
        col_idx = 0
        row_had_cells = False
        for cell_tag in tr.find_all(["th", "td"], recursive=False):
            row_had_cells = True
            while occupied.get((row_idx, col_idx)):
                col_idx += 1

            colspan = int(cell_tag.get("colspan", 1) or 1)
            rowspan = int(cell_tag.get("rowspan", 1) or 1)

            text, cell_footnotes = extract_text_and_footnotes(cell_tag)
            footnotes.extend(cell_footnotes)

            cell = {
                "row": row_idx,
                "col": col_idx,
                "text": text,
                "is_header": cell_tag.name == "th",
            }
            if colspan > 1:
                cell["colspan"] = colspan
            if rowspan > 1:
                cell["rowspan"] = rowspan
            cells.append(cell)

            for r in range(row_idx, row_idx + rowspan):
                for c in range(col_idx, col_idx + colspan):
                    occupied[(r, c)] = True

            col_idx += colspan

        if row_had_cells:
            row_idx += 1

    n_rows = row_idx
    n_cols = (max(c for _, c in occupied) + 1) if occupied else 0
    return cells, n_rows, n_cols, footnotes


def _dedupe_footnotes(footnotes: List[dict]) -> List[dict]:
    seen = set()
    out = []
    for fn in footnotes:
        key = (fn["marker"], fn["text"])
        if key not in seen:
            seen.add(key)
            out.append(fn)
    return out


def _split_into_logical_groups(table_fig: Tag) -> List[Tuple[Tag, List[Tag]]]:
    """
    Most `<figure class="ltx_table">` wrappers contain exactly one logical
    table. Occasionally one wrapper contains SEVERAL logical tables back
    to back (multiple <figcaption> direct children). This splits the
    wrapper's direct children into (figcaption, [content_tags]) groups so
    each logical table is handled independently.
    """
    groups: List[Tuple[Tag, List[Tag]]] = []
    current = None
    for child in table_fig.children:
        if not isinstance(child, Tag):
            continue
        if child.name == "figcaption":
            current = (child, [])
            groups.append(current)
        elif current is not None:
            current[1].append(child)
    return groups


def _find_panels(content_tags: List[Tag]) -> List[Tag]:
    panels: List[Tag] = []
    for tag in content_tags:
        if tag.name == "figure" and "ltx_figure_panel" in tag.get("class", []):
            panels.append(tag)
        else:
            panels.extend(tag.find_all("figure", class_="ltx_figure_panel"))
    return panels


def _find_direct_table(content_tags: List[Tag]) -> Optional[Tag]:
    for tag in content_tags:
        if tag.name == "table":
            return tag
        found = tag.find("table")
        if found is not None:
            return found
    return None


def parse_tables(soup: BeautifulSoup, paper_id: str) -> List[dict]:
    """
    Find every table on the page and return one record per logical table
    (or per sub-panel, for multi-part tables like Table 14(a)/(b)).

    Each record carries its own caption/sub_caption/footnotes so callers
    can fan them out into separate caption/reference/table-data outputs.
    """
    records: List[dict] = []
    sequential_idx = 0

    top_level_tables = [
        fig for fig in soup.find_all("figure", class_="ltx_table")
        if "ltx_figure_panel" not in fig.get("class", [])
    ]

    for table_fig in top_level_tables:
        wrapper_html_id = table_fig.get("id", "")

        for figcaption_tag, content_tags in _split_into_logical_groups(table_fig):
            sequential_idx += 1
            outer_caption, outer_footnotes = extract_text_and_footnotes(figcaption_tag)

            num_match = re.search(r"\bTable\s+(\d+)", outer_caption, re.IGNORECASE)
            table_num = int(num_match.group(1)) if num_match else sequential_idx

            panels = _find_panels(content_tags)

            if panels:
                _append_panel_records(
                    records, panels, outer_caption, outer_footnotes,
                    table_num, wrapper_html_id, paper_id,
                )
            else:
                table_tag = _find_direct_table(content_tags)
                if table_tag is None:
                    continue
                _append_single_record(
                    records, table_tag, outer_caption, outer_footnotes,
                    table_num, wrapper_html_id, paper_id,
                )

    return records


def _append_panel_records(
    records: List[dict], panels: List[Tag], outer_caption: str,
    outer_footnotes: List[dict], table_num: int, wrapper_html_id: str, paper_id: str,
) -> None:
    panel_captions = []
    panel_footnotes_list = []
    for panel in panels:
        sub_cap, sub_fns = _outer_caption(panel)
        panel_captions.append(sub_cap)
        panel_footnotes_list.append(sub_fns)

    if all(c == "" for c in panel_captions):
        sub_parts = split_subcaptions(outer_caption)
        while len(sub_parts) < len(panels):
            sub_parts.append(("", ""))
        panel_captions = [text for _, text in sub_parts[:len(panels)]]
        panel_sub_ids = [letter for letter, _ in sub_parts[:len(panels)]]
        for i, (label, _text) in enumerate(sub_parts):
            if not label:
                panel_sub_ids[i] = chr(ord("a") + i)
    else:
        panel_sub_ids = []
        for sub in panel_captions:
            sub_match = re.search(r"\(([a-z])\)", sub, re.IGNORECASE)
            panel_sub_ids.append(sub_match.group(1).lower() if sub_match else None)
        for i, sid in enumerate(panel_sub_ids):
            if sid is None:
                panel_sub_ids[i] = chr(ord("a") + i)

    for idx, panel in enumerate(panels):
        table_tag = panel.find("table")
        if table_tag is None:
            continue
        cells, n_rows, n_cols, cell_footnotes = extract_table_grid(table_tag)
        sub_id = panel_sub_ids[idx] if idx < len(panel_sub_ids) else chr(ord("a") + idx)
        sub_caption = panel_captions[idx] if idx < len(panel_captions) else ""
        fns = list(outer_footnotes)
        fns.extend(panel_footnotes_list[idx] if idx < len(panel_footnotes_list) else [])
        fns.extend(cell_footnotes)

        records.append({
            "paper_id": paper_id,
            "table_id": f"{paper_id}_T{table_num}{sub_id}",
            "html_id": panel.get("id", wrapper_html_id),
            "table_num": table_num,
            "sub_id": sub_id,
            "caption": outer_caption,
            "sub_caption": sub_caption,
            "n_rows": n_rows,
            "n_cols": n_cols,
            "cells": cells,
            "footnotes": _dedupe_footnotes(fns),
        })


def _append_single_record(
    records: List[dict], table_tag: Tag, outer_caption: str,
    outer_footnotes: List[dict], table_num: int, wrapper_html_id: str, paper_id: str,
) -> None:
    cells, n_rows, n_cols, cell_footnotes = extract_table_grid(table_tag)
    fns = list(outer_footnotes)
    fns.extend(cell_footnotes)

    records.append({
        "paper_id": paper_id,
        "table_id": f"Table {table_num}",
        "html_id": wrapper_html_id,
        "table_num": table_num,
        "sub_id": None,
        "caption": outer_caption,
        "sub_caption": None,
        "n_rows": n_rows,
        "n_cols": n_cols,
        "cells": cells,
        "footnotes": _dedupe_footnotes(fns),
    })


def parse_table_references(soup: BeautifulSoup) -> Dict[int, List[str]]:
    """Return {table_num: [paragraph_text, ...]} scanning body paragraphs
    for mentions like 'Table 3', 'Tab. 3', 'Tab 3a'."""
    table_numbers: List[int] = []
    sequential_idx = 0

    top_level_tables = [
        fig for fig in soup.find_all("figure", class_="ltx_table")
        if "ltx_figure_panel" not in fig.get("class", [])
    ]
    for table_fig in top_level_tables:
        for figcaption_tag, _content_tags in _split_into_logical_groups(table_fig):
            sequential_idx += 1
            caption, _ = extract_text_and_footnotes(figcaption_tag)
            match = re.search(r"\bTable\s+(\d+)", caption, re.IGNORECASE)
            table_num = int(match.group(1)) if match else sequential_idx
            table_numbers.append(table_num)

    references: Dict[int, List[str]] = {n: [] for n in table_numbers}

    all_paragraphs = soup.find_all("p", class_="ltx_p")
    for table_num in table_numbers:
        pattern = re.compile(
            rf"\bTab(?:le|\.)?\s*{table_num}(?:[a-z]|\([a-z]\)|-[a-z])?\b",
            re.IGNORECASE,
        )
        for para in all_paragraphs:
            if para.find_parent("figure") is not None:
                continue
            text, _ = extract_text_and_footnotes(para)
            if pattern.search(text) and text not in references[table_num]:
                references[table_num].append(text)

    return references


# ---------------------------------------------------------------------------
# Output helpers
# ---------------------------------------------------------------------------

def _tsv_writer(path: Path, fieldnames: List[str]) -> tuple:
    """Open *path* for appending in TSV mode; write header only if new."""
    path.parent.mkdir(parents=True, exist_ok=True)
    is_new = not path.exists()
    fh = open(path, "a", newline="", encoding="utf-8")
    writer = csv.DictWriter(fh, fieldnames=fieldnames, delimiter="\t")
    if is_new:
        writer.writeheader()
    return fh, writer


def write_tables_jsonl(records: List[dict], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "a", encoding="utf-8") as fh:
        for rec in records:
            fh.write(json.dumps(rec, ensure_ascii=False) + "\n")


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
    Scrape a single arXiv paper (one HTTP fetch of the ar5iv HTML page) and
    persist metadata, figures, table data, captions, and references into
    their own separated output files.

    Parameters
    ----------
    paper_id:
        The five-digit arXiv sequence number, e.g. ``"12325"``.
        Combined with *year* and *month* it forms the full ID ``YYMM.NNNNN``.
    year, month:
        Two-digit strings, e.g. ``"25"`` / ``"10"``.
    output_dir:
        Root directory for all output files.

    Returns
    -------
    bool
        True if the paper was found and processed, False otherwise.
    """
    full_id = f"{year}{month}.{paper_id}"
    paper_url = ARXIV_HTML_BASE + full_id
    log.info("Processing %s …", paper_url)

    soup = fetch_soup(paper_url)
    if soup is None:
        return False

    # ------------------------------------------------------------ parse
    title, abstract = parse_title_abstract(soup)
    categories = parse_categories(full_id)
    figures = parse_figures(soup, paper_url)
    fig_references = parse_figure_references(soup)
    tables = parse_tables(soup, full_id)
    table_references = parse_table_references(soup)

    # --------------------------------------------------- save figure images
    fig_dir = output_dir / "figures" / year / month / full_id
    for row in figures:
        if not row["source"]:
            continue
        stem = str(row["figure_id"]) + (row["sub_id"] or "")
        dest = fig_dir / f"{stem}.png"
        download_image(row["source"], dest)

    # ------------------------------------------------------- table data
    tables_path = output_dir / "tables" / f"{year}_{month}.jsonl"
    write_tables_jsonl(tables, tables_path)

    # ------------------------------------------ captions TSV (figures + tables)
    captions_path = output_dir / "captions" / f"{year}_{month}.tsv"
    cap_fh, cap_writer = _tsv_writer(
        captions_path,
        fieldnames=["paper_id", "item_type", "item_id", "sub_id", "caption", "sub_caption"],
    )
    try:
        for row in figures:
            cap_writer.writerow({
                "paper_id": full_id,
                "item_type": "figure",
                "item_id": row["figure_id"],
                "sub_id": row["sub_id"] or "",
                "caption": row["caption"],
                "sub_caption": row["sub_caption"] or "",
            })
        for rec in tables:
            cap_writer.writerow({
                "paper_id": full_id,
                "item_type": "table",
                "item_id": rec["table_id"],
                "sub_id": rec["sub_id"] or "",
                "caption": rec["caption"],
                "sub_caption": rec["sub_caption"] or "",
            })
    finally:
        cap_fh.close()

    # -------------------------------------- references TSV (figures + tables)
    refs_path = output_dir / "references" / f"{year}_{month}.tsv"
    ref_fh, ref_writer = _tsv_writer(
        refs_path,
        fieldnames=["paper_id", "item_type", "item_id", "reference_text"],
    )
    try:
        for label, paragraphs in fig_references.items():
            id_match = re.search(r"\d+", label)
            fig_id = int(id_match.group()) if id_match else label
            for para_text in paragraphs:
                ref_writer.writerow({
                    "paper_id": full_id,
                    "item_type": "figure",
                    "item_id": fig_id,
                    "reference_text": para_text,
                })
        for rec in tables:
            for para_text in table_references.get(rec["table_num"], []):
                ref_writer.writerow({
                    "paper_id": full_id,
                    "item_type": "table",
                    "item_id": rec["table_id"],
                    "reference_text": para_text,
                })
    finally:
        ref_fh.close()

    # ----------------------------------------------------------- metadata
    meta_path = output_dir / "metadata" / f"{year}_{month}.tsv"
    meta_fh, meta_writer = _tsv_writer(
        meta_path,
        fieldnames=["url", "paper_id", "title", "abstract", "categories"],
    )
    try:
        meta_writer.writerow({
            "url": paper_url,
            "paper_id": full_id,
            "title": title,
            "abstract": abstract,
            "categories": "; ".join(categories),
        })
    finally:
        meta_fh.close()

    log.info(
        "  ✓ %s – %d figure row(s), %d table(s), %d categor(ies)",
        full_id, len(figures), len(tables), len(categories),
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
    Iterate over arXiv paper IDs for a given *year* / *month* and scrape
    each one until a 404 is returned (no more papers for that month).

    Parameters
    ----------
    year, month:
        Two-digit strings, e.g. ``"25"`` / ``"10"``.
    output_dir:
        Root directory for all output files.
    max_papers:
        Stop after processing this many papers (useful for testing).
        Pass None to scrape until arXiv returns a 404.
    start_id:
        Numeric ID to begin from (default 1); useful for resuming.
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
    years: List[str],
    months: List[str],
    output_dir: Path,
    max_papers_per_month: Optional[int] = None,
) -> None:
    """Scrape every (year, month) combination in *years* × *months*."""
    for year in years:
        for month in months:
            scrape_month(year, month, output_dir, max_papers=max_papers_per_month)


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    OUTPUT_ROOT = Path("arxiv_data")

    # ------------------------------------------------------------------
    # SAMPLE 1 — quick single-paper test
    # ------------------------------------------------------------------
    # test_paper_id = "00003"   # becomes 2410.00003
    # test_year = "24"
    # test_month = "10"
    # process_paper(test_paper_id, test_year, test_month, OUTPUT_ROOT)

    # ------------------------------------------------------------------
    # SAMPLE 2 — a single full month (uncomment to run)
    # ------------------------------------------------------------------
    scrape_month("24", "10", OUTPUT_ROOT)

    # ------------------------------------------------------------------
    # SAMPLE 3 — several months in one year
    # ------------------------------------------------------------------
    # for m in ["10", "11", "12"]:
    #     scrape_month("24", m, OUTPUT_ROOT)

    # ------------------------------------------------------------------
    # SAMPLE 4 — multiple years/months (capped for safety during dev)
    # ------------------------------------------------------------------
    # scrape_range(
    #     years=["24", "25"],
    #     months=["01", "02", "03", "04", "05", "06", "07", "08", "09", "10", "11", "12"],
    #     output_dir=OUTPUT_ROOT,
    #     max_papers_per_month=None,
    # )