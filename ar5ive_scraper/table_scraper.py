"""
table_scraper.py
=================
Scrapes arXiv/ar5iv HTML pages for TABLES: caption, full cell grid
(headers + values, with rowspan/colspan preserved), in-text references
to each table, and any footnotes that live inside the caption or cells.

Designed to sit alongside `full_ar5ive_scraper.py` and reuse the same
conventions (fetch_soup, extract_text_with_math, output layout style).
You can copy the functions below into that file, or import this module
and call `build_table_records(soup, paper_id)` from within `process_paper`.

Output record shape (one dict per table / sub-table panel)
------------------------------------------------------------------
{
  "table_id": "2410.00004_T1",       # <paper_id>_T<table number>[<sub_id>]
  "html_id": "S3.T1",                # raw id attribute from the HTML, stable anchor
  "table_num": 1,
  "sub_id": null,                    # "a" / "b" / ... for multi-panel tables
  "caption": "...",                  # full caption text (footnote markers as [^N])
  "sub_caption": null,               # panel-specific caption, if any
  "n_rows": 7,
  "n_cols": 3,
  "cells": [
      {"row": 0, "col": 0, "text": "# Neighbors", "is_header": true},
      {"row": 0, "col": 1, "text": "Bert",        "is_header": true},
      ...
      # cells that span multiple rows/cols carry rowspan/colspan (default 1, omitted)
      {"row": 3, "col": 0, "text": "0", "is_header": true, "rowspan": 4},
  ],
  "footnotes": [
      {"marker": "7", "text": "The vocabulary size of the original ..."}
  ],
  "references": [
      "Overall, adding neighbors at a stage where ... (Table 1) ..."
  ]
}

Notes on ar5iv quirks handled here
------------------------------------------------------------------
- rowspan/colspan: a full grid-position algorithm expands spans so every
  cell gets correct (row, col); only the ORIGIN cell of a span is emitted,
  carrying "rowspan"/"colspan" so you can re-expand it if you want a dense
  grid downstream.
- Multi-panel tables (subtables 14(a)/14(b) etc.): handled like the
  existing figure-panel logic in the figure scraper.
- Nested tables inside a header cell (used to stack a 2-line header like
  "Alpha on" / "Sequence"): flattened to plain text instead of being
  mis-parsed as a second table.
- Footnotes: ar5iv embeds full footnote text inline (in the caption or in
  a cell) wrapped in `ltx_note ltx_role_footnote` spans. We pull these out
  into a separate `footnotes` list and leave a lightweight `[^N]` marker
  in place in the visible text.
- Rare "merged wrapper" case: occasionally a single `<figure class=
  "ltx_table">` wraps SEVERAL logical tables back-to-back (multiple
  `<figcaption>` direct children, e.g. Tables 14/15/16 in one wrapper in
  some documents). We split on figcaption boundaries so each logical
  table becomes its own record instead of getting merged/duplicated.
"""

from __future__ import annotations

import json
import logging
import re
import time
from pathlib import Path
from typing import List, Optional, Tuple

import requests
from bs4 import BeautifulSoup, NavigableString, Tag

# ---------------------------------------------------------------------------
# Configuration (mirrors full_ar5ive_scraper.py)
# ---------------------------------------------------------------------------

ARXIV_HTML_BASE = "https://ar5iv.labs.arxiv.org/html/"

# How long to wait between requests so we don't hammer the server
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
# Networking helpers (identical strategy to full_ar5ive_scraper.py)
# ---------------------------------------------------------------------------

def fetch_soup(url: str) -> Optional[BeautifulSoup]:
    """
    Download *url* and return a BeautifulSoup parse tree.

    Retries up to MAX_RETRIES times on transient HTTP errors (5xx) or
    connection problems. A 404 is treated as "paper does not exist" and
    returns None immediately without retrying. Returns None if every
    attempt fails.
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


# ---------------------------------------------------------------------------
# Text + math + footnote extraction
# ---------------------------------------------------------------------------

def extract_text_with_math(element) -> str:
    """Same helper as in full_ar5ive_scraper.py — kept here so this module
    is standalone. If merging into the main scraper file, delete this copy
    and reuse the existing one."""
    if isinstance(element, NavigableString):
        return str(element)
    if element.name == "math":
        alt = element.get("alttext", "")
        return f"${alt}$" if alt else element.get_text()
    return "".join(extract_text_with_math(child) for child in element.children)


def _flatten_nested_table(table_tag: Tag) -> str:
    """Ar5iv sometimes puts a tiny <table> inside a <th> to render a
    multi-line header (e.g. 'Alpha on' / 'Sequence' stacked). Flatten its
    cell text into one readable string instead of treating it as a table."""
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
        ltx_note/ltx_role_footnote spans found inside *element* (these
        appear inline in ar5iv's HTML, in both captions and cells)
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
                        continue  # duplicate leading marker, skip
                    if "ltx_tag_note" in child_classes:
                        continue  # duplicate marker tag, skip
                    inner.append(_walk(child))
                note_text = " ".join("".join(inner).split())

            footnotes.append({"marker": marker, "text": note_text})
            return f"[^{marker}]"

        return "".join(_walk(child) for child in el.children)

    raw_text = _walk(element)
    clean_text = " ".join(raw_text.split())
    return clean_text, footnotes


# ---------------------------------------------------------------------------
# Caption helpers
# ---------------------------------------------------------------------------

def _outer_table_caption(table_fig: Tag) -> Tuple[str, List[dict]]:
    for tag in table_fig.children:
        if isinstance(tag, Tag) and tag.name == "figcaption" and "ltx_caption" in tag.get("class", []):
            return extract_text_and_footnotes(tag)
    return "", []


def split_subcaptions(caption: str) -> List[Tuple[str, str]]:
    """Split a compound caption like '(a) First panel (b) Second panel.'
    into [(letter, text), ...]. Mirrors the helper in the figure scraper."""
    pattern = re.compile(r"\(([a-z])\)\s*([^)]*?)(?=\s*\([a-z]\)|$)")
    matches = pattern.findall(caption)
    return [(letter, text.strip()) for letter, text in matches]


# ---------------------------------------------------------------------------
# Grid extraction (handles thead/tbody, rowspan, colspan, nested tables)
# ---------------------------------------------------------------------------

def _iter_top_level_rows(table_tag: Tag) -> List[Tag]:
    """Return this table's own <tr> elements in document order, without
    descending into any nested <table> that might live inside a cell."""
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

    Returns (cells, n_rows, n_cols, footnotes).
    Each cell dict: {row, col, text, is_header, [colspan], [rowspan]}
    Cells merged INTO by a spanning cell are not emitted again — only the
    originating (top-left) cell is recorded, carrying its span size.
    """
    cells: List[dict] = []
    footnotes: List[dict] = []
    occupied: dict = {}  # (row, col) -> True

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


# ---------------------------------------------------------------------------
# Splitting a table wrapper into logical tables (handles the rare
# multi-figcaption merged-wrapper quirk)
# ---------------------------------------------------------------------------

def _split_into_logical_groups(table_fig: Tag) -> List[Tuple[Tag, List[Tag]]]:
    """
    Most `<figure class="ltx_table">` wrappers contain exactly one logical
    table: a single <figcaption> followed by either a <table> or a
    <div class="ltx_flex_figure"> of panels.

    Occasionally one wrapper contains SEVERAL logical tables back to back
    (multiple <figcaption> elements as direct children, each followed by
    its own content). This splits the wrapper's direct children into
    (figcaption, [content_tags]) groups so each logical table is handled
    independently instead of being merged/duplicated.
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


# ---------------------------------------------------------------------------
# Top-level table parsing
# ---------------------------------------------------------------------------

def parse_tables(soup: BeautifulSoup, paper_id: str) -> List[dict]:
    """
    Find every table on the page and return one record per logical table
    (or per sub-panel, for multi-part tables like Table 14(a)/(b)).
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
        sub_cap, sub_fns = _outer_table_caption(panel)
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
        "table_id": f"{paper_id}_T{table_num}",
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


# ---------------------------------------------------------------------------
# In-text references to tables (mirrors parse_figure_references)
# ---------------------------------------------------------------------------

def parse_table_references(soup: BeautifulSoup) -> dict:
    """
    Return {table_num: [paragraph_text, ...]} for every logical table,
    scanning body paragraphs for mentions like 'Table 3', 'Tab. 3', 'Tab 3a'.
    """
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

    references: dict = {n: [] for n in table_numbers}

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
# Convenience: attach references onto each table record + write JSONL
# ---------------------------------------------------------------------------

def build_table_records(soup: BeautifulSoup, paper_id: str) -> List[dict]:
    """One-stop call: parses tables + references and merges them, ready
    to json.dumps one line per table."""
    records = parse_tables(soup, paper_id)
    refs_by_num = parse_table_references(soup)
    for rec in records:
        rec["references"] = refs_by_num.get(rec["table_num"], [])
    return records


def write_tables_jsonl(records: List[dict], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "a", encoding="utf-8") as fh:
        for rec in records:
            fh.write(json.dumps(rec, ensure_ascii=False) + "\n")


# ---------------------------------------------------------------------------
# Per-paper processing — fetches the page itself (same pattern as
# full_ar5ive_scraper.py's process_paper()), instead of requiring a soup
# to be handed in.
# ---------------------------------------------------------------------------

def process_paper(
    paper_id: str,
    year: str,
    month: str,
    output_dir: Path,
) -> bool:
    """
    Fetch a single arXiv paper's ar5iv HTML page, extract all table
    records, and append them to this month's JSONL output.

    Parameters
    ----------
    paper_id:
        The five-digit arXiv sequence number, e.g. ``"12325"``.
        Combined with *year* and *month* it forms the full ID ``YYMM.NNNNN``
        (e.g. ``"2410.12325"``).
    year:
        Two-digit year string, e.g. ``"25"``.
    month:
        Two-digit month string, e.g. ``"10"``.
    output_dir:
        Root directory for all output files.

    Returns
    -------
    bool
        True if the paper was found and processed, False otherwise
        (e.g. on a 404 — signals "no such paper" to the caller).
    """
    full_id = f"{year}{month}.{paper_id}"
    paper_url = ARXIV_HTML_BASE + full_id
    log.info("Processing %s …", paper_url)

    soup = fetch_soup(paper_url)
    if soup is None:
        return False

    table_records = build_table_records(soup, paper_id=full_id)

    tables_path = output_dir / "tables" / f"{year}_{month}.jsonl"
    write_tables_jsonl(table_records, tables_path)

    log.info("  ✓ %s – %d table record(s)", full_id, len(table_records))
    return True


# ---------------------------------------------------------------------------
# Batch scraping (identical structure to full_ar5ive_scraper.py)
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
    tables from each one until a 404 is returned (indicating no more
    papers exist for that month/year combination).

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
    log.info("=== Scraping tables for %s/%s (starting at %s.%05d) ===", year, month, year + month, start_id)
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


# ---------------------------------------------------------------------------
# Drop-in integration with full_ar5ive_scraper.py's process_paper()
# ---------------------------------------------------------------------------
#
# If you're already fetching `soup` inside full_ar5ive_scraper.py's own
# process_paper() (for figures) and don't want a second HTTP request just
# for tables, skip this module's process_paper()/scrape_month() and instead
# call build_table_records() directly on that existing soup:
#
#     from table_scraper import build_table_records, write_tables_jsonl
#
#     table_records = build_table_records(soup, paper_id=full_id)
#
#     tables_path = output_dir / "tables" / f"{year}_{month}.jsonl"
#     write_tables_jsonl(table_records, tables_path)
#
#     log.info("  tables: %d found for %s", len(table_records), full_id)
#
# Each line of tables/<year>_<month>.jsonl is one full table record
# (caption, cells, footnotes, references all in one JSON object) - JSONL
# rather than flat TSV, since a table's cell grid doesn't fit neatly into
# flat columns.
#
# If you'd rather keep references in their own TSV (mirroring the existing
# references/ref_<year>_<month>.tsv used for figures), flatten them out
# separately instead of embedding them in each record:
#
#     def write_table_references_tsv(records, path, full_id):
#         import csv
#         is_new = not path.exists()
#         path.parent.mkdir(parents=True, exist_ok=True)
#         with open(path, "a", newline="", encoding="utf-8") as fh:
#             w = csv.DictWriter(fh, fieldnames=["paper_id", "table_id", "reference_text"], delimiter="\t")
#             if is_new:
#                 w.writeheader()
#             for rec in records:
#                 for ref in rec["references"]:
#                     w.writerow({"paper_id": full_id, "table_id": rec["table_id"], "reference_text": ref})


# ---------------------------------------------------------------------------
# Quick manual test against a local HTML file
# ---------------------------------------------------------------------------


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------
# Run this script directly from the command line to scrape tables for a given month:
#   python3 ar5ive_scraper/table_scraper.py <year> <month>
# Example:
#   python3 ar5ive_scraper/table_scraper.py 24 10

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Scrape arXiv tables for a given month.")
    parser.add_argument("year", help="Two-digit year (e.g. 24 for 2024)")
    parser.add_argument("month", help="Two-digit month (e.g. 10 for October)")
    args = parser.parse_args()

    OUTPUT_ROOT = Path("/home/adah.holt/Desktop/arxiv_data")
    scrape_month(args.year, args.month, OUTPUT_ROOT)
    