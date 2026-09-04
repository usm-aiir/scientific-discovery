
"""
table_scraper.py
================
Scrapes arXiv HTML pages (via ar5iv) for tables: caption text, full cell
grid (headers + values, with rowspan/colspan preserved), in-text references
to each table, and any footnotes embedded in the caption or cells.
 
Designed to sit alongside ``arxiv_scraper.py`` and reuse the same
conventions. You can either run this script standalone, or import
``build_table_records()`` and call it from within your existing
``process_paper()`` to avoid a second HTTP request.
 
Output layout
-------------
<output_dir>/
└── tables/
    └── <year>_<month>.jsonl   one JSON object per table / sub-table panel
 
Output record shape
-------------------
Each line of the JSONL file is one dict::
 
    {
      "table_id":   "2410.00004_T1",     # <paper_id>_T<num>[<sub_id>]
      "html_id":    "S3.T1",             # raw HTML id attribute
      "table_num":  1,
      "sub_id":     null,                # "a" / "b" / ... for multi-panel tables
      "caption":    "...",
      "sub_caption": null,
      "n_rows":     7,
      "n_cols":     3,
      "cells": [
          {"row": 0, "col": 0, "text": "Model",  "is_header": true},
          {"row": 0, "col": 1, "text": "Recall", "is_header": true},
          {"row": 1, "col": 0, "text": "BERT",   "is_header": false},
          ...
          {"row": 3, "col": 0, "text": "0", "is_header": true, "rowspan": 4},
      ],
      "footnotes": [
          {"marker": "1", "text": "The vocabulary size of the original ..."}
      ],
      "references": [
          "Overall, adding neighbors at this stage (Table 1) ..."
      ]
    }
 
ar5iv quirks handled
--------------------
- **rowspan / colspan**: a grid-position algorithm tracks occupied cells so
  every cell gets correct (row, col) coordinates. Only the origin cell of
  a span is emitted; it carries ``"rowspan"`` / ``"colspan"`` so the dense
  grid can be reconstructed downstream.
- **Multi-panel tables**: handled the same way as subfigures in
  ``arxiv_scraper.py`` (e.g. Table 14(a) / 14(b)).
- **Nested tables in header cells**: flattened to plain text rather than
  being mis-parsed as a second table.
- **Footnotes**: ar5iv embeds footnote text inline inside
  ``ltx_note ltx_role_footnote`` spans. These are extracted into the
  ``footnotes`` list and replaced with a lightweight ``[^N]`` marker in
  the visible text.
- **Merged wrapper quirk**: occasionally one ``<figure class="ltx_table">``
  wraps several logical tables (multiple direct ``<figcaption>`` children).
  We split on figcaption boundaries so each logical table becomes its own
  record.
 
Usage
-----
Standalone::
 
    python table_scraper.py 24 10               # scrape all of October 2024
    python table_scraper.py 24 10 --max-papers 5
    python table_scraper.py 24 10 --start-id 200
 
Imported::
 
    from table_scraper import build_table_records, write_tables_jsonl
    table_records = build_table_records(soup, paper_id=full_id)
    write_tables_jsonl(table_records, output_dir / "tables" / f"{year}_{month}.jsonl")
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
# Configuration
# ---------------------------------------------------------------------------
 
ARXIV_HTML_BASE = "https://ar5iv.labs.arxiv.org/html/"
 
REQUEST_DELAY_SECONDS = 1.0
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
 
    Retries up to MAX_RETRIES times on transient HTTP/connection errors.
    Returns None immediately on 404 (paper does not exist), or after all
    retries are exhausted.
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
 
 
# ---------------------------------------------------------------------------
# Text, math, and footnote extraction
# ---------------------------------------------------------------------------
 
def _flatten_nested_table(table_tag: Tag) -> str:
    """
    Flatten a ``<table>`` that appears inside a header cell into a single
    readable string. ar5iv occasionally uses a tiny nested table to render
    a multi-line header (e.g. "Alpha on" / "Sequence" stacked vertically).
    """
    parts = []
    for cell in table_tag.find_all(["td", "th"]):
        txt = " ".join(cell.get_text().split())
        if txt:
            parts.append(txt)
    return " ".join(parts)
 
 
def extract_text_and_footnotes(element) -> Tuple[str, List[dict]]:
    """
    Walk *element* recursively, returning ``(clean_text, footnotes)`` where:
 
    - ``clean_text`` has math rendered as ``$...$`` and inline footnote
      spans replaced with lightweight ``[^N]`` markers.
    - ``footnotes`` is a list of ``{"marker": "N", "text": "..."}`` dicts
      pulled from any ``ltx_note ltx_role_footnote`` spans found inside
      *element* (these appear in both captions and cells in ar5iv HTML).
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
 
 
def _dedupe_footnotes(footnotes: List[dict]) -> List[dict]:
    """Remove duplicate footnotes (same marker + text) from a list."""
    seen = set()
    out = []
    for fn in footnotes:
        key = (fn["marker"], fn["text"])
        if key not in seen:
            seen.add(key)
            out.append(fn)
    return out
 
 
# ---------------------------------------------------------------------------
# Caption helpers
# ---------------------------------------------------------------------------
 
def _outer_table_caption(table_fig: Tag) -> Tuple[str, List[dict]]:
    """
    Extract the top-level caption from a table figure element.
 
    Returns ``(caption_text, footnotes)``. Only looks at direct
    ``<figcaption class="ltx_caption">`` children to avoid picking up
    sub-panel captions.
    """
    for tag in table_fig.children:
        if isinstance(tag, Tag) and tag.name == "figcaption" and "ltx_caption" in tag.get("class", []):
            return extract_text_and_footnotes(tag)
    return "", []
 
 
def split_subcaptions(caption: str) -> List[Tuple[str, str]]:
    """
    Split a compound caption into labeled sub-panel entries.
 
    Example input:  ``"Table 1: (a) First panel. (b) Second panel."``
    Example output: ``[("a", "First panel."), ("b", "Second panel.")]``
 
    Returns an empty list if no sub-panel labels are found.
    """
    pattern = re.compile(r"\(([a-z])\)\s*([^)]*?)(?=\s*\([a-z]\)|$)")
    matches = pattern.findall(caption)
    return [(letter, text.strip()) for letter, text in matches]
 
 
# ---------------------------------------------------------------------------
# Grid extraction (handles thead/tbody, rowspan, colspan, nested tables)
# ---------------------------------------------------------------------------
 
def _iter_top_level_rows(table_tag: Tag) -> List[Tag]:
    """
    Return this table's own ``<tr>`` elements in document order without
    descending into any nested ``<table>`` that might live inside a cell.
    """
    rows: List[Tag] = []
    for child in table_tag.find_all(["thead", "tbody", "tfoot", "tr"], recursive=False):
        if child.name == "tr":
            rows.append(child)
        else:
            rows.extend(child.find_all("tr", recursive=False))
    return rows
 
 
def extract_table_grid(table_tag: Tag) -> Tuple[List[dict], int, int, List[dict]]:
    """
    Build the full cell grid for *table_tag*, tracking occupied positions
    so each cell gets correct ``(row, col)`` coordinates even when
    rowspan/colspan is used.
 
    Returns ``(cells, n_rows, n_cols, footnotes)``.
 
    Each cell dict contains ``row``, ``col``, ``text``, ``is_header``, and
    optionally ``rowspan`` / ``colspan`` when greater than 1. Only the
    origin (top-left) cell of a span is emitted — cells merged into it are
    not duplicated.
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
 
            cell: dict = {
                "row":       row_idx,
                "col":       col_idx,
                "text":      text,
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
 
 
# ---------------------------------------------------------------------------
# Splitting a wrapper into logical tables
# (handles the rare multi-figcaption merged-wrapper quirk)
# ---------------------------------------------------------------------------
 
def _split_into_logical_groups(table_fig: Tag) -> List[Tuple[Tag, List[Tag]]]:
    """
    Split a ``<figure class="ltx_table">`` wrapper into
    ``(figcaption, [content_tags])`` groups.
 
    Most wrappers contain exactly one logical table. Occasionally one
    wrapper contains several back-to-back logical tables (multiple direct
    ``<figcaption>`` children). This function handles both cases uniformly
    so each logical table is processed independently.
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
    """Find sub-panel figure elements within a multi-part table."""
    panels: List[Tag] = []
    for tag in content_tags:
        if tag.name == "figure" and "ltx_figure_panel" in tag.get("class", []):
            panels.append(tag)
        else:
            panels.extend(tag.find_all("figure", class_="ltx_figure_panel"))
    return panels
 
 
def _find_direct_table(content_tags: List[Tag]) -> Optional[Tag]:
    """Find the first ``<table>`` element within a list of content tags."""
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
    (or per sub-panel for multi-part tables like Table 14(a)/(b)).
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
    records: List[dict],
    panels: List[Tag],
    outer_caption: str,
    outer_footnotes: List[dict],
    table_num: int,
    wrapper_html_id: str,
    paper_id: str,
) -> None:
    """Build and append one record per sub-panel of a multi-part table."""
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
        panel_captions = [text for _, text in sub_parts[: len(panels)]]
        panel_sub_ids = [letter for letter, _ in sub_parts[: len(panels)]]
        for i, (label, _) in enumerate(sub_parts):
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
        sub_id     = panel_sub_ids[idx] if idx < len(panel_sub_ids) else chr(ord("a") + idx)
        sub_caption = panel_captions[idx] if idx < len(panel_captions) else ""
        fns = list(outer_footnotes)
        fns.extend(panel_footnotes_list[idx] if idx < len(panel_footnotes_list) else [])
        fns.extend(cell_footnotes)
 
        records.append({
            "table_id":    f"{paper_id}_T{table_num}{sub_id}",
            "html_id":     panel.get("id", wrapper_html_id),
            "table_num":   table_num,
            "sub_id":      sub_id,
            "caption":     outer_caption,
            "sub_caption": sub_caption,
            "n_rows":      n_rows,
            "n_cols":      n_cols,
            "cells":       cells,
            "footnotes":   _dedupe_footnotes(fns),
        })
 
 
def _append_single_record(
    records: List[dict],
    table_tag: Tag,
    outer_caption: str,
    outer_footnotes: List[dict],
    table_num: int,
    wrapper_html_id: str,
    paper_id: str,
) -> None:
    """Build and append one record for a simple (non-panel) table."""
    cells, n_rows, n_cols, cell_footnotes = extract_table_grid(table_tag)
    fns = list(outer_footnotes)
    fns.extend(cell_footnotes)
 
    records.append({
        "table_id":    f"{paper_id}_T{table_num}",
        "html_id":     wrapper_html_id,
        "table_num":   table_num,
        "sub_id":      None,
        "caption":     outer_caption,
        "sub_caption": None,
        "n_rows":      n_rows,
        "n_cols":      n_cols,
        "cells":       cells,
        "footnotes":   _dedupe_footnotes(fns),
    })
 
 
# ---------------------------------------------------------------------------
# In-text references to tables
# ---------------------------------------------------------------------------
 
def parse_table_references(soup: BeautifulSoup) -> dict:
    """
    Return ``{table_num: [paragraph_text, ...]}`` for every logical table,
    scanning body paragraphs for mentions like ``"Table 3"``, ``"Tab. 3"``,
    or ``"Tab 3a"``.
    """
    table_numbers: List[int] = []
    sequential_idx = 0
 
    top_level_tables = [
        fig for fig in soup.find_all("figure", class_="ltx_table")
        if "ltx_figure_panel" not in fig.get("class", [])
    ]
    for table_fig in top_level_tables:
        for figcaption_tag, _ in _split_into_logical_groups(table_fig):
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
# Public API
# ---------------------------------------------------------------------------
 
def build_table_records(soup: BeautifulSoup, paper_id: str) -> List[dict]:
    """
    Parse all tables from *soup* and attach in-text references to each record.
 
    This is the main entry point when importing this module from another
    script that already has a ``soup`` object (e.g. ``arxiv_scraper.py``'s
    ``process_paper()``), so you avoid making a second HTTP request.
 
    Returns a list of table record dicts ready to be written with
    ``write_tables_jsonl()``.
    """
    records = parse_tables(soup, paper_id)
    refs_by_num = parse_table_references(soup)
    for rec in records:
        rec["references"] = refs_by_num.get(rec["table_num"], [])
    return records
 
 
def write_tables_jsonl(records: List[dict], path: Path) -> None:
    """
    Append *records* to a JSONL file at *path* (one JSON object per line).
    Creates parent directories as needed.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "a", encoding="utf-8") as fh:
        for rec in records:
            fh.write(json.dumps(rec, ensure_ascii=False) + "\n")
 
 
# ---------------------------------------------------------------------------
# Per-paper and batch processing
# ---------------------------------------------------------------------------
 
def process_paper(
    paper_id: str,
    year: str,
    month: str,
    output_dir: Path,
) -> bool:
    """
    Fetch a single arXiv paper's ar5iv HTML page, extract all table records,
    and append them to this month's JSONL output file.
 
    Parameters
    ----------
    paper_id:
        Zero-padded five-digit sequence number, e.g. ``"12325"``.
        Combined with *year* and *month* it forms the full ID ``YYMM.NNNNN``.
    year:
        Two-digit year string, e.g. ``"24"``.
    month:
        Two-digit month string, e.g. ``"10"``.
    output_dir:
        Root directory for all output files.
 
    Returns
    -------
    bool
        True if the paper was found and processed, False otherwise.
    """
    full_id   = f"{year}{month}.{paper_id}"
    paper_url = ARXIV_HTML_BASE + full_id
    log.info("Processing %s …", paper_url)
 
    soup = fetch_soup(paper_url)
    if soup is None:
        return False
 
    table_records = build_table_records(soup, paper_id=full_id)
    write_tables_jsonl(table_records, output_dir / "tables" / f"{year}_{month}.jsonl")
 
    log.info("  ✓ %s – %d table record(s)", full_id, len(table_records))
    return True
 
 
def scrape_month(
    year: str,
    month: str,
    output_dir: Path,
    max_papers: Optional[int] = None,
    start_id: int = 1,
) -> None:
    """
    Scrape tables from all papers for a given *year* / *month* in sequence.
 
    Stops when a paper is not found (end of that month's submissions) or
    when *max_papers* papers have been successfully processed.
 
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
    log.info(
        "=== Scraping tables for %s/%s (starting at %s.%05d) ===",
        year, month, year + month, start_id,
    )
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
        description="Scrape arXiv tables and metadata for a given month.",
        epilog="Example: python table_scraper.py 24 10",
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