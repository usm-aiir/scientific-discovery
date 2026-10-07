"""
arxiv_scraper.py
=================

One combined scraper for arXiv (ar5iv) papers. Replaces the old
scraper_functions.py + all_scraper.py split.

For each paper, output is spread across five subfolders of the output
directory (default "<year>_<month>", e.g. "25_04/"):

    <output_dir>/
        figures/
            images/
                <paper_id>_<figure_id>[_<sub_id>].<ext>   # downloaded images
            <paper_id>.json      # {"figures": [...], "figure_references": {...}}
        tables/
            <paper_id>.json      # {"tables": [...], "table_references": {...}}
        HTMLs/
            <paper_id>.html      # raw ar5iv HTML
        paragraphs/
            <paper_id>.json      # {"paragraphs": {"<paper_id>_1": "...", ...}}
        metadata/
            <paper_id>.json      # title, abstract, categories, counts, url

Each figure row in figures/<paper_id>.json carries "image_id" (matches the
downloaded filename, minus extension) and "local_path" (relative to the
figures/ folder, i.e. "images/<file>") so you can join the JSON row to the
file on disk.

What changed vs. the old scraper
---------------------------------
1. Output is split into figures/tables/HTMLs/paragraphs/metadata folders
   instead of one combined per-paper JSON, and figure images are actually
   downloaded (previously only the remote URL was recorded).
2. Tables are extracted just like figures: caption, sub-caption (for
   sub-tables), and the actual cell text (with LaTeX-ish math).
3. "table_references" mirrors "figure_references": for every table/figure,
   every paragraph in the body text that mentions it ("Table 3", "Tab. 3",
   "Figure 3a", ...) is collected, so you always know where it's discussed
   even if it isn't right next to it in the document.
4. Paragraph extraction no longer just grabs every <p class="ltx_p"> in
   isolation and no longer sweeps in figure/table captions or the abstract.
   Fragments that don't look like a finished sentence -- ending in ':',
   ';', ',', very short (<= 6 words), or missing terminal punctuation
   altogether -- are merged forward into the next fragment (and repeatedly,
   until a real sentence ending is found), which is what you need when
   ar5iv splits a sentence across a dropped display equation, e.g.
   "...can be decomposed as" / "with equality if ..." get stitched back
   into one paragraph. See _is_incomplete_fragment for the exact rules.
5. Only <figure class="ltx_figure"> elements (and sub-panels) that contain
   a real <img src="..."> are treated as figures. ar5iv also uses
   ltx_figure for text boxes, pseudo-code listings, and empty layout
   wrappers; those are now skipped entirely -- not downloaded, not written
   to the figures JSON, and not given a "figure_references" entry.

Usage
-----
    python3 arxiv_scraper.py <year> <month> [--max_papers N] [--start_id N]
                              [--output_dir DIR] [--delay SECONDS]

Example
-------
    python3 arxiv_scraper.py 2025 04    # scrape ALL papers from April 2025
                                         # -> saved under 25_04/
"""

from __future__ import annotations

import argparse
import json
import logging
import re
import time
import xml.etree.ElementTree as ET
from pathlib import Path
from typing import Optional, List, Tuple
from urllib.parse import urljoin

import requests
from bs4 import BeautifulSoup, NavigableString, Tag

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

ARXIV_HTML_BASE = "https://ar5iv.labs.arxiv.org/html/"

REQUEST_DELAY_SECONDS = 1.0

MAX_RETRIES = 3
RETRY_BACKOFF_SECONDS = 5.0

# ar5iv sometimes serves a tiny stub/placeholder page for a given ID (e.g. a
# withdrawn submission, or a conversion failure) instead of a 404. Anything
# smaller than this is treated as "nothing worth recording" and skipped --
# but note this is NOT the same as "paper doesn't exist": scraping continues
# to the next ID either way, only a real 404 stops the month's scan.
MIN_HTML_SIZE_BYTES = 15 * 1024  # 15 KB

# A paragraph fragment this short (in words) is almost never a complete
# thought on its own -- it's a heading-like lead-in ("As follows:"), a
# stray caption remnant, or a line break artifact. Merge it with its
# neighbor instead of keeping it as a standalone "paragraph".
SHORT_FRAGMENT_MAX_WORDS = 6

# Characters that genuinely end a sentence/clause-thought when they are the
# LAST substantive character. Includes normal terminal punctuation plus the
# proof/QED marks papers use to close an argument.
SENTENCE_END_CHARS = {".", "!", "?", "\u220e", "\u25a1", "\u25a0", "\u25fb"}  # . ! ? ∎ □ ■ ◻

# Trailing "wrapper" characters to peel off before checking for a real
# sentence-ending character underneath -- e.g. a quote or closing math
# delimiter right after the period ('$P(X<x)=0$.' , 'the result."').
_TRAILING_WRAP_CHARS = "\"')]}$*~`\u201d\u2019\u203a\u00bb"

# Characters that, when they are literally the last character, always mean
# "there's more coming" -- a colon/semicolon introduces what follows, and a
# trailing comma (very common right before a dropped display equation, e.g.
# "...decomposed as," or "Second,") means the clause continues.
_ALWAYS_CONTINUES_CHARS = (":", ";", ",")

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Networking helpers
# ---------------------------------------------------------------------------

def fetch_soup(url: str) -> Optional[Tuple[BeautifulSoup, int]]:
    """
    Download *url* and return (BeautifulSoup parse tree, raw HTML size in
    bytes).

    Retries up to MAX_RETRIES times on transient HTTP errors (5xx) or
    connection problems. Returns None if every attempt fails (or on a
    404, which means the paper doesn't exist).
    """
    for attempt in range(1, MAX_RETRIES + 1):
        try:
            response = requests.get(url, timeout=30)
            response.raise_for_status()
            return BeautifulSoup(response.text, "html.parser"), len(response.content)

        except requests.exceptions.HTTPError as exc:
            # NOTE: must be "is not None" -- a requests.Response is falsy
            # for any 4xx/5xx status, so "if exc.response" would always
            # fall through to "?" and the 404 check would never fire.
            status = exc.response.status_code if exc.response is not None else "?"
            if status == 404:
                log.debug("404 - paper does not exist: %s", url)
                return None  # not a transient error; stop retrying
            log.warning("HTTP %s on attempt %d/%d for %s", status, attempt, MAX_RETRIES, url)

        except requests.exceptions.RequestException as exc:
            log.warning("Request error on attempt %d/%d for %s: %s", attempt, MAX_RETRIES, url, exc)

        if attempt < MAX_RETRIES:
            time.sleep(RETRY_BACKOFF_SECONDS)

    log.error("All %d attempts failed for %s", MAX_RETRIES, url)
    return None


# ---------------------------------------------------------------------------
# Low-level text / math extraction helpers
# ---------------------------------------------------------------------------

def extract_text_with_math(element) -> str:
    """Depth-first text extraction that renders <math alttext="..."> as $...$
    instead of dropping/mangling it, so LaTeX survives in the plain text."""
    if isinstance(element, NavigableString):
        return str(element)
    if element.name == "math":
        alt = element.get("alttext", "")
        return f"${alt}$" if alt else element.get_text()
    return "".join(extract_text_with_math(child) for child in element.children)


def _resolve_image_url(raw_src: str, paper_url: str) -> str:
    """Return an absolute URL for *raw_src*, resolving relative paths against
    *paper_url*. Works correctly for ar5iv's absolute paths (starting with
    '/html/...')."""
    return urljoin(paper_url, raw_src)


def _outer_caption(tag: Tag) -> str:
    """Grab the direct <figcaption class="ltx_caption"> of a <figure>-like tag
    (used for both figures and tables -- ar5iv marks both the same way)."""
    for child in tag.children:
        if getattr(child, "name", None) == "figcaption" and "ltx_caption" in child.get("class", []):
            raw = extract_text_with_math(child)
            return " ".join(raw.split())
    return ""


def split_subcaptions(caption: str) -> List[Tuple[str, str]]:
    """
    Split a compound caption like "Figure 1: (a) First panel (b) Second
    panel." into a list of (label, text) for each sub-panel.
    """
    pattern = re.compile(r"\(([a-z])\)\s*([^)]*?)(?=\s*\([a-z]\)|$)")
    matches = pattern.findall(caption)
    return [(letter, text.strip()) for letter, text in matches]


# ---------------------------------------------------------------------------
# Title / abstract
# ---------------------------------------------------------------------------

def parse_title_abstract(soup: BeautifulSoup) -> Tuple[str, str]:
    """Extract the paper title and abstract text. Either value is an empty
    string when the expected element cannot be found."""
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


# ---------------------------------------------------------------------------
# Figures
# ---------------------------------------------------------------------------

def _find_real_img(tag: Tag) -> Optional[Tag]:
    """Return the first <img> inside *tag* that has a non-empty src, or None.

    This is the "is this actually a figure?" test. ar5iv reuses
    <figure class="ltx_figure"> for text boxes, pseudo-code listings and
    empty layout wrappers; none of those carry an image, so anything that
    fails this check is not treated as a figure."""
    for img in tag.find_all("img"):
        if (img.get("src") or "").strip():
            return img
    return None


def _top_level_image_figures(soup: BeautifulSoup) -> List[Tag]:
    """Top-level <figure class="ltx_figure"> elements that contain a real
    image. Shared by parse_figures and parse_figure_references so both see
    the exact same set of figures (and the same fallback numbering)."""
    return [
        fig for fig in soup.find_all("figure", class_="ltx_figure")
        if "ltx_figure_panel" not in fig.get("class", [])
        and _find_real_img(fig) is not None
    ]


def _figure_id(fig: Tag, sequential_idx: int) -> int:
    """'Figure N' number from the caption, else the sequential position."""
    match = re.search(r"\bFigure\s+(\d+)", _outer_caption(fig), re.IGNORECASE)
    return int(match.group(1)) if match else sequential_idx


def parse_figures(soup: BeautifulSoup, paper_url: str) -> List[dict]:
    """Extract every top-level figure (and its sub-panels, if any) as one row
    per panel: {figure_id, sub_id, source, caption, sub_caption}.

    Figures and sub-panels without a real <img> are skipped."""
    rows: List[dict] = []

    for sequential_idx, fig in enumerate(_top_level_image_figures(soup), start=1):
        outer_caption = _outer_caption(fig)
        fig_id = _figure_id(fig, sequential_idx)

        panels = fig.find_all("figure", class_="ltx_figure_panel")

        if panels:
            # Sub-ids/captions are worked out over ALL panels first, so the
            # (a)/(b)/(c) lettering still lines up with the caption even if
            # some panels are dropped below for having no image.
            panel_captions = [_outer_caption(panel) for panel in panels]

            if all(c == "" for c in panel_captions):
                sub_parts = split_subcaptions(outer_caption)
                if sub_parts:
                    while len(sub_parts) < len(panels):
                        sub_parts.append(("", ""))
                    sub_parts = sub_parts[:len(panels)]
                    panel_captions = [text for _, text in sub_parts]
                    panel_sub_ids = [
                        letter if letter else chr(ord("a") + i)
                        for i, (letter, _) in enumerate(sub_parts)
                    ]
                else:
                    panel_sub_ids = [chr(ord("a") + i) for i in range(len(panels))]
                    panel_captions = ["" for _ in panels]
            else:
                panel_sub_ids = []
                for sub in panel_captions:
                    sub_match = re.search(r"\(([a-z])\)", sub, re.IGNORECASE)
                    panel_sub_ids.append(sub_match.group(1).lower() if sub_match else None)

            for idx, panel in enumerate(panels):
                img_tag = _find_real_img(panel)
                if img_tag is None:
                    # Text box / pseudo-code / empty panel -- not a figure.
                    continue
                rows.append({
                    "figure_id": fig_id,
                    "sub_id": panel_sub_ids[idx] if idx < len(panel_sub_ids) else None,
                    "source": _resolve_image_url(img_tag["src"], paper_url),
                    "caption": outer_caption,
                    "sub_caption": panel_captions[idx] if idx < len(panel_captions) else "",
                })
        else:
            img_tag = _find_real_img(fig)  # guaranteed non-None by the filter
            rows.append({
                "figure_id": fig_id,
                "sub_id": None,
                "source": _resolve_image_url(img_tag["src"], paper_url),
                "caption": outer_caption,
                "sub_caption": None,
            })

    return rows


# ---------------------------------------------------------------------------
# Tables
# ---------------------------------------------------------------------------

def _extract_table_rows(table_tag: Tag) -> List[List[str]]:
    """Turn an HTML <table> into a list of rows, each a list of cell texts
    (math-aware, whitespace-normalised). Header/body distinction is not
    tracked separately -- ar5iv usually marks header cells with <th> but
    downstream consumers rarely need that split for text-only use, so all
    rows are returned in document order."""
    rows: List[List[str]] = []
    for tr in table_tag.find_all("tr"):
        cells = tr.find_all(["td", "th"])
        if not cells:
            continue
        row = [" ".join(extract_text_with_math(cell).split()) for cell in cells]
        rows.append(row)
    return rows


def parse_tables(soup: BeautifulSoup, paper_url: str) -> List[dict]:
    """Extract every top-level table (and its sub-tables, if any) as one row
    per sub-table: {table_id, sub_id, rows, caption, sub_caption}.

    ar5iv wraps tables the same way it wraps figures: a <figure
    class="ltx_table"> holding a <figcaption class="ltx_caption"> plus one
    or more <table> elements (sub-tables use class "ltx_table_panel", just
    like figure sub-panels)."""
    rows_out: List[dict] = []
    sequential_idx = 0

    top_level_tables = [
        tbl for tbl in soup.find_all("figure", class_="ltx_table")
        if "ltx_table_panel" not in tbl.get("class", [])
    ]

    for tbl in top_level_tables:
        sequential_idx += 1
        outer_caption = _outer_caption(tbl)

        num_match = re.search(r"\bTable\s+(\d+)", outer_caption, re.IGNORECASE)
        table_id = int(num_match.group(1)) if num_match else sequential_idx

        # Sub-tables can appear either as nested <figure class="ltx_table_panel">
        # (mirroring figure sub-panels) or, more commonly for tables, as several
        # sibling <table> elements directly inside the same <figure class="ltx_table">.
        panels = tbl.find_all("figure", class_="ltx_table_panel")

        if panels:
            panel_captions = [_outer_caption(panel) for panel in panels]
            if all(c == "" for c in panel_captions):
                sub_parts = split_subcaptions(outer_caption)
                if sub_parts:
                    while len(sub_parts) < len(panels):
                        sub_parts.append(("", ""))
                    sub_parts = sub_parts[:len(panels)]
                    panel_captions = [text for _, text in sub_parts]
                    panel_sub_ids = [
                        letter if letter else chr(ord("a") + i)
                        for i, (letter, _) in enumerate(sub_parts)
                    ]
                else:
                    panel_sub_ids = [chr(ord("a") + i) for i in range(len(panels))]
                    panel_captions = ["" for _ in panels]
            else:
                panel_sub_ids = []
                for sub in panel_captions:
                    sub_match = re.search(r"\(([a-z])\)", sub, re.IGNORECASE)
                    panel_sub_ids.append(sub_match.group(1).lower() if sub_match else None)

            for idx, panel in enumerate(panels):
                inner_table = panel.find("table")
                table_rows = _extract_table_rows(inner_table) if inner_table else []
                rows_out.append({
                    "table_id": table_id,
                    "sub_id": panel_sub_ids[idx] if idx < len(panel_sub_ids) else None,
                    "rows": table_rows,
                    "caption": outer_caption,
                    "sub_caption": panel_captions[idx] if idx < len(panel_captions) else "",
                })
            continue

        sibling_tables = tbl.find_all("table")
        if len(sibling_tables) > 1:
            # Multiple <table> elements sharing one caption -- treat as
            # lettered sub-tables (a), (b), ... in document order.
            for idx, inner_table in enumerate(sibling_tables):
                rows_out.append({
                    "table_id": table_id,
                    "sub_id": chr(ord("a") + idx),
                    "rows": _extract_table_rows(inner_table),
                    "caption": outer_caption,
                    "sub_caption": "",
                })
        else:
            inner_table = sibling_tables[0] if sibling_tables else None
            rows_out.append({
                "table_id": table_id,
                "sub_id": None,
                "rows": _extract_table_rows(inner_table) if inner_table else [],
                "caption": outer_caption,
                "sub_caption": None,
            })

    return rows_out


# ---------------------------------------------------------------------------
# Cross-references ("where is Figure N / Table N discussed")
# ---------------------------------------------------------------------------

def _body_paragraph_texts(soup: BeautifulSoup) -> List[str]:
    """All ltx_p paragraph texts that live in the paper body -- i.e. NOT
    inside a figure/table (so captions and table cells don't get treated as
    "body text that mentions this figure") and NOT inside the abstract
    (which is already captured separately)."""
    texts: List[str] = []
    for para in soup.find_all("p", class_="ltx_p"):
        if para.find_parent("figure") is not None:
            continue
        if para.find_parent("div", class_="ltx_abstract") is not None:
            continue
        text = " ".join(extract_text_with_math(para).split())
        if text:
            texts.append(text)
    return texts


def _build_reference_map(
    item_ids: List[int],
    body_paragraphs: List[str],
    label: str,
    abbrev_pattern: str,
) -> dict:
    """
    Generic "which paragraphs mention item N" builder, shared by figures and
    tables.

    label:           display label, e.g. "Figure" or "Table"
    abbrev_pattern:  regex alternation for how the label may be abbreviated
                      in running text, e.g. r"fig(?:ure)?" or r"tab(?:le)?"
    """
    references: dict = {f"{label} {item_id}": [] for item_id in item_ids}
    for item_id in item_ids:
        key = f"{label} {item_id}"
        pattern = re.compile(
            rf"\b{abbrev_pattern}\.?\s*{re.escape(str(item_id))}(?:[a-z]|\([a-z]\)|-[a-z])?\b",
            re.IGNORECASE,
        )
        for text in body_paragraphs:
            if pattern.search(text) and text not in references[key]:
                references[key].append(text)
    return references


def parse_figure_references(soup: BeautifulSoup) -> dict:
    # Same image-only filter (and same numbering) as parse_figures, so text
    # boxes / pseudo-code don't get a "Figure N" entry here either.
    figure_ids = [
        _figure_id(fig, seq_idx)
        for seq_idx, fig in enumerate(_top_level_image_figures(soup), start=1)
    ]
    body_paragraphs = _body_paragraph_texts(soup)
    return _build_reference_map(figure_ids, body_paragraphs, "Figure", r"fig(?:ure)?")


def parse_table_references(soup: BeautifulSoup) -> dict:
    top_level_tables = [
        tbl for tbl in soup.find_all("figure", class_="ltx_table")
        if "ltx_table_panel" not in tbl.get("class", [])
    ]
    table_ids: List[int] = []
    for seq_idx, tbl in enumerate(top_level_tables, start=1):
        caption = _outer_caption(tbl)
        match = re.search(r"\bTable\s+(\d+)", caption, re.IGNORECASE)
        table_ids.append(int(match.group(1)) if match else seq_idx)

    body_paragraphs = _body_paragraph_texts(soup)
    return _build_reference_map(table_ids, body_paragraphs, "Table", r"tab(?:le|\.)?")


# ---------------------------------------------------------------------------
# Categories (via the arXiv API, since ar5iv HTML doesn't carry them)
# ---------------------------------------------------------------------------

def parse_categories(soup: BeautifulSoup, paper_id: str) -> List[str]:
    """Extract arXiv subject categories by cross-referencing the official
    arXiv API (categories aren't present in the ar5iv HTML body)."""
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
    except Exception as e:
        log.warning("arXiv API metadata fallback query failed: %s", e)

    return categories


# ---------------------------------------------------------------------------
# Paragraphs (with incomplete-fragment merging)
# ---------------------------------------------------------------------------

def _is_incomplete_fragment(text: str) -> bool:
    """
    A paragraph fragment is treated as "the sentence hasn't ended yet" if
    ANY of the following hold:

      1. It's very short (<= SHORT_FRAGMENT_MAX_WORDS words) -- even if it's
         grammatically a full sentence (e.g. "See below.", "Thanks."), short
         fragments like this are usually dangling references or asides that
         read better attached to a neighboring paragraph than standing
         alone. This is a deliberate trade-off: a few genuinely standalone
         short paragraphs will also get merged. Tune
         SHORT_FRAGMENT_MAX_WORDS down (or drop this clause) if that's too
         aggressive for your use case.

      2. It ends in ':', ';', or ',' -- almost always a lead-in to a list,
         an equation, or a clause that continues in the next fragment. This
         is extremely common right where a display equation was stripped
         out of the running text (e.g. "...can be decomposed as," or
         "Second,").

      3. It does NOT end with real sentence-ending punctuation once you
         peel off trailing quotes/brackets/math delimiters (see
         _TRAILING_WRAP_CHARS). This is the general-purpose check: a proper,
         finished sentence ends in '.', '!', '?', or a proof mark like '∎' --
         anything else ("...Jensen's inequality have", "...simplifies to",
         "...decomposed as") means the text just stops mid-thought, almost
         always because the equation/table that continued the sentence
         wasn't itself a <p> and got parsed out separately.

    Note: because ar5iv/LaTeXML often renders a display equation as its own
    element (not inside the surrounding <p>), the sentence that leads into
    it and the clause that follows it are two separate <p class="ltx_p">
    fragments even though they're one sentence with the equation missing
    from the middle. This function detects that split via the "does this
    look finished" check above and merge_incomplete_paragraphs stitches the
    surrounding text back together -- the equation content itself is still
    not recovered, only the prose around it is reunited.
    """
    text = text.strip()
    if not text:
        return True

    word_count = len(text.split())
    if word_count <= SHORT_FRAGMENT_MAX_WORDS:
        return True

    if text[-1] in _ALWAYS_CONTINUES_CHARS:
        return True

    stripped_tail = text.rstrip(_TRAILING_WRAP_CHARS)
    if not stripped_tail:
        return True
    if stripped_tail[-1] not in SENTENCE_END_CHARS:
        return True

    return False


def merge_incomplete_paragraphs(paragraphs: List[str]) -> List[str]:
    """
    Merge paragraph fragments that look incomplete (see
    _is_incomplete_fragment) into a neighboring paragraph, so that
    downstream consumers get whole thoughts instead of "As shown below:"
    as its own paragraph.

    Fragments are merged FORWARD into the next paragraph by default (a
    colon/semicolon or a short lead-in usually introduces what follows).
    If the merging leaves a trailing fragment with nothing after it (it was
    the last paragraph in the document), it's merged BACKWARD into the
    previous paragraph instead.
    """
    if not paragraphs:
        return []

    merged: List[str] = []
    current = paragraphs[0]
    for nxt in paragraphs[1:]:
        if _is_incomplete_fragment(current):
            current = current.rstrip() + " " + nxt.lstrip()
        else:
            merged.append(current)
            current = nxt
    merged.append(current)

    # The very last paragraph has no "next" to merge into -- if it's still
    # incomplete, fold it backward into the previous one instead.
    while len(merged) > 1 and _is_incomplete_fragment(merged[-1]):
        tail = merged.pop()
        merged[-1] = merged[-1].rstrip() + " " + tail.lstrip()

    return merged


def extract_paragraphs(soup: BeautifulSoup) -> List[str]:
    """
    Extract the text of every body paragraph (<p class="ltx_p">), in
    document order, with math converted to LaTeX-ish text and whitespace
    normalised, EXCLUDING paragraphs that live inside a figure/table
    (those belong to captions/table cells, not the running text) or inside
    the abstract (captured separately). Incomplete fragments -- see
    merge_incomplete_paragraphs -- are stitched back together.
    """
    raw_paragraphs = _body_paragraph_texts(soup)
    return merge_incomplete_paragraphs(raw_paragraphs)


# ---------------------------------------------------------------------------
# Per-paper processing
# ---------------------------------------------------------------------------

def _guess_image_extension(url: str) -> str:
    """Best-effort file extension from the image URL; falls back to .png."""
    suffix = Path(url.split("?")[0]).suffix
    return suffix if suffix else ".png"


def download_image(url: str, dest_path: Path) -> bool:
    """Download the image at *url* to *dest_path*. Returns True on success."""
    for attempt in range(1, MAX_RETRIES + 1):
        try:
            response = requests.get(url, timeout=30)
            response.raise_for_status()
            dest_path.write_bytes(response.content)
            return True
        except requests.exceptions.HTTPError as exc:
            status = exc.response.status_code if exc.response is not None else "?"
            log.warning("HTTP %s downloading image (attempt %d/%d): %s", status, attempt, MAX_RETRIES, url)
        except requests.exceptions.RequestException as exc:
            log.warning("Error downloading image (attempt %d/%d) %s: %s", attempt, MAX_RETRIES, url, exc)
        if attempt < MAX_RETRIES:
            time.sleep(RETRY_BACKOFF_SECONDS)
    log.error("Giving up downloading image after %d attempts: %s", MAX_RETRIES, url)
    return False


# ---------------------------------------------------------------------------
# Output layout
# ---------------------------------------------------------------------------
#
#   <output_dir>/
#       figures/
#           images/
#               <paper_id>_<figure_id>[_<sub_id>].<ext>
#           <paper_id>.json          # figures list + figure_references
#       tables/
#           <paper_id>.json          # tables list + table_references
#       HTMLs/
#           <paper_id>.html
#       paragraphs/
#           <paper_id>.json          # {"<paper_id>_1": "...", ...}
#       metadata/
#           <paper_id>.json          # title, abstract, categories, counts, url


def _ensure_output_subdirs(output_dir: Path) -> dict:
    """Create (if needed) and return the paths of the five output subfolders."""
    subdirs = {
        "figures": output_dir / "figures",
        "images": output_dir / "figures" / "images",
        "tables": output_dir / "tables",
        "html": output_dir / "HTMLs",
        "paragraphs": output_dir / "paragraphs",
        "metadata": output_dir / "metadata",
    }
    for path in subdirs.values():
        path.mkdir(parents=True, exist_ok=True)
    return subdirs


def process_paper(
    paper_id: str,
    year: str,
    month: str,
    output_dir: Optional[Path] = None,
) -> bool:
    """
    Scrape a single arXiv paper and save its data across five subfolders of
    *output_dir* (created if needed): figures/ (+ figures/images/), tables/,
    HTMLs/, paragraphs/, metadata/ -- see the layout diagram above
    process_paper for exact paths.

    Parameters
    ----------
    paper_id: the five-digit arXiv sequence number, e.g. "12325".
              Combined with *year* and *month* -> "YYMM.NNNNN".
    year:     two-digit year string, e.g. "25".
    month:    two-digit month string, e.g. "10".
    output_dir: base directory for the five subfolders (current dir if None).

    Returns
    -------
    bool: True if a paper was found at this ID (whether or not its data was
    actually saved -- see MIN_HTML_SIZE_BYTES), False only when no paper
    exists at this ID at all (used by scrape_month to know when to stop).
    """
    full_id = f"{year}{month}.{paper_id}"
    paper_url = ARXIV_HTML_BASE + full_id
    log.info("Processing %s ...", paper_url)

    fetch_result = fetch_soup(paper_url)
    if fetch_result is None:
        # Genuinely doesn't exist (404, or every retry failed) -- the
        # caller (scrape_month) treats this as "end of month".
        return False
    soup, html_size = fetch_result

    if html_size < MIN_HTML_SIZE_BYTES:
        # The paper exists at this ID (so scanning should continue to the
        # next one), but the page is too small to be a real rendered paper
        # -- don't save anything for it.
        log.info(
            "  \u2717 Skipping %s - HTML is only %d bytes (< %d byte minimum), "
            "looks like a stub/placeholder page.",
            full_id, html_size, MIN_HTML_SIZE_BYTES,
        )
        return True

    base_dir = output_dir if output_dir else Path.cwd()
    dirs = _ensure_output_subdirs(base_dir)

    # ------------------------------------------------------------------ parse
    title, abstract = parse_title_abstract(soup)
    figures = parse_figures(soup, paper_url)
    figure_references = parse_figure_references(soup)
    tables = parse_tables(soup, paper_url)
    table_references = parse_table_references(soup)
    categories = parse_categories(soup, full_id)
    paragraphs = extract_paragraphs(soup)

    # ------------------------------------------------------------------ HTML
    html_path = dirs["html"] / f"{full_id}.html"
    with open(html_path, "w", encoding="utf-8") as fh:
        fh.write(str(soup))
    log.info("  \u2713 HTML saved to: %s", html_path)

    # ------------------------------------------------------------------ figures
    # Download each figure's image and tag each row with the image_id /
    # local path so the JSON and the file on disk can be matched up.
    # (Every row is guaranteed to have a source -- parse_figures drops
    # anything without a real <img>.)
    downloaded = 0
    for fig in figures:
        src = fig["source"]
        image_id = f"{full_id}_{fig['figure_id']}"
        if fig.get("sub_id"):
            image_id += f"_{fig['sub_id']}"
        ext = _guess_image_extension(src)
        image_filename = f"{image_id}{ext}"
        image_path = dirs["images"] / image_filename
        fig["image_id"] = image_id
        if download_image(src, image_path):
            fig["local_path"] = str(Path("images") / image_filename)
            downloaded += 1
        else:
            fig["local_path"] = None

    figures_path = dirs["figures"] / f"{full_id}.json"
    with open(figures_path, "w", encoding="utf-8") as fh:
        json.dump(
            {"paper_id": full_id, "figures": figures, "figure_references": figure_references},
            fh, indent=2, ensure_ascii=False,
        )
    log.info("  \u2713 Figures JSON saved to: %s (%d image(s) downloaded)", figures_path, downloaded)

    # ------------------------------------------------------------------ tables
    tables_path = dirs["tables"] / f"{full_id}.json"
    with open(tables_path, "w", encoding="utf-8") as fh:
        json.dump(
            {"paper_id": full_id, "tables": tables, "table_references": table_references},
            fh, indent=2, ensure_ascii=False,
        )
    log.info("  \u2713 Tables JSON saved to: %s", tables_path)

    # ------------------------------------------------------------------ paragraphs
    paragraphs_path = dirs["paragraphs"] / f"{full_id}.json"
    paragraphs_dict = {f"{full_id}_{i}": text for i, text in enumerate(paragraphs, start=1)}
    with open(paragraphs_path, "w", encoding="utf-8") as fh:
        json.dump({"paper_id": full_id, "paragraphs": paragraphs_dict}, fh, indent=2, ensure_ascii=False)
    log.info("  \u2713 Paragraphs JSON saved to: %s", paragraphs_path)

    # ------------------------------------------------------------------ metadata
    metadata_path = dirs["metadata"] / f"{full_id}.json"
    metadata = {
        "paper_id": full_id,
        "title": title,
        "abstract": abstract,
        "categories": categories,
        "url": paper_url,
        "num_paragraphs": len(paragraphs),
        "num_figures": len(figures),
        "num_tables": len(tables),
        "num_categories": len(categories),
    }
    with open(metadata_path, "w", encoding="utf-8") as fh:
        json.dump(metadata, fh, indent=2, ensure_ascii=False)
    log.info("  \u2713 Metadata JSON saved to: %s", metadata_path)

    log.info(
        "    %d paragraphs, %d figures, %d tables, %d categories",
        len(paragraphs), len(figures), len(tables), len(categories),
    )

    return True


# ---------------------------------------------------------------------------
# Batch scraping
# ---------------------------------------------------------------------------

def scrape_month(
    year: str,
    month: str,
    output_dir: Optional[Path] = None,
    max_papers: Optional[int] = None,
    start_id: int = 1,
    delay: float = REQUEST_DELAY_SECONDS,
) -> None:
    """
    Iterate over arXiv paper IDs for a given *year*/*month* and scrape each
    one until a 404 is returned (no more papers exist for that month).
    """
    log.info("=== Scraping %s/%s (starting at %s.%05d) ===", year, month, year + month, start_id)
    processed = 0

    for numeric_id in range(start_id, 100_000):
        paper_id = f"{numeric_id:05d}"
        found = process_paper(paper_id, year, month, output_dir)

        if not found:
            log.info("No paper found for id %s - assuming end of %s/%s.", paper_id, year, month)
            break

        processed += 1
        if max_papers is not None and processed >= max_papers:
            log.info("Reached max_papers limit (%d). Stopping.", max_papers)
            break

        time.sleep(delay)

    log.info("Done. Processed %d paper(s) for %s/%s.", processed, year, month)


def scrape_range(
    years: List[str],
    months: List[str],
    output_dir: Optional[Path] = None,
    max_papers_per_month: Optional[int] = None,
    delay: float = REQUEST_DELAY_SECONDS,
) -> None:
    """Scrape every (year, month) combination in years x months."""
    for year in years:
        for month in months:
            scrape_month(year, month, output_dir, max_papers=max_papers_per_month, delay=delay)


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------

def normalize_year(year: str) -> str:
    """arXiv IDs use a 2-digit year (YYMM.NNNNN). Accept either "25" or
    "2025" on the command line and normalize to the 2-digit form used in
    the actual paper ID / URL. Raises ValueError for anything else."""
    year = year.strip()
    if len(year) == 2 and year.isdigit():
        return year
    if len(year) == 4 and year.isdigit():
        return year[2:]
    raise ValueError(f"Year must be 2 digits (e.g. '25') or 4 digits (e.g. '2025'), got {year!r}")


def normalize_month(month: str) -> str:
    """Accept "4" or "04" and normalize to zero-padded 2-digit form."""
    month = month.strip().zfill(2)
    if len(month) != 2 or not month.isdigit() or not (1 <= int(month) <= 12):
        raise ValueError(f"Month must be 01-12, got {month!r}")
    return month


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Scrape ALL arXiv papers for a given year/month: figures, "
                     "tables, and paragraphs, kept separate, with captions and "
                     "cross-references preserved."
    )
    parser.add_argument("year", help="Year, 2-digit (24) or 4-digit (2024) - both work")
    parser.add_argument("month", help="Month, 1-2 digits (8 or 08) - both work")
    parser.add_argument(
        "--max_papers", type=int, default=None,
        help="Maximum number of papers to scrape (for testing)",
    )
    parser.add_argument(
        "--start_id", type=int, default=1,
        help="Starting paper ID (default: 1, useful for resuming)",
    )
    parser.add_argument(
        "--output_dir", type=str, default=None,
        help="Directory to save .json/.html files into "
             "(default: '<year>_<month>', e.g. '2025_04')",
    )
    parser.add_argument(
        "--delay", type=float, default=REQUEST_DELAY_SECONDS,
        help=f"Seconds to sleep between papers (default: {REQUEST_DELAY_SECONDS})",
    )
    args = parser.parse_args()

    year = normalize_year(args.year)
    month = normalize_month(args.month)

    # Default output directory: "<year>_<month>", using the same (zero-padded,
    # normalized) values so e.g. `2025 4` and `25 04` both land in `25_04`.
    output_dir = Path(args.output_dir) if args.output_dir else Path(f"{year}_{month}")

    print(f"\U0001F4DA Scraping ALL papers from {year}/{month}...")
    print(f"   Output directory: {output_dir}")
    print(f"   Starting at paper {args.start_id:05d}")
    if args.max_papers:
        print(f"   Max papers: {args.max_papers}")
    print()

    scrape_month(
        year=year,
        month=month,
        output_dir=output_dir,
        max_papers=args.max_papers,
        start_id=args.start_id,
        delay=args.delay,
    )

    print(f"\n\u2705 Done! Check {output_dir}/ for {year}{month}.*.json and .html files")


if __name__ == "__main__":
    main()