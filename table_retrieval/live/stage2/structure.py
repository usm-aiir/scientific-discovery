# Recovers reliable headers and row context for Stage 2 cell representations.
import re
from collections import defaultdict
from table_retrieval.data import digest
from .cells import make_cells, text_value as text

# Checks whether a cell value looks numeric.
def numeric(value):
    value = text(value)
    return bool(re.match(r'^\$?[-+]?\d', value))

# Tries to identify the header rows and where the table body begins.
def header_layout(table):
    rows = table['rows']
    if len(rows) < 3:
        return None
    width = max(map(len, rows))
    for header_row in range(min(3, len(rows) - 2)):
        header = rows[header_row]
        body = rows[header_row + 1:]
        if not body or any(len(r) != width for r in body):
            continue
        if len(header) == width:
            if (all(text(v) and not numeric(v) for v in header)
                    and sum(any(numeric(r[c]) for r in body) for c in range(width)) >= width / 2):
                return dict(header_row=header_row, offset=0, body_start=header_row + 1, indicator_count=0)
        elif header_row == 1 and 0 < width - len(header) < width:
            offset = width - len(header)
            indicators = {'', '$\\checkmark$', '✓', '✔', '\\checkmark'}
            if (len(rows[0]) >= offset and all(text(v) for v in rows[0][:offset])
                    and all(text(v) and not numeric(v) for v in header)
                    and all(text(r[c]) in indicators for r in body for c in range(offset))
                    and all(numeric(r[c]) for r in body for c in range(offset, width))):
                return dict(header_row=header_row, offset=offset, body_start=header_row + 1, indicator_count=offset)
    return None

# Normalizes whitespace so cell text can be compared reliably.
def normalize(value):
    # Only normalize whitespace: do not equate different numbers or math labels.
    return " ".join(text(value).split())

# Matches corpus table rows to the same rows in the optional source HTML.
# It tries to answer: Does this corpus table exactly correspond to one table section in the original HTML?
def find_alignment(rows, sources):
    """Match all original cells, including blanks, to one contiguous HTML block."""
    wanted = [[normalize(value) for value in row] for row in rows]
    matches = []
    for source in sources:
        by_row = defaultdict(list)
        for cell in source["cells"]:
            by_row[cell["row"]].append(cell)
        source_rows = [sorted(by_row[row], key=lambda cell: cell["col"])
                       for row in range(source["n_rows"])]
        values = [[normalize(cell["text"]) for cell in row] for row in source_rows]
        for start in range(len(values) - len(wanted) + 1):
            if values[start:start + len(wanted)] == wanted:
                matches.append((source, start, source_rows[start:start + len(wanted)]))
    if len(matches) != 1:
        return None, "ambiguous_alignment" if matches else "no_exact_alignment"
    source, start, aligned_rows = matches[0]
    return {
        "source": source, "start": start, "rows": aligned_rows,
        "coordinates": {(row, column): cell for row, cells in enumerate(aligned_rows)
                        for column, cell in enumerate(cells)},
    }, "exact_html_row_alignment"

# Determines where header rows end and the table body begins.
def body_start(table, alignment):
    rows = table["rows"]
    inferred = header_layout(table)
    if inferred:
        return inferred["body_start"], "conservative_numeric_layout"
    # Source spans plus textual header rows and consistently numerical body rows
    # establish an alignment even when the legacy corpus compressed merged cells.
    if alignment:
        for first in range(1, min(4, len(rows) - 1)):
            headers, body = rows[:first], rows[first:]
            if any(numeric(value) for row in headers for value in row if text(value)):
                continue
            if all(sum(numeric(value) for value in row) >= max(1, len(row) // 3)
                   for row in body):
                return first, "html_spans_with_numeric_body_boundary"
    cells = make_cells("table", table)
    if cells[0]["header_source"] == "inferred_hierarchy":
        width = max(map(len, rows))
        return next(index for index, row in enumerate(rows) if len(row) == width), "existing_inferred_hierarchy"
    if cells[0]["header_source"] in ("flat_first_row", "source_structure"):
        return 1, "existing_header_convention"
    return None, "unknown_header_boundary"

# Collects all header labels that apply to a column.
def headers_for_column(header_rows, column):
    labels = []
    for row in header_rows:
        for cell in row:
            value = text(cell['text'])
            if cell['col'] <= column < cell['col'] + cell.get('colspan', 1) and value and value not in labels:
                labels.append(value)
    return labels

# Extracts non-numeric row conditions from aligned source structure.
def aligned_row_context(alignment, row, first_body, target):
    """Only named categorical prefixes, never neighboring numeric results."""
    prefix = []
    for cell in alignment['rows'][row]:
        if numeric(cell['text']):
            break
        prefix.append(cell['col'])
    labels = []
    for column in prefix[:5]:
        names = headers_for_column(alignment['rows'][:first_body], column)
        if not names:
            continue
        origins = [cell for source_row in alignment['rows'][first_body:row + 1]
                   for cell in source_row if cell['col'] == column and
                   cell['row'] <= target['row'] < cell['row'] + cell.get('rowspan', 1)]
        if len(origins) == 1:
            labels.append(' > '.join(names) + '=' + (text(origins[0]['text']) or '<empty>'))
    return labels

# Recovers header and row context using only the corpus table structure.
def corpus_cell_context(table, row, column, first_body, headers):
    inferred = header_layout(table)
    labels, recovery = [], 'original'
    if inferred and row >= first_body:
        index = column - inferred['offset']
        header = table['rows'][inferred['header_row']]
        if 0 <= index < len(header):
            headers = [text(header[index])]
            recovery = 'conservative_corpus_leaf_header'
        if inferred['indicator_count']:
            labels = [f'{text(table["rows"][0][index])}='
                      f'{text(table["rows"][row][index]) or "<empty>"}'
                      for index in range(inferred['indicator_count'])]
    return headers, labels, recovery

# Builds reliable header and row context for every cell while preserving its original coordinates.
def recover_table(table_id, table, sources=(), html_path=None):
    """Build source-backed context while retaining the original cell IDs."""
    if not sources and isinstance(table.get("cells"), list) and table["cells"]:
        cells = table["cells"]
        if all(isinstance(cell, dict) and type(cell.get("row")) is int and cell['row'] >= 0 and
               type(cell.get("col")) is int and cell['col'] >= 0 and 'text' in cell and
               all(type(cell.get(key, 1)) is int and cell.get(key, 1) >= 1
                   for key in ('rowspan', 'colspan')) for cell in cells):
            sources = [{"index": 0, "cells": cells,
                        "n_rows": max(cell["row"] for cell in cells) + 1}]
    alignment, status = find_alignment(table["rows"], sources) if sources else (None, "html_unavailable")
    first_body, boundary = body_start(table, alignment)
    original = make_cells(table_id, table)
    contexts = {}
    for cell in original:
        # Source-facing cell coordinates are one-based; list/span indices are not.
        row, column = cell["row"] - 1, cell["column"] - 1
        headers = list(cell["header_path"])
        row_labels = []
        recovery = "original"
        if alignment and first_body is not None:
            target = alignment["coordinates"][(row, column)]
            labels = headers_for_column(alignment['rows'][:first_body], target['col'])
            if labels:
                headers = labels
                recovery = "aligned_html_spans"
            if row >= first_body:
                row_labels = aligned_row_context(alignment, row, first_body, target)
        elif first_body is not None:
            headers, row_labels, recovery = corpus_cell_context(table, row, column, first_body, headers)
        contexts[cell["cell_id"]] = {
            "header_path": headers, "row_context": row_labels,
            "cell_kind": "unknown" if first_body is None else "header" if row < first_body else "body",
            "header_recovery": recovery,
            "header_known": bool(headers) and not all(re.fullmatch(r"Column \d+", value) for value in headers),
        }
    audit = {
        "table_id": table_id, "alignment": status, "header_boundary": boundary,
        "body_start": first_body + 1 if first_body is not None else None,
        "html_path": str(html_path) if html_path else None,
        "html_sha256": digest(html_path) if html_path and html_path.is_file() else None,
        "source_table_index": alignment["source"]["index"] if alignment else None,
        "source_row_offset": alignment["start"] if alignment else None,
        "cells": len(original),
        "recovered_header_cells": sum(context["header_recovery"] != "original" for context in contexts.values()),
    }
    return contexts, audit
