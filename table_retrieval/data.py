# Objective: Load table/query data and convert tables into searchable text.

import json
import re
from pathlib import Path

DATA = Path("arxiv_data/SIGIRSciDis/tableGen/table_query_output")


def load_json(path):
    """Read a UTF-8 JSON file."""
    return json.loads(Path(path).read_text(encoding="utf-8"))


def check_id(value):
    """Return an identifier safe for whitespace-separated TREC files."""
    value = str(value)
    if not value or any(char.isspace() for char in value):
        raise ValueError(f"Invalid TREC identifier: {value!r}")
    return value


def tokenize(text):
    """Lowercase words/numbers; retain internal decimal points and hyphens.

    Examples: 'Qwen-0.5B' and '2.59' each remain one token. No stemming
    or stopword removal is applied in this first baseline.
    """
    return re.findall(r"\w+(?:[.\-]\w+)*", text.lower())


def table_to_text(table):
    """Include captions, all cells, and reference text, without paper abstracts.

    The first row is used as a provisional header. Scientific tables may have
    multiple header rows, so we retain every later row, including irregular
    ones. This is a text baseline, not a reconstruction of merged headers.
    """
    parts = [str(table.get(key) or "") for key in ("caption", "sub_caption")]
    rows = table.get("rows") or []
    if rows:
        headers = ["" if cell is None else str(cell) for cell in rows[0]]
        parts.append(" | ".join(headers))
        for row in rows[1:]:
            cells = []
            for column, value in enumerate(row):
                header = headers[column] if column < len(headers) else ""
                value = "" if value is None else str(value)
                cells.append(f"{header}: {value}" if header else value)
            parts.append(" | ".join(cells))
    parts.append(str(table.get("reference_text") or ""))
    return "\n".join(part for part in parts if part)
