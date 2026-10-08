"""Prepare tables while preserving each cell's original row and column position."""
from dataclasses import dataclass
import hashlib
import json

import pandas as pd


def rows_digest(rows):
    return hashlib.sha256(json.dumps(rows, ensure_ascii=False, separators=(',', ':')).encode()).hexdigest()


@dataclass
class StructuredTable:
    frame: pd.DataFrame
    # Each model body cell maps to an original Corpus.json rows[r][c].
    coordinates: list
    note: str


def prepare_table(table, layout=None):
    rows = table.get('rows') or []
    if len(rows) < 2 or not all(isinstance(row, list) for row in rows):
        raise ValueError('Need a header and at least one data row.')
    if layout is None:
        width = len(rows[0])
        if not width or any(len(row) != width for row in rows):
            raise ValueError('Irregular rows: merged-cell positions cannot be recovered from Corpus.json. '
                             'Supply a reviewed layout instead of silently padding or shifting cells.')
        headers = [str(cell) if cell is not None else '' for cell in rows[0]]
        coordinates = [[[r, c] for c in range(width)] for r in range(1, len(rows))]
        note = 'First corpus row treated as the header; no merged-cell structure inferred.'
    else:
        if layout.get('rows_sha256') != rows_digest(rows):
            raise ValueError('Reviewed layout does not match these corpus rows; inspect it again.')
        headers, coordinates = layout['headers'], layout['body_coordinates']
        note = layout['note']
        if (not headers or not all(isinstance(h, str) for h in headers) or not coordinates or
                any(len(row) != len(headers) for row in coordinates)):
            raise ValueError('Reviewed layout must define a rectangular table and string headers.')
    body = []
    for row in coordinates:
        values = []
        for coord in row:
            if (not isinstance(coord, (list, tuple)) or len(coord) != 2 or
                    any(type(i) is not int for i in coord)):
                raise ValueError('Each cell needs an original integer [row, column] coordinate.')
            r, c = coord
            if r < 0 or r >= len(rows) or c < 0 or c >= len(rows[r]):
                raise ValueError(f'Source coordinate outside the table: {coord}')
            values.append('' if rows[r][c] is None else str(rows[r][c]))
        body.append(values)
    # TAPAS accepts duplicate headers; positional provenance stays unambiguous.
    return StructuredTable(pd.DataFrame(body, columns=headers, dtype=str), coordinates, note)


def map_cells(table, structured, predicted):
    """Map TAPAS's zero-based body positions to zero-based original positions."""
    cells = []
    seen = set()
    for r, c in predicted:
        r, c = int(r), int(c)
        if not (0 <= r < len(structured.frame) and 0 <= c < len(structured.frame.columns)):
            raise ValueError(f'Model returned a cell outside its input: {(r, c)}')
        source_r, source_c = structured.coordinates[r][c]
        source = (source_r, source_c)
        if source in seen:
            continue
        seen.add(source)
        cells.append(dict(row=source_r, column=source_c,
                          value=table['rows'][source_r][source_c],
                          header=str(structured.frame.columns[c]),
                          model_row=r, model_column=c))
    return cells
