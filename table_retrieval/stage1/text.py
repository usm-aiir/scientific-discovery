# Converts each table into text for Stage 1 retrieval.

# Combines the table title, caption, and rows into one searchable text string.
def table_text(table):
    # Uses the same table fields for BM25 and BGE and keeps all row values.
    parts = [str(table.get(key) or '') for key in ('paper_title', 'caption')]
    for row in table.get('rows') or []:
        parts.append(' | '.join('' if cell is None else str(cell) for cell in row))
    return '\n'.join(part for part in parts if part)
