# Builds the structural text representation used for Stage 2 BGE cell ranking.

from .cells import make_cells, text_value as text
from .structure import recover_table

# Identifies the current structural cell representation format.
REPRESENTATION_VERSION = 'structural_v1'

# Builds the BGE text for one cell using its row, header, value, and table context.
def structural_bge_text(cell, table, context):
    """Place local row/header/value semantics before optional table metadata."""
    parts = [f'{cell["row_label"]} | {" > ".join(context["header_path"])} | {text(cell["value"])}']
    if context['row_context']:
        parts.append('Row context: ' + '; '.join(context['row_context']))
    for key, label in (('caption', 'Table'), ('sub_caption', 'Subtable'), ('paper_title', 'Paper')):
        if text(table.get(key)):
            parts.append(f'{label}: {text(table[key])}')
    return ' | '.join(parts)

# Creates cells, recovers their table structure, and replaces each cell's text with the structural BGE representation.
def build_structural_cells(table_id, table, sources=(), html_path=None):
    """Enrich scoring text, retaining original metadata for downstream compatibility."""
    cells = make_cells(table_id, table)
    contexts, audit = recover_table(table_id, table, sources, html_path)
    for cell in cells:
        cell['text'] = structural_bge_text(cell, table, contexts[cell['cell_id']])
    return cells, audit
