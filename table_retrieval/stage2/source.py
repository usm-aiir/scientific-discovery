"""Optional HTML-based structural recovery for Stage 2.

Normal Stage 2 retrieval uses corpus structure and requires no local HTML files.
These helpers enhance recovery only when matching source HTML is available.
"""


from functools import lru_cache
from pathlib import Path
import re

# Converts an HTML table cell into clean text while preserving math and footnotes.
def cell_text(element):
    from bs4 import NavigableString
    if isinstance(element, NavigableString):
        return str(element)
    if element.name == 'math':
        alt = element.get('alttext', '')
        return f'${alt}$' if alt else element.get_text()
    if element.name == 'table':
        return ' '.join(' '.join(cell.get_text().split())
                        for cell in element.find_all(['td', 'th']) if cell.get_text().strip())
    classes = element.get('class', [])
    if 'ltx_note' in classes and 'ltx_role_footnote' in classes:
        marker = element.find('sup', class_='ltx_note_mark')
        return '[^' + (marker.get_text(strip=True) if marker else '?') + ']'
    return ''.join(cell_text(child) for child in element.children)

# Converts an HTML table into cells with their original row and column positions.
def extract_grid(table):
    rows = []
    for child in table.find_all(['thead', 'tbody', 'tfoot', 'tr'], recursive=False):
        rows.extend([child] if child.name == 'tr' else child.find_all('tr', recursive=False))
    cells, occupied, row_index = [], set(), 0
    for row in rows:
        tags = row.find_all(['th', 'td'], recursive=False)
        column = 0
        for tag in tags:
            while (row_index, column) in occupied:
                column += 1
            colspan = int(tag.get('colspan', 1) or 1)
            rowspan = int(tag.get('rowspan', 1) or 1)
            # Preserve the source parser's origin positions, including legacy
            # zero spans. Recovery still requires an exact, unique row match.
            cell = dict(row=row_index, col=column, text=' '.join(cell_text(tag).split()),
                        is_header=tag.name == 'th')
            if colspan > 1:
                cell['colspan'] = colspan
            if rowspan > 1:
                cell['rowspan'] = rowspan
            cells.append(cell)
            occupied.update((r, c) for r in range(row_index, row_index + rowspan)
                            for c in range(column, column + colspan))
            column += colspan
        if tags:
            row_index += 1
    return cells, row_index

# Extracts all top-level tables and their cell structure from an HTML file.
def source_tables(path):
    from bs4 import BeautifulSoup
    soup = BeautifulSoup(Path(path).read_text(encoding='utf-8'), 'html.parser')
    sources = []
    for index, table in enumerate(soup.find_all('table')):
        if table.find_parent('table') is not None:
            continue
        cells, n_rows = extract_grid(table)
        if cells:
            sources.append(dict(index=index, html_id=table.get('id'), cells=cells, n_rows=n_rows))
    return sources

# Caches parsed HTML tables so the same file does not need to be parsed repeatedly.
@lru_cache(maxsize=32)
def _cached_sources(path, modified, size):
    # File identity participates in the key; changing HTML invalidates the cache.
    return source_tables(path)

def load_table_sources(table_id, directory=None):
    """Load optional paper HTML; ``directory=None`` is a normal, valid setup.

    Missing HTML or an unsupported paper ID is not an error. Returning
    ``(), None`` means no source HTML is available; use corpus structure.
    Read/parse errors are handled by the Stage 2 pipeline's corpus fallback.
    """
    if directory is None:
        return (), None
    paper = table_id.split('::', 1)[0]
    if not re.fullmatch(r'\d{4}\.\d{4,5}(?:v\d+)?', paper):
        return (), None
    path = Path(directory) / f'{paper}.html'
    if not path.is_file():
        return (), None
    stat = path.stat()
    return _cached_sources(path.resolve(), stat.st_mtime_ns, stat.st_size), path
