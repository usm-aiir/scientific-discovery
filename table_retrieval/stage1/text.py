"""Turn tables into text for Stage 1 retrieval."""
from ..data import check_id


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


def table_text(table):
    """One shared field selection; preserve zero cells and irregular rows."""
    parts = [str(table.get(key) or '') for key in ('paper_title', 'caption')]
    for row in table.get('rows') or []:
        parts.append(' | '.join('' if cell is None else str(cell) for cell in row))
    return '\n'.join(part for part in parts if part)


def prepare_text_inputs(corpus, tokenizer, max_length):
    """Apply the shared BGE token budget used by dense and matched BM25 inputs."""
    from tqdm.auto import tqdm
    ids = sorted(check_id(uid) for uid in corpus)
    encoded, clipped = [], []
    budget = max_length - tokenizer.num_special_tokens_to_add(pair=False)
    for uid in tqdm(ids, desc='Preparing shared table text'):
        text = table_text(corpus[uid])
        token_ids = tokenizer(text, add_special_tokens=False, truncation=False)['input_ids']
        if len(token_ids) > budget:
            clipped.append({'table_id': uid, 'original_tokens': len(token_ids)})
        encoded.append(tokenizer(text, add_special_tokens=True, truncation=True,
                                 max_length=max_length)['input_ids'])
    return ids, encoded, clipped
