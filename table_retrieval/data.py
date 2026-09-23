# Objective: Load table/query data and convert tables into searchable text.

from collections import defaultdict
from dataclasses import asdict, dataclass
import hashlib
import warnings

import pandas as pd
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


def normalize_table_id(uid):
    """Match corpus IDs to qrels IDs without changing either input file.

    For example, 2504.16674::T1::b becomes 2504.16674_1_b.
    Preserve subtable suffixes (including empty ones) and unrelated IDs.
    """
    match = re.fullmatch(r"(\d{4}\.\d{4,5}(?:v\d+)?)::T(\d+)(?:::(.*))?", uid)
    if match is None:
        return uid
    paper, table, subtable = match.groups()
    return f"{paper}_{table}" + (f"_{subtable}" if subtable is not None else "")


def table_text(table):
    """One shared field selection; preserve zero cells and irregular rows."""
    parts = [str(table.get(key) or '') for key in ('paper_title', 'caption')]
    for row in table.get('rows') or []:
        parts.append(' | '.join('' if cell is None else str(cell) for cell in row))
    return '\n'.join(part for part in parts if part)


def digest(path):
    result = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b''):
            result.update(block)
    return result.hexdigest()


def write_json(path, value):
    path = Path(path)
    if path.exists():
        if load_json(path) == value:
            return
        raise ValueError(f'Existing {path} differs; choose a new experiment/output path')
    temporary = path.with_suffix(path.suffix + '.tmp')
    with temporary.open('x') as stream:
        stream.write(json.dumps(value, indent=2) + '\n')
    temporary.replace(path)


def write_run(path, rankings, name):
    path = Path(path)
    temporary = path.with_suffix(path.suffix + '.tmp')
    if path.exists():
        raise ValueError(f'Existing rankings {path}; choose a new experiment/output path')
    with temporary.open('x') as stream:
        for qid, hits in rankings.items():
            for rank, (uid, score) in enumerate(hits, 1):
                stream.write(f'{qid} Q0 {uid} {rank} {score:.12g} {name}\n')
    temporary.replace(path)


def read_run(path):
    result = defaultdict(list)
    for line in Path(path).read_text().splitlines():
        qid, _, uid, rank, score, _ = line.split()
        result[qid].append((int(rank), uid, float(score)))
    return {qid: [(uid, score) for _, uid, score in sorted(hits)] for qid, hits in result.items()}


@dataclass
class InputLimits:
    """Deterministic limits shared by training and corpus indexing."""

    query_length: int = 128
    table_length: int = 512
    context_tokens: int = 64
    cell_tokens: int = 16
    max_rows: int = 64
    max_columns: int = 32

    def __post_init__(self):
        if any(value < 1 for value in asdict(self).values()):
            raise ValueError("Input limits must be positive.")
        if self.query_length < 3 or self.table_length < self.context_tokens + 5:
            raise ValueError("Sequence lengths must leave room for special tokens and cells.")


def clip_text(tokenizer, text, budget):
    """Bound WordPiece length before TAPAS packs rows into its token budget."""
    text = "" if text is None else str(text)
    tokens = tokenizer.tokenize(text)
    return text if len(tokens) <= budget else tokenizer.convert_tokens_to_string(tokens[:budget])


def table_inputs(table, tokenizer, limits):
    """Keep a rectangular table, with row/column structure available to TAPAS.

    The first row is the provisional header, as in the BM25 baseline. Short
    rows are padded; extra cells get an empty header. Large tables retain the
    leading rows/columns. TAPAS may then drop trailing rows to fit the budget.
    """
    rows = table.get("rows") or []
    context = " ".join(str(table.get(key) or "") for key in
                       ("paper_title", "caption", "sub_caption", "reference_text"))
    context = clip_text(tokenizer, context, limits.context_tokens)
    width = max((len(row) for row in rows), default=1)
    width = min(max(1, width), limits.max_columns,
                (limits.table_length - limits.context_tokens - 3) // 2)
    # Reserve enough space for headers and at least one data row.
    cell_budget = min(limits.cell_tokens,
                      (limits.table_length - limits.context_tokens - 3) // (2 * width))

    def cells(row):
        return [clip_text(tokenizer, row[i] if i < len(row) else "", cell_budget)
                for i in range(width)]

    headers = cells(rows[0] if rows else [])
    body = [cells(row) for row in rows[1:limits.max_rows + 1]]
    frame = pd.DataFrame(body, columns=headers, dtype=str)
    # Transformers 4.44 uses positional Series indexing internally. Suppress
    # only that known pandas deprecation while tokenizing, not other warnings.
    with warnings.catch_warnings():
        warnings.filterwarnings(
            "ignore",
            message=r"Series\.__getitem__ treating keys as positions is deprecated\.",
            category=FutureWarning,
            module=r"transformers\.models\.tapas\.tokenization_tapas$",
        )
        return dict(tokenizer(table=frame, queries=context, max_length=limits.table_length,
                              truncation="drop_rows_to_fit", padding=False))


def table_features(table, tokenizer, config, table_length=512):
    rows = table.get('rows') or []
    max_cols = config.type_vocab_sizes[1] - 1
    max_rows = config.type_vocab_sizes[2] - 1
    width = min(max_cols, max(1, max(map(len, rows), default=1)))
    height = min(max_rows, max(0, len(rows) - 1))
    title = str(table.get('paper_title') or '')
    title_tokens = tokenizer.tokenize(title)
    if len(title_tokens) > table_length - 4:
        # Some extracted titles contain much more than a title. Retain a prefix
        # and leave room for table evidence, without changing the source corpus.
        warnings.warn('Oversized paper title truncated for encoding; source corpus unchanged.',
                      UserWarning)
        title_tokens = title_tokens[:min(128, table_length // 2)]
        title = tokenizer.convert_tokens_to_string(title_tokens)
        while len(tokenizer.tokenize(title)) > min(128, table_length // 2):
            title_tokens.pop()
            title = tokenizer.convert_tokens_to_string(title_tokens)
    while True:
        def cells(row):
            # Preserve zero-valued numeric cells too.
            return [str(row[i]) if i < len(row) and row[i] is not None else '' for i in range(width)]
        frame = pd.DataFrame([cells(row) for row in rows[1:height + 1]],
                             columns=cells(rows[0] if rows else []), dtype=str)
        # Compute the budget before asking HF to encode. An untruncated
        # encode raises on overflow and would otherwise skip cell trimming.
        tokenized = tokenizer._tokenize_table(frame)
        budget = tokenizer._get_max_num_tokens(
            tokenizer.tokenize(title), tokenized, width, height, table_length)
        if budget is not None:
            with warnings.catch_warnings():
                warnings.filterwarnings('ignore', category=FutureWarning,
                                        module=r'transformers\.models\.tapas\.tokenization_tapas$')
                encoded = dict(tokenizer(table=frame, queries=title,
                    max_length=table_length, truncation='drop_rows_to_fit', padding=False))
            break
        if width >= height and width > 1:
            width -= 1
        elif height > 0:
            height -= 1
        else:
            raise ValueError('Cannot fit table into sequence')
    if len(encoded['input_ids']) > table_length:
        raise ValueError('Table packing exceeded the sequence budget')
    for channel in encoded['token_type_ids']:
        channel[6] = 0
    return encoded


def training_examples(queries, qrels_path, corpus, min_grade=1):
    """Resolve qrels to corpus IDs; never use explanations as encoder inputs."""
    lookup = {}
    for uid in corpus:
        normalized = normalize_table_id(check_id(uid))
        if normalized in lookup:
            raise ValueError(f"Ambiguous corpus table ID: {normalized}")
        lookup[normalized] = uid
    positives = defaultdict(set)
    query_ids = [check_id(query["query_id"]) for query in queries]
    if len(set(query_ids)) != len(query_ids):
        raise ValueError("Duplicate training query IDs.")
    allowed = set(query_ids)
    missing = set()
    for line in Path(qrels_path).read_text().splitlines():
        if not line.strip():
            continue
        qid, _, uid, grade = line.split()
        if qid not in allowed or int(grade) < min_grade:
            continue
        normalized = normalize_table_id(uid)
        if normalized in lookup:
            positives[qid].add(lookup[normalized])
        else:
            missing.add(uid)
    examples = []
    skipped = []
    for qid, query in zip(query_ids, queries):
        if not isinstance(query["query"], str) or not query["query"].strip():
            raise ValueError(f"Training query {qid} must contain nonempty text.")
        if not positives[qid]:
            skipped.append(qid)
        elif len(positives[qid]) == len(corpus):
            raise ValueError(f"Query {qid} has no negative tables in the corpus.")
        else:
            examples.append((qid, query["query"], positives[qid]))
    if not examples:
        raise ValueError("No training queries have positive judgments in the corpus.")
    report = {"missing_judged_table_ids": sorted(missing), "skipped_query_ids": skipped}
    if missing or skipped:
        warnings.warn(f"Training data: {len(missing)} missing judged table IDs; "
                      f"{len(skipped)} queries skipped. Details saved with the checkpoint.")
    return examples, report


def load_corpus(path):
    corpus = load_json(path)
    if not isinstance(corpus, dict) or not corpus:
        raise ValueError('Corpus must be a nonempty object keyed by table ID')
    for uid, table in corpus.items():
        if not isinstance(table, dict):
            raise ValueError(f'Table {uid} must be an object')
        rows = table.get('rows')
        if rows is not None and (not isinstance(rows, list) or any(not isinstance(row, list) for row in rows)):
            raise ValueError(f'Table {uid} rows must be lists of cells')
    normalized = [normalize_table_id(check_id(uid)) for uid in corpus]
    if len(normalized) != len(set(normalized)):
        raise ValueError('Ambiguous corpus table IDs')
    return corpus


def load_queries(path):
    queries = load_json(path)
    if not isinstance(queries, list) or any(not isinstance(q, dict) or
            'query_id' not in q or 'query' not in q for q in queries):
        raise ValueError('Queries must be a list of objects with query_id and query fields')
    ids = [check_id(q['query_id']) for q in queries]
    if not queries or len(ids) != len(set(ids)):
        raise ValueError('Queries must be nonempty and have unique IDs')
    if any(not isinstance(q['query'], str) or not q['query'].strip() for q in queries):
        raise ValueError('Queries must contain nonempty text')
    return queries


def load_qrels(path, min_grade=1):
    """Read positive judgments, normalizing table IDs for matching."""
    relevant = defaultdict(set)
    for line in Path(path).read_text().splitlines():
        if line.strip():
            qid, _, uid, grade = line.split()
            if int(grade) >= min_grade:
                relevant[check_id(qid)].add(normalize_table_id(check_id(uid)))
    return relevant


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
        retained = token_ids[:budget]
        encoded.append(tokenizer.build_inputs_with_special_tokens(retained))
    return ids, encoded, clipped
