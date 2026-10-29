# Shared data structures and file utilities used across the retrieval system.

from dataclasses import dataclass, asdict
from typing import Optional
import math
import hashlib
import json
import re
from pathlib import Path

# Historical inspection scripts use this path; live retrieval reads configuration.
DATA = Path("arxiv_data/SIGIRSciDis/tableGen/table_query_output")

# Both experiment stages use this source unless another directory is supplied.
DEFAULT_DATA_DIR = Path('/mnt/netstore1_home/behrooz.mansouri/SciDis/tableGen/table_query_output_v2')
RESULTS_DIR = Path(__file__).resolve().parents[1] / 'results'


def table_split_paths(data_dir=DEFAULT_DATA_DIR, split='val'):
    """Find the original v2 questions, table judgments, and matched-cell labels."""
    if split not in ('train', 'val', 'test'):
        raise ValueError('Choose split train, val or test.')
    data_dir = Path(data_dir).resolve()
    suffix = '.INSTRUCTOR_ONLY' if split == 'test' else ''
    title = split.title()
    return {
        'queries': data_dir / f'{title}.json',
        'qrels': data_dir / f'{title}_table_qrels{suffix}.tsv',
        'cells': data_dir / f'{title}_matched_cells{suffix}.tsv',
    }


def load_json(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))

# Validates IDs so they are safe to use in retrieval/evaluation files.
def check_id(value):
    value = str(value)
    if not value or any(char.isspace() for char in value):
        raise ValueError(f"Invalid TREC identifier: {value!r}")
    return value

# Normalizes table IDs for shared corpus collision checks and evaluation matching.
def normalize_table_id(uid):
    match = re.fullmatch(r"(\d{4}\.\d{4,5}(?:v\d+)?)::T(\d+)(?:::(.*))?", uid)
    if match is None:
        return uid
    paper, table, subtable = match.groups()
    return f"{paper}_{table}" + (f"_{subtable}" if subtable is not None else "")

# Computes a SHA-256 hash used to verify that files have not changed.
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


COORDINATES = 'one-based source r{row}c{column}; Corpus.json rows[row-1][column-1], including header rows'
SCHEMA_VERSION = '4.0'

# Represents a table passed from Stage 1 retrieval into Stage 2.
@dataclass
class Candidate:
    """A table passed between stages, with optional retrieval rank and score."""
    table_id: str
    table: Optional[dict]
    retrieval_rank: Optional[int] = None
    retrieval_score: Optional[float] = None
    
    # makes sure malformed candidates don't enter Stage 2.
    def __post_init__(self):
        if not isinstance(self.table_id, str) or not self.table_id:
            raise ValueError('Candidate table_id must be nonempty text.')
        if self.table is not None and not isinstance(self.table, dict):
            raise ValueError('Candidate table must be an object or null.')
        if self.retrieval_rank is not None and (type(self.retrieval_rank) is not int or self.retrieval_rank < 1):
            raise ValueError('Candidate retrieval_rank must be a positive integer.')
        if self.retrieval_score is not None:
            if isinstance(self.retrieval_score, bool) or not isinstance(self.retrieval_score, (int, float)):
                raise ValueError('Candidate retrieval_score must be a number.')
            if not math.isfinite(self.retrieval_score):
                raise ValueError('Candidate retrieval_score must be finite.')

    def to_dict(self):
        return asdict(self)

    @classmethod
    def from_dict(cls, value):
        if not isinstance(value, dict) or 'table_id' not in value or 'table' not in value:
            raise ValueError('Each candidate needs table_id and table fields.')
        return cls(value['table_id'], value['table'], value.get('retrieval_rank'), value.get('retrieval_score'))

# Safely writes retrieval results without overwriting an existing result file.
def write_predictions(path, predictions):
    """Write JSONL results without replacing an existing output file."""
    import os
    path = Path(path)
    partial = path.with_name(path.name + '.partial')
    if path.exists() or partial.exists():
        raise ValueError('Output or partial output already exists; choose a new output path.')
    path.parent.mkdir(parents=True, exist_ok=True)
    count = failed = 0
    with partial.open('x', encoding='utf-8') as stream:
        for result in predictions:
            stream.write(json.dumps(result, ensure_ascii=False, allow_nan=False) + '\n')
            stream.flush()
            count += 1
            failed += result['status'] != 'complete'
    # Publish only a complete file and never replace another writer's output.
    os.link(partial, path)
    partial.unlink()
    return count, failed
