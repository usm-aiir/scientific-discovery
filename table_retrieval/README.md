# Table Retrieval

A two-stage retrieval pipeline for finding evidence in scientific tables.

```text
Query
  ↓
Stage 1: Table Retrieval
BM25 + BGE-M3 + Reciprocal Rank Fusion
  ↓
Top candidate tables
  ↓
Stage 2: Cell Retrieval
Fine-tuned BGE
  ↓
Top evidence cells
```

## Usage

```python
from table_retrieval.pipeline import EvidencePipeline

pipeline = EvidencePipeline.from_artifacts(
    stage1_index="/path/to/index",
    stage2_checkpoint="/path/to/checkpoint",
)

result = pipeline.search("What accuracy was reported?")
```

The pipeline is designed to be initialized once and reused for incoming queries.

## Structure

```text
table_retrieval/
├── pipeline.py
├── stage1/
├── stage2/
├── experiments/
├── demo/
└── tests/
```

- `pipeline.py` connects table retrieval and cell retrieval.
- `stage1/` retrieves relevant tables.
- `stage2/` ranks evidence cells from the retrieved tables.
- `experiments/` contains training and evaluation workflows.
- `demo/` contains the local interface.
- `tests/` contains runtime and regression tests.

## Stage 1

Stage 1 combines lexical and dense retrieval.

- **BM25** retrieves tables using token overlap.
- **BGE-M3** retrieves tables using dense embeddings.
- **Reciprocal Rank Fusion** combines both rankings.

The highest-ranked tables are passed to Stage 2.

## Stage 2

Stage 2 creates contextual representations for cells in the retrieved tables and
ranks them with a fine-tuned BGE model.

Cells are ranked together across all candidate tables, and the highest-scoring
cells are returned as evidence.
