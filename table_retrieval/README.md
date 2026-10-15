# Table Retrieval

Retrieves evidence cells from scientific tables with a fixed two-stage pipeline:

```text
Query
→ BM25 + BGE-M3
→ Fusion
→ Top-5 Tables
→ Structural BGE
→ Global Top-5 Evidence Cells
```

## Setup

```bash
python -m venv .venv
.venv/bin/pip install -e .
```

Then edit `table_retrieval/config.json` so
`stage1_index` and `stage2_checkpoint` point to completed local artifacts.
Device, Top-k values, and the optional source-HTML directory are configured in
the same file.

## Search

From the repository root:

```bash
python -m table_retrieval.search "What accuracy was reported?"
```

The command prints the five retrieved tables and the five highest-scoring cells
across those tables as JSON. It only reads existing artifacts; it does not build
indexes, train models, evaluate judgments, or run aggregation.

Offline index construction and saved-run metrics live under
`table_retrieval/experiments/stage1/`. Stage 2 training, fixed splits, and the
Structural BGE benchmark live under `table_retrieval/experiments/stage2_bge/`.
See [experiments/README.md](experiments/README.md) for reproduction commands.
