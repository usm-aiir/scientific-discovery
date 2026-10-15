# Table retrieval

## Objective

Retrieve evidence from scientific tables. Stage 1 finds relevant tables;
Stage 2 ranks their cells and returns the most relevant evidence cells.

## Pipeline

```text
Question
  ↓
Stage 1: BM25 + BGE-M3 + rank fusion
  ↓
Top-5 tables
  ↓
Stage 2: Structural BGE cell ranking
  ↓
Top-5 evidence cells
```

## Structure

```text
table_retrieval/
├── README.md
├── config.example.json
├── search.py
├── pipeline.py
├── data.py
├── stage1/
├── stage2/
└── experiments/
    ├── stage1/
    └── stage2_bge/
```

`stage1/` is the live table retrieval code. `stage2/` is the live cell retrieval
code. `search.py` loads settings and runs a query; `pipeline.py` connects the two
stages. `data.py` provides shared data loading and ID handling.

## Setup

With Conda installed, run once from the repository root:

```bash
bash setup.sh
conda activate table-retrieval
```

The same environment runs live retrieval and both experiments.

## Run a query

Copy the example once, edit the local paths, then search from the repository root:

```bash
cp table_retrieval/config.example.json table_retrieval/config.json
# Set stage1_index, stage2_checkpoint and table_source_dir to your local paths.
python -m table_retrieval.search "What accuracy was reported?"
```

Live retrieval needs the completed Stage 1 index and its corpus, the external
[BGE checkpoint](experiments/stage2_bge/README.md#checkpoint), and the original
table HTML for structural context. The [example config](config.example.json)
identifies the reference artifact paths; these assets are not installed by setup.

`config.json` is local and gitignored. Relative paths resolve from its directory.
If it is absent, search reads the example. Use `--config /path/to/settings.json`
to select another config. Results are JSON with cell/table IDs, original
coordinates and scores.

## Experiments

- [Stage 1](experiments/stage1/README.md): `experiments/stage1/` reproduces table indexing and evaluation.
- [Stage 2 BGE](experiments/stage2_bge/README.md): `experiments/stage2_bge/` reproduces cell retriever training and evaluation.
