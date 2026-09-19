# Scientific Discovery

Table retrieval with BM25 and a BERT + TAPAS dense retriever. Both search the
same corpus and are evaluated using Recall@10 on the validation queries.

BM25 code is in `table_retrieval/baseline/`, DTR is in `table_retrieval/DTR/`,
and data loading and evaluation are shared.

## Setup

Run from the repository root with Conda installed:

```bash
conda env create -f environment.yml
conda activate scidiscovery
```

This installs the libraries in `requirements.txt`. Model weights download on
first use. DTR commands below use a CUDA GPU; use `--device cpu` if needed.

## Data

Place `Corpus.json`, `Train.json`, `Train_table_qrels.tsv`, `Val.json`, and
`Val_table_qrels.tsv` in:

```text
arxiv_data/SIGIRSciDis/tableGen/table_query_output/
```

The corpus contains table contents, query files contain query text, and qrels
identify relevant tables. The dataset must be obtained separately.

## BM25

```bash
python -m table_retrieval.baseline.run
python -m table_retrieval.evaluate
```

Indexes the corpus and retrieves 10 tables per validation query. Rankings are
saved to `results/bm25_val.run` and scores to `results/bm25_val.metrics.json`.

## DTR

Run these commands in order:

```bash
python -m table_retrieval.DTR.train --device cuda
python -m table_retrieval.DTR.index --device cuda
python -m table_retrieval.DTR.run --device cuda
python -m table_retrieval.evaluate dtr
```

Training fine-tunes BERT and TAPAS on training queries and judgments for 3 epochs
(batch size 4, seed 42). Indexing encodes the corpus; retrieval returns 10 tables
per validation query. Test queries and judgments are not used.

Models are saved in `results/dtr_model/`, embeddings in `results/dtr_index/`,
and rankings and scores in `results/dtr_val.run` and `results/dtr_val.metrics.json`.
Training and indexing require new or empty output folders. Training saves only
at completion. Skip training and indexing when reusing a completed model and index.

TAPAS inputs are limited to 512 tokens, so large tables are truncated. Unlike
BM25, DTR includes paper titles.

## Current results

Evaluated on 63,021 tables and 250 validation queries, retrieving 10 tables per query.

| Method | Recall@10 |
| --- | ---: |
| BM25 | 79.20% |
| DTR (BERT + TAPAS) | 37.05% |
