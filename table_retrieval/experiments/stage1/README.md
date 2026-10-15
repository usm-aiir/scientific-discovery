# Stage 1: table retrieval evaluation

## Purpose

Reproduce Stage 1 table retrieval evaluation using the live retrievers.

## Method

- BM25: lexical retrieval, with full-text and BGE-token-matched variants.
- BGE-M3: dense retrieval using the pretrained `BAAI/bge-m3` model.
- Equal-weight reciprocal rank fusion: combine token-matched BM25 and BGE ranks.

The fixed recipe uses model revision
`5617a9f61b028005a4858fdac845db406aefb181`, 2,048-token table inputs,
100 candidates per retriever and fusion constant 60. No model is trained here.

## Data

| File | Contents |
|---|---|
| `data/val.json` | 250 validation questions. |
| `data/test.json` | 250 test questions. |
| `data/val_qrels.tsv` | Validation table judgments: query ID, `0`, table ID, relevance grade. |
| `data/manifest.json` | Data checksums and the expected full-corpus hash. |

The full `Corpus.json` is external (about 237 MiB). Its expected local location is
`arxiv_data/SIGIRSciDis/tableGen/table_query_output/Corpus.json`; pass your actual
path with `--corpus`. A public download location is not documented, so obtain the
corpus from the dataset owner before building an index.

Instructor-only test qrels are private and are not distributed. Obtain an
approved file from the dataset owner and supply it with `--qrels`. Do not use
validation judgments or question metadata as replacement test labels.

## Commands

Complete the shared [setup](../../README.md#setup), then run from the repository root:

```bash
python -m table_retrieval.experiments.stage1.build_index \
  --corpus /path/to/Corpus.json --device cuda:0

python -m table_retrieval.experiments.stage1.evaluate --split val --device cuda:0

python -m table_retrieval.experiments.stage1.evaluate --split test \
  --qrels /path/to/approved_test_qrels.tsv --device cuda:0
```

Use `--device cpu` without a compatible GPU. The build writes
`results/stage1/index/`; both evaluations use that same index without rebuilding.
The build needs a new or empty output directory. Evaluation writes rankings and
metrics to `results/stage1/val/` and `results/stage1/test/`.

Use `--output` for another output directory and evaluation's `--index` for a
different completed index. Changed inputs or settings require a new output path.
These generated files stay under the gitignored `results/` directory.

To rescore saved rankings without running the models:

```bash
python -m table_retrieval.experiments.stage1.evaluate --split val \
  --rankings /path/to/saved_runs --output results/stage1/rescored
```

## Metrics

`metrics.py` reports per-query metrics averaged over questions with positive
judgments:

- Precision@k: relevant tables in the first k results, divided by k.
- Recall@k: relevant tables in the first k results, divided by all relevant tables.
- nDCG@k: binary relevance with logarithmic rank discounts and an ideal ranking.

Grades >= 1 count as relevant; higher grades receive no extra weight. Unjudged
tables count as non-gold. Missing query results score zero. Questions without
positive judgments are reported separately and excluded from the averages.

The default cutoff is 10; use `--k 5` for Top-5 metrics. The live system returns
Top-5 tables. This table benchmark uses different questions and judgments from
the Stage 2 cell benchmark.
