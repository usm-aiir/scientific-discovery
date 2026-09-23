# Scientific Table Retrieval

This project retrieves scientific tables using:

- BM25
- Pretrained BGE-M3
- Fine-tuned TAPAS
- BM25 + BGE-M3 fusion

The best-performing approach is BM25 + BGE-M3 fusion:

| Split | Recall@10 |
| --- | --- |
| Validation | 90.44% |
| Test | 89.02% |

## Setup

```bash
bash bin/install
conda activate table-dtr
```

The existing scraping, query-generation, and image-search tools use a separate
optional environment name. Both installers use the same `requirements.txt`:

```bash
bash bin/install --tools
conda activate scidiscovery
```

## Data

Place the following files in:

```text
arxiv_data/SIGIRSciDis/tableGen/table_query_output/
```

```text
Corpus.json
Val.json
Val_table_qrels.tsv
Test.json
Test_table_qrels.INSTRUCTOR_ONLY.tsv
```

Reproducing the TAPAS training experiment also requires:

```text
Train.json
Train_table_qrels.tsv
```

Data, model weights, indexes, and generated rankings are not included in Git.

## Run retrieval

Run the recommended fusion model on validation:

```bash
python -m table_retrieval --model fusion --split val
```

Run it on test after completing validation:

```bash
python -m table_retrieval --model fusion --split test
```

To reproduce an individual model result, replace `fusion` with `bm25`, `bge`, or `tapas`:

```bash
python -m table_retrieval --model bge --split val
```

TAPAS retrieval requires a trained checkpoint. Its training code is retained for experiment reproduction.

Running the package without arguments displays help:

```bash
python -m table_retrieval
```

## Repository structure

```text
table_retrieval/
├── __main__.py
├── data.py
├── pipeline.py
├── fusion.py
├── evaluation.py
├── settings.json
├── retrievers/
│   ├── bm25.py
│   ├── bge.py
│   └── tapas.py
└── training/
    ├── tapas.py
    └── convert.py
```

- `pipeline.py` runs the complete retrieval workflow.
- `retrievers/` contains the model-specific retrieval code.
- `fusion.py` combines BM25 and BGE-M3 results.
- `evaluation.py` calculates retrieval metrics.
- `training/` contains the research-only TAPAS training code.
