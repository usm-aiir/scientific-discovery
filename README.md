# Scientific Table Retrieval

Stage 1 retrieves relevant scientific tables using:

- **BM25**
- **Pretrained BGE-M3**
- **BM25 + BGE-M3 fusion**
- **TAPAS** as an optional comparison

The best-performing method is **BM25 + BGE-M3 fusion**.

| Split | Recall@10 |
| --- | --- |
| Validation | 90.44% |
| Test | 89.02% |

## Setup

```bash
bash bin/install
conda activate table-dtr
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

## Run

Run the fusion model on validation:

```bash
python -m table_retrieval.stage1.cli --model fusion --split val
```

Run on test:

```bash
python -m table_retrieval.stage1.cli --model fusion --split test
```

Run an individual retriever:

```bash
python -m table_retrieval.stage1.cli --model bge --split val
```

Available models:

```text
bm25
bge
fusion
tapas //a comparison experiment
```

Use a different judgment file with:

```bash
python -m table_retrieval.stage1.cli \
    --model fusion \
    --split test \
    --qrels path/to/judgments.tsv
```

Use `--experiment` when running a new experiment with different data or settings.

## Table representation

The main BM25 and BGE pipeline uses:

```text
paper title + caption + table headers + table rows
```

Subcaptions and reference text are not included.

## Notes

- Existing rankings are reused when the experiment data and settings match.
- Missing rankings are generated automatically.
- TAPAS is retained as an optional comparison experiment.
- Stage 1 settings are stored in `table_retrieval/settings.json`.
