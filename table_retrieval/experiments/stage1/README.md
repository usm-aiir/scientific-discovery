# Stage 1: Table retrieval

Compare three methods for finding tables relevant to a question:

- **BM25:** keyword search over the table text included in BGE inputs.
- **BGE-M3:** similarity search using pretrained BGE embeddings.
- **Fusion:** combines BM25 and BGE results using their rank positions.

The index uses 2,048-token table inputs, 100 candidates per retriever, and
reciprocal rank fusion with constant 60.
The retrieval implementations are shared with `live/stage1/`.

## Data

All commands read Behrooz's v2 dataset by default:

```text
/mnt/netstore1_home/behrooz.mansouri/SciDis/tableGen/table_query_output_v2/
```

Indexing reads `Corpus.json`. Evaluation reads `Val.json` or `Test.json` and
its table relevance labels: `Val_table_qrels.tsv` or
`Test_table_qrels.INSTRUCTOR_ONLY.tsv`. Each evaluation split has 250 questions.

## Run

For individual Python commands, activate the environment from the
[experiment setup](../README.md#setup), then run from the repository root:

```bash
# Build the index once.
python -m table_retrieval.experiments.stage1.build_index

# Evaluate validation, then test, using that index.
python -m table_retrieval.experiments.stage1.evaluate
python -m table_retrieval.experiments.stage1.evaluate --split test
```

The index goes to `results/stage1/index/`. Rankings and metrics go to
`results/stage1/val/` and `results/stage1/test/`.
Each evaluation saves `fusion.run` in its split's folder by default, where
Stage 2 reads it automatically.

## Options

| Option | Purpose |
|---|---|
| `--data-dir` | Read another dataset folder. |
| `--corpus` | Use another corpus file when building the index. |
| `--queries`, `--qrels` | Use other questions or table labels for evaluation. |
| `--index` | Evaluate a different completed index. |
| `--output` | Save to another output folder. |
| `--model` | Choose `bm25`, `bge`, or `fusion` (default). |

Individual file options override `--data-dir`. Use `--device cpu` or
`--device cuda:0` to select a device; the default chooses automatically.
Build into a new or empty folder, and use new output folders when changing inputs
or settings.

To score saved rankings without running retrieval again:

```bash
python -m table_retrieval.experiments.stage1.evaluate \
  --rankings results/stage1/val --output results/stage1/rescored
```

## Metrics

Each evaluation saves one `metrics.json` containing overall scores for each
retrieval method. Per-query scores are omitted; the `.run` files retain the rankings.

Precision, Recall, and nDCG use the first 10 results by default; change this with
`--k`. **MRR** averages `1 / rank` of the first relevant table across the full
saved ranking, with zero when none is found. Questions without positive relevance
labels are excluded from the averages and listed separately.
