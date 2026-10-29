# Retrieval experiments

Run table retrieval (Stage 1), then train and evaluate cell retrieval (Stage 2).
Data can be found here:

```text
/mnt/netstore1_home/behrooz.mansouri/SciDis/tableGen/table_query_output_v2/
```

Run the commands below from the repository root. The scripts also work from
other directories when called by their full paths.

## Setup

Python 3.10 or newer is required. Setup tries `python3.10`, then `python3`.

```bash
bash table_retrieval/experiments/scripts/setup.sh
```

This creates `table_retrieval/experiments/.venv`, installs the pinned dependencies
from the root `requirements.txt`, installs the retrieval package, and checks the
dependencies. Both stages use this environment automatically.

To select another Python executable, set `PYTHON` when running setup:

```bash
PYTHON=/path/to/python3.10 bash table_retrieval/experiments/scripts/setup.sh
```

## Run everything

```bash
bash table_retrieval/experiments/scripts/run_all.sh
```

This runs Stage 1 followed by Stage 2, stopping if any step fails.

To run the stages separately:

```bash
bash table_retrieval/experiments/scripts/run_stage1.sh
bash table_retrieval/experiments/scripts/run_stage2.sh
```

- **Stage 1:** build the BM25/BGE index, then evaluate validation and test questions.
  A completed index is reused.
- **Stage 2:** train BGE using training data and validation-based checkpoint
  selection, then evaluate pretrained and selected models on validation and test.
  It requires the completed Stage 1 rankings.

## Results

Results are saved under the repository's `results/` folder:

| Folder | Contents |
|---|---|
| `stage1/index/` | Reusable table index. |
| `stage1/val/`, `stage1/test/` | Table rankings and `metrics.json`. |
| `stage2_bge/train/` | Checkpoints and training history. |
| `stage2_bge/val_pretrained/`, `stage2_bge/val_trained/` | Validation predictions and metrics. |
| `stage2_bge/test_pretrained/`, `stage2_bge/test_trained/` | Test predictions and metrics. |

Each evaluation saves one aggregate `metrics.json`, including MRR. Per-query
scores are omitted; ranking and prediction files retain the detailed results.
Existing Stage 2 output folders are protected. Choose a new results directory
for another full run.

For individual Python commands and file overrides, first activate the environment:

```bash
source table_retrieval/experiments/.venv/bin/activate
```

See the [Stage 1](stage1/README.md) and [Stage 2](stage2_bge/README.md) instructions.
