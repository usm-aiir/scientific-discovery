# Stage 2: Structural BGE cell retrieval

## Purpose

Reproduce training and evaluation of the Structural BGE cell retriever.
It ranks candidate cells from Stage 1's Top-5 tables and returns the global
Top-5 evidence cells.

## Input

Evaluation uses frozen Stage 1 tables and candidate cells. Each cell is formatted
with its value, labels and recovered structural context, then scored against the
question using BGE cosine similarity. The experiment uses the live cell formatter.

The snapshots include the tables and recovered context, so this experiment does
not need to rebuild Stage 1 or load the original HTML. Keep the candidate pools,
cell IDs and coordinates unchanged.

## Data

| File | Contents |
|---|---|
| `data/train.json` | 1,223 query/table examples with 2,829 positive cells. |
| `data/val.json` | 122 examples grouped into 93 validation questions. |
| `data/test.json` | 122 examples grouped into 91 test questions. |
| `data/val_candidates.json` | Frozen validation Top-5 tables, cells and structural context. |
| `data/test_candidates.json` | Frozen test Top-5 tables, cells and structural context. |
| `data/manifest.json` | Data checksums and expected training-pair counts. |

Gold evidence labels, coordinates and values are included in the annotation
files. `dataset.py` validates the fixed files and prepares model inputs; it does
not reshuffle the splits. These are the project's existing annotations, not a
newly human-verified benchmark.

## Training

Complete the shared [setup](../../README.md#setup). Run all commands from the
repository root:

```bash
python -m table_retrieval.experiments.stage2_bge.train --device cuda:0
```

Training downloads the pinned base BGE-M3 model and follows [config.json](config.json):
seed 42, three epochs, AdamW at 1e-5, batch size one, accumulation four,
four negatives per positive, temperature 0.02 and 512-token inputs.
The fixed recipe trains on table-aware cell text; evaluation adds recovered
structural context.

Validation selects the best epoch using full-ranking binary nDCG within each
annotated table. Test labels are not used for training or checkpoint selection.
Improving checkpoints and `training.json`, which identifies the selected epoch,
are written to `results/stage2_bge/train/`.

Use `--device cpu` without a compatible GPU, although training will be slower.
Use `--output /path/to/new_run` for another run. Existing outputs are protected.

## Evaluation

Evaluate the selected checkpoint on both fixed splits:

```bash
python -m table_retrieval.experiments.stage2_bge.evaluate \
  --checkpoint results/stage2_bge/train --split val --device cpu \
  --output results/stage2_bge/val

python -m table_retrieval.experiments.stage2_bge.evaluate \
  --checkpoint results/stage2_bge/train --split test --device cpu \
  --output results/stage2_bge/test
```

`--checkpoint` accepts a training run folder or a checkpoint directory. To score
the exact reported weights, replace `results/stage2_bge/train` with
`experiment2/checkpoint`. Evaluation uses float32, 512-token inputs and 32-cell
encoding batches. `--device cuda:0` also supports float32 experiment evaluation.

Evaluation writes `predictions.json` and `metrics.json`. Choose a new `--output`
directory for each run. Checkpoints and outputs belong under gitignored `results/`.
Rescore saved predictions without loading a model:

```bash
python -m table_retrieval.experiments.stage2_bge.evaluate --split test \
  --predictions results/stage2_bge/test/predictions.json
```

## Metrics

Reported test results use all 91 questions and 333 annotated gold cells:

| Metric | Test result |
|---|---:|
| Precision@5 | 23.08% |
| Recall@5 | 31.53% |
| F1@5 | 26.65% |
| nDCG@5 | 47.59% |

`metrics.py` pools cell counts for precision, recall and F1 (micro): 105 correct
cells among 455 returned cells. Gold cells are identified by `(table_id, cell_id)`;
unjudged cells count as non-gold.

nDCG@5 uses binary relevance in descending BGE score order:

```text
DCG@5  = sum(rel(rank) / log2(rank + 1), rank = 1..5)
IDCG@5 = sum(1 / log2(rank + 1), rank = 1..min(5, total annotated gold cells))
nDCG@5 = mean(DCG@5 / IDCG@5 over all 91 test questions)
```

All questions count equally, including those with zero Top-5 hits. Recall and
IDCG include gold cells missing from Stage 1; no availability filter is applied.
Only 265 of the 333 test gold cells are in the frozen candidate pools.

## Checkpoint

The trained checkpoint is external and is not committed to Git. The expected
checkpoint is `experiment2/checkpoint/`, relative to the repository root, with
the selected epoch-3 weights. Its `model.safetensors` SHA-256 must be:

```text
47bab214c0ec63d676586ab0632af33a6915ca9d306d1a3b2eaf026b85433c48
```

Supply this checkpoint to reproduce the exact reported weights. Fresh training
may produce different hashes or near-tie scores across hardware/software setups.
The live [example config](../../config.example.json) points to this checkpoint
and uses CPU float32 inference; the live CUDA selector uses float16 instead.
