# Stage 2: Structural BGE

This is the reproducible path for Experiment 3, arm `B_bge_structural`:
Stage 1's five retrieved tables → structural cell text → BGE cosine ranking
→ five evidence cells. The checkpoint was trained in Experiment 2 on
table-aware inputs. Recovered structure was added when ranking cells; the
reported run did not retrain on recovered HTML text.

## Check the reported metrics

From the repository root, install the project and verify the supplied data:

```bash
pip install -e . -r table_retrieval/experiments/stage2_bge/requirements.txt
python -m table_retrieval.experiments.stage2_bge.prepare
python -m table_retrieval.experiments.stage2_bge.evaluate --saved --split test
```

The last command scores the original, complete rankings in `reference/`
against the committed test labels. It checks every query, candidate table,
and cell ID. It performs no model inference.

| Split | Queries | Gold cells | Correct@5 | Precision@5 | Recall@5 | nDCG@5 |
|---|---:|---:|---:|---:|---:|---:|
| Validation | 93 | 240 | 103 | 22.15% | 42.92% | 52.93% |
| Test | 91 | 333 | 105 | 23.08% | 31.53% | 47.59% |

Precision and recall pool counts across questions (micro). nDCG@5 is binary
discounted gain averaged across questions; the ideal ranking includes all
gold cells up to five, including evidence absent from Stage 1. Test Stage 1
contains 265/333 gold cells (79.58%). Its table Precision@5 and Recall@5 are
22.42% and 83.61% against the Stage 2 annotations. These differ from Stage 1's
250-query table benchmark.

These are the five ranked cells before evidence aggregation. The historical
34.83% recall after aggregation is a separate output and is not a metric of
the live Stage 2 Top-5 response. `reference/metrics.json` is calculated from
the ranked predictions; nDCG@5 was added here, not copied from the old report.

## Train and evaluate BGE

```bash
python -m table_retrieval.experiments.stage2_bge.train --device cuda:0
python -m table_retrieval.experiments.stage2_bge.evaluate \
  --checkpoint table_retrieval/experiments/stage2_bge/runs/reported \
  --split val --device cuda:0 \
  --output table_retrieval/experiments/stage2_bge/runs/validation
python -m table_retrieval.experiments.stage2_bge.evaluate \
  --checkpoint table_retrieval/experiments/stage2_bge/runs/reported \
  --split test --device cuda:0 \
  --output table_retrieval/experiments/stage2_bge/runs/test
```

`config.json` fixes the original recipe: BAAI/bge-m3 revision
`5617a9f61b028005a4858fdac845db406aefb181`, 1,223 training examples,
2,829 positive pairs, four negatives per positive, seed 42, three epochs,
AdamW at 1e-5, batch size one, accumulation four, temperature 0.02, and
512-token inputs. Each improving epoch is saved. `training.json` records
the selected checkpoint, input settings, environment, and checkpoint hashes.
The selected epoch uses full-ranking validation nDCG within the annotated
table, as in the original training. No test labels enter training or epoch
selection. The separate data check reads test labels only to validate their
coordinates and split isolation.

Inference uses float32, stable score ties in candidate-table/row/column order,
and the live structural formatter. The original run used Torch 2.14.0+cu130
and Transformers 5.17.0; those versions and checkpoint checksums are recorded
in `reference/provenance.json`. Training uses bfloat16 on a supported GPU,
otherwise float32. CPU works but full BGE-M3 training is expensive. Different
software or hardware can change scores near ties and retrained weight hashes.

## Fixed data

- `data/train.json`, `val.json`, `test.json`: original annotated table snapshots
  with evidence coordinates and values. Validation and test have 122 annotated
  query/table pairs each, grouped into 93 and 91 questions.
- `data/*_cell_qrels.tsv`: the same validation and test evidence as explicit judgments.
- `data/*_candidates.json`: exact Stage 1 Top-5 table order, table snapshots,
  and recovered structural context for every candidate cell. These include
  distractor tables. They preserve the full-corpus retrieval experiment without
  requiring its large index or the original HTML directory.
- `data/manifest.json`: SHA-256 checksums and training-pair counts. The prepare
  command checks coordinates, labels, split isolation, candidate pools, and
  both training recipes. It does not regenerate or reshuffle the splits.
- `reference/`: complete original Structural BGE rankings, derived metrics,
  training histories, and source/checkpoint hashes.

The annotations came from the project's scientific-table annotation workflow;
they are not a new human-verified benchmark. Original provenance fields,
including historical local paths, are retained in the split files but are not
opened by these scripts. Published metrics treat unjudged cells as non-gold.

Stage 1 is frozen for the reported comparison. Changing its index or corpus
creates a new experiment; the reported numbers apply to these exact candidates.
See `../stage1/README.md` to build Stage 1 against the original full corpus.

## Reproduce the currently configured runtime checkpoint

The live `table_retrieval/config.json` currently points to the earlier
`results/stage2/bge_finetuned/best_model`, trained on 975 examples using legacy
row/header/value text. Its weights are distinct from Experiment 2's checkpoint.
For that training recipe use:

```bash
python -m table_retrieval.experiments.stage2_bge.train \
  --config table_retrieval/experiments/stage2_bge/runtime.json \
  --output table_retrieval/experiments/stage2_bge/runs/runtime --device cuda:0
```

`runtime.json` uses `data/runtime_train.json` (975 examples; 2,241 positive
pairs). It shares the same fixed validation/test split and training settings.
`reference/runtime_training.json` records the original training result. The
31.53% result belongs to the main recipe above, not this earlier checkpoint.

Either saved checkpoint can be loaded by live Stage 2: set `stage2_checkpoint`
in `table_retrieval/config.json` to the chosen epoch directory. Relative paths
there resolve from `table_retrieval/`. Structural source recovery remains
optional in live retrieval; the benchmark uses its committed recovered context.
The scripts do not change the live checkpoint automatically.
