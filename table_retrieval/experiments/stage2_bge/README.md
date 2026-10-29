# Stage 2: Cell retrieval

Rank evidence cells within Stage 1's Top-5 tables using BGE-M3. Each cell's text
includes its value and structural context, such as row labels and column headers.
Training, evaluation, and `live/stage2/` use the same cell formatter and BGE encoder.

## Data

Training and evaluation read Behrooz's v2 dataset by default:

```text
/mnt/netstore1_home/behrooz.mansouri/SciDis/tableGen/table_query_output_v2/
```

The loader reads `Corpus.json`, the question split (`Train.json`, `Val.json`, or
`Test.json`), and its table relevance labels and matched-cell TSVs. Test label
filenames include `.INSTRUCTOR_ONLY`; the loader selects them automatically.

First run [Stage 1](../stage1/README.md) to produce the candidate table rankings.
Use rankings generated from the same dataset, and keep the Stage 1 output folder
intact. Stage 2 reads these saved rankings.

## Train

For individual Python commands, activate the environment from the
[experiment setup](../README.md#setup), then run from the repository root:

```bash
python -m table_retrieval.experiments.stage2_bge.train
```

Training reads `results/stage1/val/fusion.run` and saves to
`results/stage2_bge/train/` by default. Use `--val-stage1-run` or `--output` to
choose other paths.

Training settings are in [config.json](config.json). Positive cells compete with
four negatives sampled from that question's explicitly irrelevant training
tables. Questions without those negative tables are excluded and recorded.

The model is selected by validation nDCG@5 on the saved Stage 1 candidates.

## Evaluate

Evaluate pretrained BGE-M3 on the saved validation tables:

```bash
python -m table_retrieval.experiments.stage2_bge.evaluate
```

This reads `results/stage1/val/fusion.run` and saves to
`results/stage2_bge/val_pretrained/`. To evaluate the selected training checkpoint:

```bash
python -m table_retrieval.experiments.stage2_bge.evaluate \
  --checkpoint results/stage2_bge/train
```

Checkpoint results go to `results/stage2_bge/val_trained/` by default.
Adding `--split test` selects `results/stage1/test/fusion.run` and the corresponding
`test_pretrained/` or `test_trained/` output folder automatically.
`--checkpoint` accepts a training run folder or an individual checkpoint.
Validation and evaluation use float32 weights and 512-token inputs.

## Options

| Option | Purpose |
|---|---|
| `--data-dir` | Read another v2 dataset folder. |
| `--stage1-run` | Evaluate another saved Stage 1 ranking. |
| `--val-stage1-run` | Train using another saved validation ranking. |
| `--output` | Save to another new folder. |
| `--config` | Use another training settings file. |
| `--check-data` | Check training inputs without loading a model or writing outputs. |
| `--device cpu` or `--device cuda:0` | Select a device; default is automatic. |
| `--k 5` | Report evaluation metrics at Top-5 only. |

Commands check that Stage 1 rankings and their metadata exist before loading the
data or model. Missing inputs produce an error explaining which Stage 1 command
to run. Existing output folders are preserved; use `--output` for a new run.
