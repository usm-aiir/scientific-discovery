# ModernBERT modality classifier

Train and evaluate ModernBERT to classify scientific questions by the modalities
needed to answer them: tables, figures, text, or a combination. Only the question
text enters the model.

## Organization

- **[modernbert_classifier/](modernbert_classifier/)** — Data loading, training,
  checkpoint loading, and evaluation helpers.
- **[evaluate.py](evaluate.py)** — Evaluate a trained checkpoint.

## Setup

Run commands from `router/`, using the existing environment and the parent
repository's `requirements.txt`:

```bash
source ../.venv-annotation/bin/activate
```

## Train

Check the training and validation data, then train:

```bash
python -m modernbert_classifier.train --check-data
python -m modernbert_classifier.train
```

Defaults: three epochs, learning rate `2e-5`, batch size 8, maximum query length
256, and seed 42. Training uses `data/Train.json`, evaluates `data/Val.json`
after each epoch, and automatically saves the checkpoint with the highest
validation routing accuracy. Ties keep the earliest epoch.

Each run creates new checkpoint and result directories under `model/` and
`results/`, preserving previous runs.

## Evaluate

```bash
python evaluate.py
```

Evaluation uses `data/Val.json` and, by default, the newest completed checkpoint
in `model/`, falling back to `model/modernbert/`. Use `--checkpoint PATH` to
choose a checkpoint. Predictions, metrics, and a report are saved in a new
`results/val_*/` directory. Evaluation reports:

- **Routing accuracy:** the selected route matches a known acceptable route.
- **Sufficient-set coverage:** the selected route contains a known sufficient set.
- **Mean selected modalities:** the average number of modalities selected.

These measure agreement with reviewed labels, not retrieval or answer quality.
Test review is unfinished; keep Test held out and use Val for development.

## Data

Use the individual `Train.json` and `Val.json` splits for training and evaluation.
Each record contains `router_id`, `query`, `acceptable_sets`, and `source_split`.
`[["table"], ["text"]]` means either modality is sufficient;
`[["table", "text"]]` means both are required. Unlisted routes are unverified.
