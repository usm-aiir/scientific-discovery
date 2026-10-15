# Offline retrieval evaluation

This directory contains Stage 1 index construction, saved-run generation, and
metric calculation. It is not imported by the live retrieval pipeline.

Its separate `settings.json` describes datasets and evaluation outputs. Run it
only when rebuilding or measuring retrieval artifacts:

```bash
python -m table_retrieval.evaluation.cli --model fusion --split val
```
