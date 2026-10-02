# Optional TAPAS experiment

The production Stage 1 retriever is BM25 + BGE fusion. This folder keeps the
TAPAS experiment used for comparison:

- `tapas.py` contains TAPAS table preparation, retrieval, indexing, and training.
- `convert_tapas.py` converts the original TensorFlow checkpoint when needed.

Run TAPAS retrieval with the normal Stage 1 command:

```bash
python -m table_retrieval.stage1.cli --model tapas --split val
```

Train the experimental TAPAS retriever:

```bash
python -m table_retrieval.stage1.experiments.tapas
```

Both commands use `table_retrieval/settings.json`. The experiment reuses the
shared data, ranking, and evaluation code in Stage 1.
