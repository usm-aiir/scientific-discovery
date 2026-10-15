# Stage 1: table retrieval

Build the BM25 and BGE-M3 indexes, fuse their rankings, and measure table
Recall@10. The validation and test queries and table judgments are in `data/`.
These are the 250-query table retrieval splits, distinct from the Stage 2
cell annotations.

Place the original full `Corpus.json` at the `corpus` path in `settings.json`,
or edit that path. Its SHA-256 is recorded in `data/manifest.json`. The full
corpus is about 237 MiB and is distributed separately; this repository does
not provide a public download URL for it. Stage 2 includes its exact frozen
candidate tables and can run without the full corpus.

From the repository root, after `pip install -e .`:

```bash
python -m table_retrieval.experiments.stage1.cli --model fusion --split val --experiment reproduce
python -m table_retrieval.experiments.stage1.cli --model fusion --split test --experiment reproduce
```

`settings.json` fixes BGE's revision, 2,048-token inputs, 100 candidates per
retriever, and RRF constant 60. Test reuses the validation index. Outputs go
under `outputs/`; existing incompatible runs require a new `--experiment` name.
This code imports the live Stage 1 retrievers and fusion implementation.

The supplied test qrels are the original `INSTRUCTOR_ONLY` judgment set used
by the saved Stage 1 experiment. Alternative local MLLM/PENCILS judgments are
different evaluations and are not substituted here.
