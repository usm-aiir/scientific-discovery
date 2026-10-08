# Table retrieval code

The production inference path is deliberately small:

```text
Stage 1 fusion Top-5 tables
  -> pooled fine-tuned BGE cell ranking
  -> global Top-3 cells
```

## Runtime

- `data.py` defines shared input/output records and JSON helpers.
- `stage1/` loads existing BM25/BGE indexes and retrieves candidate tables.
- `stage2/structure.py` preserves original cell coordinates and reviewed layouts.
- `stage2/bge/model.py` constructs and ranks cells with BGE.
- `stage2/bge/pipeline.py`, `stage2/pipeline.py`, and `stage2/cli.py` expose Stage 2 inference.
- `pipeline.py` connects Stage 1 and Stage 2.
- `tests/` contains the canonical runtime and frozen-regression tests.

The runtime requires two external artifacts that are intentionally not committed:

1. a completed Stage 1 index, selected with `--stage1-index` or `--experiment`;
2. a local fine-tuned BGE checkpoint selected with `--checkpoint`.

It does not build indexes, train models, read qrels, or run evaluation during inference.

## Python integration

Initialize one pipeline when the service process starts, then reuse `search()` for
each request. The defaults are Stage 1 fusion Top-5, pooled fine-tuned BGE cell
ranking, and global Top-3 cells.

```python
from table_retrieval.pipeline import EvidencePipeline

pipeline = EvidencePipeline.from_settings(
    stage1_index="/path/to/index",
    checkpoint="/path/to/bge/checkpoint",
    stage1_device="cuda:0",
    cell_device="cuda:0",
)

result = pipeline.search(
    "What is the reported accuracy?",
    query_id="request-123",
)
```

The Stage 1 index and fine-tuned BGE checkpoint are external deployment
artifacts. `stage1_index`, `checkpoint`, `layouts_path`, and `settings_path` may
be absolute paths and are resolved during startup. A settings file is not
required just to choose devices. For GPU deployment, start with one initialized
`EvidencePipeline` per process and bounded request concurrency; unrestricted
concurrent CUDA inference is not guaranteed to be safe or memory-stable.

### Input and result contract

`search(query, query_id=None)` requires a nonempty string `query`. `query_id` is
optional and is returned as a string when supplied. The optional
`table_top_k` and `cell_top_k` arguments must be positive integers; integrations
should leave them at 5 and 3 to retain the canonical behavior.

The result has `schema_version`, `query_id`, normalized `query`,
`created_at_utc`, `configuration`, `status`, `tables`, `cells`, and `errors`.
Status is:

- `complete` when every candidate table was scored;
- `partial` when at least one table failed but other cells were scored;
- `error` when no candidate cells were available.

Each `tables` entry contains `table_id`, `retrieval_rank`, `retrieval_score`,
`status`, and `candidate_cell_count`. A failed table also contains `error` and
`error_type`.

Each selected `cells` entry contains `global_rank`, `table_id`, `cell_id`,
`row`, `column`, `value`, `row_label`, `header_path`, `header_source`,
`stage1_table_rank`, `stage1_table_score`, and `stage2_bge_score`. Row and column
coordinates are zero-based positions in the original `Corpus.json` rows,
including header rows. Stage 1 and Stage 2 scores are model-specific ranking
scores, not probabilities and not directly comparable to each other.

Startup/configuration failures raise Python exceptions. These include a missing
or incomplete Stage 1 index, missing checkpoint, invalid or unavailable device,
and invalid query or top-k configuration. Per-table candidate preparation
errors remain in the returned `status`/`errors` structure so one bad table does
not discard valid evidence. A global model inference failure raises an
exception because it affects the entire request.

### Deployment artifacts and paths

An integrating service must provide:

1. A completed Stage 1 index directory containing `prepared.json`, `index.json`,
   table IDs, BM25 data, dense embeddings, and its tokenizer. The corpus path in
   `prepared.json` may be absolute or relative to the index (or one of its
   parents). The referenced `Corpus.json` must remain available and match its
   recorded digest. Any embeddings symlink must also remain valid.
2. The Stage 1 base model/revision recorded by `prepared.json`. It must be
   available locally, in the standard Hugging Face cache, or downloadable under
   the deployment's normal Transformers configuration (`HF_HOME` and related
   environment variables).
3. A local fine-tuned BGE checkpoint directory for Stage 2.
4. Optionally, a reviewed-layout JSON path. The bundled
   `stage2/reviewed_layouts.json` is installed with the package but is not loaded
   implicitly; pass it explicitly when required by the deployment.

From a repository checkout, install the package with `python -m pip install -e .`.
The root package configuration excludes experiments, demos, generated datasets,
indexes, checkpoints, outputs, results, caches, and annotation data.

```bash
python -m table_retrieval.pipeline \
  --query "What is the reported accuracy?" \
  --stage1-index /path/to/stage1/index \
  --checkpoint /path/to/bge/checkpoint \
  --table-top-k 5 --cell-top-k 3 \
  --output evidence.jsonl
```

Stage 1 experiments and standalone Stage 2 inference remain available:

```bash
python -m table_retrieval.stage1.cli --model fusion --split val --experiment NAME
python -m table_retrieval.stage2.cli --query "..." --candidates candidates.json \
  --checkpoint /path/to/bge/checkpoint --output cells.json
```

## Research and reproducibility

`experiments/bge/` contains dataset preparation, fixed splitting, fine-tuning,
zero-shot evaluation, policy comparison, end-to-end evaluation, and the final
frozen evaluation. These scripts retain their historical output paths.

`experiments/tapas/` contains the TAPAS selector, per-table baseline pipeline,
training/evaluation, inspection notes, and synthetic-data preparation. TAPAS is
a research baseline, not the default Stage 2 runtime. The shared structural
mapping remains in `stage2/structure.py` because both BGE and TAPAS use it.

The entire `experiments/` tree and the demo are currently local-only and ignored
by Git. The commands below apply when those local research files are present.

Typical research commands are:

```bash
python -m table_retrieval.experiments.bge.train
python -m table_retrieval.experiments.bge.evaluate_final_end_to_end --device cpu
python -m table_retrieval.experiments.tapas.train
python -m table_retrieval.experiments.tapas.evaluate
```

Generated datasets remain under `stage2/bge/data/` and
`stage2/synthetic/{cleaned,rejected,reports,splits}/`. Results, outputs, model
checkpoints, source corpora, and local annotation data are also generated or
external. All are intentionally ignored by Git.
