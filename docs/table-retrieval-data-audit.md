# Retrieval data utility usage audit

Audit completed before refactoring `table_retrieval/data.py`. Import locations below are the pre-refactor snapshot; line numbers may shift after imports are updated.

Scope: repository-wide hidden/ignored source search, including Python scripts under `results/`, numbered experiments, tests, and generated `build/lib` files. Parsed all 173 Python files then present; local virtual environments, `.git`, and `node_modules` were excluded. Text searches also checked Python, Markdown, notebook, shell, and project configuration references. Direct imports, relative imports, aliases, and module-qualified accesses were inspected. None of the 15 requested symbols is genuinely unused.

## Decisions

| Classification | Symbols | Decision |
| --- | --- | --- |
| Live/shared | `load_json`, `check_id`, `normalize_table_id`, `digest`, `load_corpus`, `SCHEMA_VERSION`, `Candidate`, `write_predictions` | Keep in shared `data.py`. |
| Indexing/evaluation generic file I/O | `write_json` | Keep in shared `data.py`, as requested. |
| Evaluation/experiment formats | `write_run`, `read_run`, `load_queries`, `load_qrels` | Move verbatim into `evaluation/data.py`; update all non-generated callers. |
| Historical configuration dependency | `DATA` | Keep: the historical inspection script still uses it. Add a comment distinguishing it from live configuration. |
| Experiment schema metadata | `COORDINATES` | Keep: TAPAS and aggregation outputs use it. |
| Unused | None | Remove no symbols as dead code. |

## Per-symbol imports and use

### `DATA`

Historical experiment only; retained. Imported and used by results/inspection/compare_bm25_titles.py for Corpus.json, Val.json, and judgments. The script also has older baseline/evaluator imports; that does not justify silently removing its DATA dependency. Live retrieval obtains paths from configuration.

Source imports:

- `results/inspection/compare_bm25_titles.py:5` — historical experiment/audit.

### `load_json`

Live/shared, indexing, training, evaluation, experiments, CLI, tests; retained. Shared UTF-8 JSON reader. Also called inside data.py by write_json and load_corpus.

Source imports:

- `experiment1/run_experiment.py:27` — historical experiment/training.
- `experiment2/evaluate.py:15` — historical experiment/training.
- `experiment2/train.py:11` — historical experiment/training.
- `experiment4/report.py:8` — historical experiment/training.
- `experiment4/run.py:16` — historical experiment/training.
- `results/inspection/compare_bm25_titles.py:5` — historical experiment/audit.
- `table_retrieval/evaluation/config.py:5` — offline evaluation/CLI.
- `table_retrieval/evaluation/fusion.py:2` — offline evaluation/CLI.
- `table_retrieval/evaluation/indexing.py:5` — offline indexing/evaluation.
- `table_retrieval/evaluation/metrics.py:7` — offline evaluation/CLI.
- `table_retrieval/evaluation/retrievers/bge.py:6` — offline indexing/evaluation.
- `table_retrieval/evaluation/retrievers/bm25.py:5` — offline indexing/evaluation.
- `table_retrieval/evaluation/workflow.py:5` — offline evaluation/CLI.
- `table_retrieval/experiments/bge/evaluate_aggregation.py:13` — experiment/evaluation/metadata.
- `table_retrieval/experiments/bge/evaluate_end_to_end.py:9` — experiment/evaluation/metadata.
- `table_retrieval/experiments/bge/evaluate_final_end_to_end.py:9` — experiment/evaluation/metadata.
- `table_retrieval/experiments/bge/evaluate_finetuned.py:7` — experiment/evaluation/metadata.
- `table_retrieval/experiments/bge/evaluate_global_top3.py:9` — experiment/evaluation/metadata.
- `table_retrieval/experiments/bge/evaluate_seed_diversity.py:9` — experiment/evaluation/metadata.
- `table_retrieval/experiments/bge/evaluate_selection_policies.py:6` — experiment/evaluation/metadata.
- `table_retrieval/experiments/bge/evaluate_splits.py:8` — experiment/evaluation/metadata.
- `table_retrieval/experiments/bge/evaluate_top3.py:6` — experiment/evaluation/metadata.
- `table_retrieval/experiments/bge/evaluation.py:9` — experiment/evaluation/metadata.
- `table_retrieval/experiments/bge/prepare_splits.py:8` — training/data preparation.
- `table_retrieval/experiments/bge/train.py:11` — training/data preparation.
- `table_retrieval/experiments/demo/app.py:23` — demo runtime.
- `table_retrieval/experiments/stage1/audit_test_qrels.py:10` — experiment/evaluation/metadata.
- `table_retrieval/experiments/tapas/io.py:4` — experiment/evaluation/metadata.
- `table_retrieval/experiments/tapas/synthetic/split.py:10` — training/data preparation.
- `table_retrieval/experiments/tapas/synthetic/validate_tapas.py:9` — experiment/evaluation/metadata.
- `table_retrieval/experiments/tapas/train.py:8` — training/data preparation.
- `table_retrieval/search.py:6` — live CLI.
- `table_retrieval/stage1/artifacts.py:4` — live/shared runtime.
- `table_retrieval/stage1/pipeline.py:5` — live/shared runtime.
- `table_retrieval/stage1/retrievers/bge.py:8` — live/shared runtime.
- `table_retrieval/stage2/cli.py:7` — live CLI.

Additional module-qualified uses:

- `table_retrieval/evaluation/tests/test_cli.py:35`.
- `table_retrieval/evaluation/tests/test_cli.py:63`.
- `table_retrieval/evaluation/tests/test_cli.py:88`.
- `table_retrieval/experiments/stage1/tests/test_bge_comparison.py:65`.

Stale generated imports (reported, not edited):

- `build/lib/table_retrieval/stage1/evaluation.py:12`.
- `build/lib/table_retrieval/stage1/fusion.py:5`.
- `build/lib/table_retrieval/stage1/pipeline.py:11`.
- `build/lib/table_retrieval/stage1/retrievers/bge.py:14`.
- `build/lib/table_retrieval/stage1/retrievers/bm25.py:12`.
- `build/lib/table_retrieval/stage2/cli.py:6`.
- `build/lib/table_retrieval/stage2/pipeline.py:2`.

### `check_id`

Live/shared, indexing, training-data preparation, evaluation; retained. Used by live BM25 and load_corpus; the moved evaluation helpers continue importing this same implementation.

Source imports:

- `table_retrieval/evaluation/fusion.py:2` — offline evaluation/CLI.
- `table_retrieval/evaluation/indexing.py:5` — offline indexing/evaluation.
- `table_retrieval/evaluation/metrics.py:7` — offline evaluation/CLI.
- `table_retrieval/evaluation/retrievers/bge.py:6` — offline indexing/evaluation.
- `table_retrieval/evaluation/retrievers/bm25.py:5` — offline indexing/evaluation.
- `table_retrieval/evaluation/text.py:4` — offline indexing/evaluation.
- `table_retrieval/experiments/bge/prepare_splits.py:8` — training/data preparation.
- `table_retrieval/experiments/tapas/synthetic/split.py:10` — training/data preparation.
- `table_retrieval/stage1/retrievers/bm25.py:6` — live/shared runtime.

Stale generated imports (reported, not edited):

- `build/lib/table_retrieval/stage1/evaluation.py:12`.
- `build/lib/table_retrieval/stage1/fusion.py:5`.
- `build/lib/table_retrieval/stage1/pipeline.py:11`.
- `build/lib/table_retrieval/stage1/retrievers/bge.py:14`.
- `build/lib/table_retrieval/stage1/retrievers/bm25.py:12`.
- `build/lib/table_retrieval/stage1/text.py:2`.

### `normalize_table_id`

Live/shared indirectly, evaluation, experiments; retained. load_corpus calls this to reject normalized-ID collisions during live Stage 1 loading. Moving it into evaluation would create an unnecessary live-to-evaluation dependency. The regular expression and empty-subtable handling remain unchanged.

Source imports:

- `results/data_quality_audit/additional_checks.py:6` — historical experiment/audit.
- `results/data_quality_audit/audit.py:7` — historical experiment/audit.
- `results/test_qrels_comparison/evaluate_all.py:6` — tests.
- `table_retrieval/evaluation/metrics.py:7` — offline evaluation/CLI.
- `table_retrieval/experiments/stage1/audit_test_qrels.py:10` — experiment/evaluation/metadata.

Stale generated imports (reported, not edited):

- `build/lib/table_retrieval/stage1/evaluation.py:12`.

### `digest`

Live/shared, indexing, training, evaluation, experiments, tests; retained. Stage 1 artifact validation and Stage 2 structural provenance require hashing.

Source imports:

- `experiment1/run_experiment.py:27` — historical experiment/training.
- `experiment4/run.py:16` — historical experiment/training.
- `results/test_qrels_comparison/evaluate_all.py:6` — tests.
- `table_retrieval/evaluation/config.py:5` — offline evaluation/CLI.
- `table_retrieval/evaluation/indexing.py:5` — offline indexing/evaluation.
- `table_retrieval/evaluation/metrics.py:7` — offline evaluation/CLI.
- `table_retrieval/evaluation/retrievers/bge.py:6` — offline indexing/evaluation.
- `table_retrieval/evaluation/retrievers/bm25.py:5` — offline indexing/evaluation.
- `table_retrieval/experiments/bge/evaluate_aggregation.py:13` — experiment/evaluation/metadata.
- `table_retrieval/experiments/bge/evaluate_end_to_end.py:9` — experiment/evaluation/metadata.
- `table_retrieval/experiments/bge/evaluate_final_end_to_end.py:9` — experiment/evaluation/metadata.
- `table_retrieval/experiments/bge/evaluate_finetuned.py:7` — experiment/evaluation/metadata.
- `table_retrieval/experiments/bge/evaluate_global_top3.py:9` — experiment/evaluation/metadata.
- `table_retrieval/experiments/bge/evaluate_seed_diversity.py:9` — experiment/evaluation/metadata.
- `table_retrieval/experiments/bge/evaluate_selection_policies.py:6` — experiment/evaluation/metadata.
- `table_retrieval/experiments/bge/evaluate_splits.py:8` — experiment/evaluation/metadata.
- `table_retrieval/experiments/bge/evaluate_top3.py:6` — experiment/evaluation/metadata.
- `table_retrieval/experiments/bge/prepare_splits.py:8` — training/data preparation.
- `table_retrieval/experiments/bge/train.py:11` — training/data preparation.
- `table_retrieval/experiments/stage1/audit_test_qrels.py:10` — experiment/evaluation/metadata.
- `table_retrieval/experiments/tapas/pipeline.py:3` — experiment/evaluation/metadata.
- `table_retrieval/experiments/tapas/synthetic/split.py:10` — training/data preparation.
- `table_retrieval/experiments/tapas/synthetic/validate_tapas.py:9` — experiment/evaluation/metadata.
- `table_retrieval/experiments/tests/test_pipeline_legacy.py:10` — tests.
- `table_retrieval/stage1/artifacts.py:4` — live/shared runtime.
- `table_retrieval/stage1/pipeline.py:5` — live/shared runtime.
- `table_retrieval/stage1/retrievers/bge.py:8` — live/shared runtime.
- `table_retrieval/stage2/structure.py:4` — live/shared runtime.

Additional module-qualified uses:

- `table_retrieval/evaluation/tests/test_cli.py:105`.
- `table_retrieval/evaluation/tests/test_cli.py:126`.

Stale generated imports (reported, not edited):

- `build/lib/table_retrieval/stage1/evaluation.py:12`.
- `build/lib/table_retrieval/stage1/pipeline.py:11`.
- `build/lib/table_retrieval/stage1/retrievers/bge.py:14`.
- `build/lib/table_retrieval/stage1/retrievers/bm25.py:12`.
- `build/lib/table_retrieval/stage2/bge/pipeline.py:7`.

### `write_json`

Indexing/evaluation and tests; retained shared file helper. No current live retrieval writer calls this helper. It remains in data.py as requested: it is generic safe JSON artifact I/O, not an evaluation format.

Source imports:

- `table_retrieval/evaluation/indexing.py:5` — offline indexing/evaluation.
- `table_retrieval/evaluation/metrics.py:7` — offline evaluation/CLI.
- `table_retrieval/evaluation/retrievers/bge.py:6` — offline indexing/evaluation.
- `table_retrieval/evaluation/workflow.py:5` — offline evaluation/CLI.

Additional module-qualified uses:

- `table_retrieval/evaluation/tests/test_cli.py:103`.
- `table_retrieval/evaluation/tests/test_cli.py:124`.
- `table_retrieval/evaluation/tests/test_cli.py:68`.
- `table_retrieval/evaluation/tests/test_cli.py:75`.
- `table_retrieval/evaluation/tests/test_cli.py:77`.
- `table_retrieval/evaluation/tests/test_cli.py:80`.

Stale generated imports (reported, not edited):

- `build/lib/table_retrieval/stage1/evaluation.py:12`.
- `build/lib/table_retrieval/stage1/pipeline.py:11`.
- `build/lib/table_retrieval/stage1/retrievers/bge.py:14`.

### `write_run`

Evaluation workflow and tests; moved. Moved verbatim to table_retrieval/evaluation/data.py. TREC six-column output, score formatting, ordering, and overwrite/partial-file protection are unchanged.

Source imports:

- `table_retrieval/evaluation/workflow.py:5` — offline evaluation/CLI.

Additional module-qualified uses:

- `table_retrieval/evaluation/tests/test_cli.py:84`.

Stale generated imports (reported, not edited):

- `build/lib/table_retrieval/stage1/pipeline.py:11`.

### `read_run`

Evaluation and historical experiments; moved. Moved verbatim to table_retrieval/evaluation/data.py, including the caller under results/test_qrels_comparison/.

Source imports:

- `results/test_qrels_comparison/evaluate_all.py:6` — tests.
- `table_retrieval/evaluation/fusion.py:2` — offline evaluation/CLI.
- `table_retrieval/evaluation/metrics.py:7` — offline evaluation/CLI.
- `table_retrieval/experiments/stage1/audit_test_qrels.py:10` — experiment/evaluation/metadata.

Stale generated imports (reported, not edited):

- `build/lib/table_retrieval/stage1/evaluation.py:12`.
- `build/lib/table_retrieval/stage1/fusion.py:5`.
- `build/lib/table_retrieval/stage1/pipeline.py:11`.

### `load_corpus`

Live/shared and offline indexing/evaluation; retained. Live Stage 1 resolves the recorded corpus and validates its contents through this function.

Source imports:

- `table_retrieval/evaluation/indexing.py:5` — offline indexing/evaluation.
- `table_retrieval/evaluation/retrievers/bm25.py:5` — offline indexing/evaluation.
- `table_retrieval/experiments/stage1/audit_test_qrels.py:10` — experiment/evaluation/metadata.
- `table_retrieval/stage1/artifacts.py:4` — live/shared runtime.

Stale generated imports (reported, not edited):

- `build/lib/table_retrieval/stage1/pipeline.py:11`.
- `build/lib/table_retrieval/stage1/retrievers/bm25.py:12`.

### `load_queries`

Offline preparation/indexing, evaluation, experiments; moved. Moved verbatim to table_retrieval/evaluation/data.py. Neither live search CLI nor the Stage 2 CLI loads an evaluation query dataset.

Source imports:

- `table_retrieval/evaluation/fusion.py:2` — offline evaluation/CLI.
- `table_retrieval/evaluation/indexing.py:5` — offline indexing/evaluation.
- `table_retrieval/evaluation/metrics.py:7` — offline evaluation/CLI.
- `table_retrieval/evaluation/retrievers/bge.py:6` — offline indexing/evaluation.
- `table_retrieval/evaluation/retrievers/bm25.py:5` — offline indexing/evaluation.
- `table_retrieval/experiments/stage1/audit_test_qrels.py:10` — experiment/evaluation/metadata.

Stale generated imports (reported, not edited):

- `build/lib/table_retrieval/pipeline.py:61`.
- `build/lib/table_retrieval/stage1/evaluation.py:12`.
- `build/lib/table_retrieval/stage1/fusion.py:5`.
- `build/lib/table_retrieval/stage1/pipeline.py:11`.
- `build/lib/table_retrieval/stage1/retrievers/bge.py:14`.
- `build/lib/table_retrieval/stage1/retrievers/bm25.py:12`.

### `load_qrels`

Evaluation only; moved. Moved verbatim to table_retrieval/evaluation/data.py. This is the four-column table-qrels reader. The similarly named BGE cell-qrels reader and figure evaluator functions are distinct formats and remain separate.

Source imports:

- `table_retrieval/evaluation/metrics.py:7` — offline evaluation/CLI.

Stale generated imports (reported, not edited):

- `build/lib/table_retrieval/stage1/evaluation.py:12`.

### `COORDINATES`

Experiment output metadata; retained. TAPAS selection/pipeline output and aggregation output use this coordinate-system string. It is not dead code.

Source imports:

- `table_retrieval/experiments/aggregation.py:6` — experiment/evaluation/metadata.
- `table_retrieval/experiments/tapas/model.py:5` — experiment/evaluation/metadata.
- `table_retrieval/experiments/tapas/pipeline.py:3` — experiment/evaluation/metadata.

Stale generated imports (reported, not edited):

- `build/lib/table_retrieval/stage2/bge/model.py:3`.

### `SCHEMA_VERSION`

Live output schema and Stage 2 CLI; retained. The current value 3.0 is emitted by EvidencePipeline.search() and the standalone Stage 2 CLI.

Source imports:

- `table_retrieval/pipeline.py:5` — live/shared runtime.
- `table_retrieval/stage2/cli.py:7` — live CLI.

Stale generated imports (reported, not edited):

- `build/lib/table_retrieval/pipeline.py:4`.
- `build/lib/table_retrieval/stage2/cli.py:6`.

### `Candidate`

Live/shared Stage 1-to-Stage 2 interface, experiments, CLI, tests; retained. Stage 1 constructs these objects; the Stage 2 CLI deserializes them; Stage 2 consumes the same class. Some consumers receive Candidate instances without directly importing the class.

Source imports:

- `experiment1/run_experiment.py:27` — historical experiment/training.
- `experiment2/evaluate.py:15` — historical experiment/training.
- `experiment3/run.py:18` — historical experiment/training.
- `experiment5/run.py:20` — historical experiment/training.
- `table_retrieval/experiments/bge/evaluate_aggregation.py:13` — experiment/evaluation/metadata.
- `table_retrieval/experiments/bge/evaluate_seed_diversity.py:9` — experiment/evaluation/metadata.
- `table_retrieval/experiments/bge/tests/test_bge.py:6` — tests.
- `table_retrieval/experiments/demo/app.py:24` — demo runtime.
- `table_retrieval/experiments/experiment6/tests/test_experiment6.py:9` — tests.
- `table_retrieval/experiments/experiment6/tests/test_extension.py:8` — tests.
- `table_retrieval/experiments/tapas/tests/test_localisation.py:9` — tests.
- `table_retrieval/experiments/tests/test_aggregation.py:7` — tests.
- `table_retrieval/experiments/tests/test_pipeline_legacy.py:10` — tests.
- `table_retrieval/stage1/pipeline.py:5` — live/shared runtime.
- `table_retrieval/stage2/cli.py:7` — live CLI.
- `tests/retrieval/test_bge.py:13` — tests.
- `tests/retrieval/test_stage2.py:11` — tests.
- `tests/retrieval/test_stage2_cli.py:9` — tests.

Stale generated imports (reported, not edited):

- `build/lib/table_retrieval/stage1/pipeline.py:11`.
- `build/lib/table_retrieval/stage2/cli.py:6`.

### `write_predictions`

Stage 2 CLI runtime; retained. Publishes JSONL without overwriting existing output or partial files and counts non-complete results.

Source imports:

- `table_retrieval/stage2/cli.py:7` — live CLI.

Stale generated imports (reported, not edited):

- `build/lib/table_retrieval/pipeline.py:61`.
- `build/lib/table_retrieval/stage2/cli.py:6`.

## Refactor boundaries

- The four moved function bodies and signatures are unchanged; there is one source implementation of each, with no compatibility re-export from runtime `data.py`.
- Shared normalization, corpus validation, JSON helpers, coordinate text, schema version, and Candidate behavior are unchanged.
- The evaluation code-provenance file list now includes `evaluation/data.py`, so the relocated implementation remains covered by source hashes.
- Scripts under `results/` were inspected without executing them; some perform writes or depend on already-retired APIs. Only the affected helper import is updated.
- Stale generated `build/lib` files are untouched and do not determine source architecture. Saved results, models, datasets, and manifests are untouched.
- In particular, `build/lib/table_retrieval/data.py` still contains its old helper definitions and schema version `2.0`; its generated pipeline/evaluation callers retain old imports. The current source schema remains `3.0`.
- No currently listed symbol was deleted as unused. `DATA`, `COORDINATES`, and `SCHEMA_VERSION` are intentionally retained.

## Files changed

| File | Change |
| --- | --- |
| `table_retrieval/data.py` | Remove the four relocated definitions and their unused `defaultdict` import; retain all shared definitions and constants. Clarify the legacy DATA path and shared normalization comments. |
| `table_retrieval/evaluation/data.py` | New home for the four unchanged evaluation helpers. |
| `table_retrieval/evaluation/fusion.py` | Update query/run helper imports. |
| `table_retrieval/evaluation/indexing.py` | Update query helper import; include the new module in the existing source-hash provenance list. |
| `table_retrieval/evaluation/metrics.py` | Update query/run/qrels helper imports. |
| `table_retrieval/evaluation/workflow.py` | Update run writer import. |
| `table_retrieval/evaluation/retrievers/bge.py` | Update query helper import. |
| `table_retrieval/evaluation/retrievers/bm25.py` | Update query helper import. |
| `table_retrieval/evaluation/tests/test_cli.py` | Update the run writer call/import. |
| `table_retrieval/evaluation/tests/test_data.py` | New format, validation, and local offline-workflow regression tests. |
| `table_retrieval/experiments/stage1/audit_test_qrels.py` | Update query/run helper imports. |
| `results/test_qrels_comparison/evaluate_all.py` | Update the run reader import only. |
| `docs/table-retrieval-data-audit.md` | Record pre-edit usage evidence, decisions, and validation. |

## Remaining shared data.py contents

`DATA`, `load_json`, `check_id`, `normalize_table_id`, `digest`, `write_json`,
`load_corpus`, `COORDINATES`, `SCHEMA_VERSION`, `Candidate`, `write_predictions`.

## Validation

- 182 tests passed across retrieval, evaluation, BGE experiments, Experiment 6,
  aggregation, demo comparison, and TAPAS tests.
- The new offline-workflow test uses a tiny local checkpoint without downloads.
  It builds BM25/BGE indexes, evaluates validation/test splits, confirms saved
  rankings are reused byte-for-byte without modification, reuses corpus embeddings,
  and checks that source-hash provenance includes the relocated helper module.
- AST comparison against the pre-edit snapshot confirms all 12 original
  function/class implementations are structurally identical, with one source
  definition each. This covers both the four moved helpers and the retained code.
- Parsed all 175 Python files after the change. No non-generated caller imports
  or accesses a relocated helper through shared `table_retrieval.data`.
- Updated evaluation/audit modules import successfully. Historical result-writing
  scripts were not executed.
- Fresh-process imports of live search and the Stage 2 CLI load no evaluation modules.
- Generated build copies, saved results, datasets, checkpoints, and existing
  manifests were not modified.
