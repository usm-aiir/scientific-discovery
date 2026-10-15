# Historical table-retrieval notes

This document preserves implementation and regression notes from earlier
iterations. The stable architecture and run command are documented in
`table_retrieval/README.md`; aggregation and RAG are not part of that path.

The stable retrieval path is:

```text
Query → BM25 + BGE-M3 → reciprocal rank fusion → Top-5 tables
      → Structural BGE → global Top-5 evidence cells
```

Stage 1 retrieves tables. Stage 2 ranks **all nonempty cells from those tables
together**, including header cells; it does not take five cells per table.
Neither stage uses gold labels, rerankers, fact-aware selection, or training at
inference. Existing aggregation is a separate downstream hook, described below,
and was not changed by this refactor.

## Run

Use an existing completed Stage 1 index and an explicit local fine-tuned BGE
checkpoint. The tested structural representation uses the same weights as the
table-aware checkpoint tested in Experiment 2; it is not a new model.
No checkpoint or index is created, selected automatically, or overwritten.

```python
from table_retrieval.pipeline import EvidencePipeline

pipeline = EvidencePipeline.from_settings(
    stage1_index="/path/to/completed/stage1/index",
    checkpoint="/path/to/stable/stage2/checkpoint",
    table_source_dir="/path/to/HTMLs",  # optional; <paper_id>.html files
)
result = pipeline.search("What accuracy was reported?", table_top_k=5,
                         cell_top_k=5, aggregation_enabled=False)
# result["tables"]: Top-5 tables; result["cells"]: global Top-5 cells
```

Initialize once and reuse the pipeline. Omit `table_source_dir` when source HTML
is unavailable; conservative corpus structure and available title/caption are
still used. To reproduce source-recovered structural results, supply the same
HTML files. Missing/ambiguous structure is not filled in by guessing.

From the repository root, with the project's environment activated:

```bash
python -m table_retrieval.pipeline \
  --query "What accuracy was reported?" \
  --stage1-index /path/to/completed/stage1/index \
  --checkpoint /path/to/stable/stage2/checkpoint \
  --table-source-dir /path/to/HTMLs \
  --output /tmp/table-evidence.jsonl
```

The combined CLI preserves the existing aggregation output alongside unchanged
seed metadata. The Python example above disables that hook for raw retrieval.
Use a fresh output filename; existing and partial outputs are protected.
Devices default to `auto`; use `--stage1-device cpu --cell-device cpu` if needed.
`--table-top-k` and `--cell-top-k` both default to **5**.

Stage 2 can also receive candidates directly, bypassing Stage 1:

```bash
python -m table_retrieval.stage2.cli \
  --query "What accuracy was reported?" \
  --candidates /path/to/candidates.json \
  --config table_retrieval/config.json \
  --output /tmp/cell-evidence.jsonl
```

Stage 2 reads `stage2_checkpoint`, `device`, `table_source_dir`, `cell_top_k`,
and `stage2_max_length` from the selected runtime config. Relative artifact
paths are resolved beside that config file.

`candidates.json` is a list of `Candidate.to_dict()` records:
`table_id`, `table`, optional `retrieval_rank` and `retrieval_score`.
No gold cell information is needed or read. This command ranks cells only.

## Code map

```text
table_retrieval/
├── data.py                  # Candidate interface and shared JSON/TREC I/O
├── pipeline.py              # connect the two stages; existing aggregation hook
├── stage1/
│   ├── pipeline.py          # loaded RetrievalPipeline → ordered Candidates
│   ├── retrieval.py         # BM25/BGE search → fusion → Top-5
│   ├── artifacts.py         # resolve/validate existing index and corpus
│   ├── text.py              # frozen table text and tokenizer budget
│   ├── fusion.py            # equal-weight reciprocal rank fusion
│   ├── retrievers/
│   │   ├── bm25.py          # lexical index/search; matched BM25 for fusion
│   │   ├── bge.py           # normalized CLS embedding index/search
│   │   └── ranking.py       # stable exact dense ranking
│   ├── indexing.py          # offline preparation and frozen index reuse
│   ├── workflow.py          # offline saved-run retrieval/evaluation
│   ├── evaluation.py        # TREC validation and table Recall@10
│   └── cli.py               # existing offline CLI, not the live query command
└── stage2/
    ├── pipeline.py          # build/pool candidate cells → global Top-5
    ├── cells.py             # original coordinates and legacy local metadata
    ├── structure.py         # trustworthy header/row-condition recovery
    ├── source.py            # local HTML spans/text; bounded read-only cache
    ├── representation.py    # exact structural scoring text
    ├── ranking.py           # scores → stable global cell order
    ├── bge/
    │   ├── model.py         # unchanged checkpoint loading and CLS encoding
    │   └── pipeline.py      # construct configured cell pipeline
    └── cli.py               # supplied Candidate list → ranked cells

tests/retrieval/
├── __init__.py              # discoverable test package (also tests/__init__.py)
├── test_stage1.py           # fusion, tie-breaking, fields, Candidate defaults
├── test_stage2.py           # exact text, spans, fallback, identity, global Top-5
└── test_bge.py              # tiny checkpoint replay and 512-token truncation
```

Data directories, weights, saved results, TAPAS, and numbered experiments remain
under the local `table_retrieval/experiments/` tree.
The previous `table_retrieval/tests/` suite also remains available locally.
New refactor tests live outside that gitignored directory so they can be reviewed
and committed with the source. Historical imports from `stage1.pipeline` and
`stage2.bge.model.make_cells` still work; new live code should use the focused
modules above.

## Behavior preserved

Stage 1 uses the **matched** BM25 index and BGE-M3 at the saved candidate depth
(currently 100). Fusion is equal-weight RRF, `1 / (60 + rank)`, or the constant
recorded in the index. RRF ties use table ID; dense ties use saved corpus order.
Table input fields, tokenization, embeddings and saved index manifests are
unchanged. Offline Recall@10 runs keep their saved Top-10 output policy; that is
independent of the live Top-5 default.

Stage 2 retains original table IDs and zero-based `rows[row][column]`
coordinates, including headers. Equal cell scores keep Stage 1 candidate order,
then original row/column order. Scores remain normalized-CLS cosine similarities.
The unchanged encoder uses batches of 32, a **512-token** limit including special
tokens, and right truncation as configured by the checkpoint tokenizer.
GPU inference retains FP16; CPU retains FP32. Historical experimental FP32 GPU
scores are not asserted to be bit-identical to FP16 live scores.

The intentional behavior change is Stage 2's **scoring text**, now the tested
structural representation rather than legacy row/header/value text:

```text
<row label> | <header > path> | <target value>
 | Row context: <named condition>=<value>; ...
 | Table: <caption> | Subtable: <sub-caption> | Paper: <paper title>
```

This is **one line** joined with ` | `, not a multiline prompt. Optional segments
are omitted when absent; local semantics and value always come first. Query text
is encoded separately, with whitespace stripped and no extra instruction.
No abstract, reference prose, neighboring numeric measurements, labels, or
experiment data are added. Header spans/row conditions require exact, unique alignment
of all corpus cells (including blanks) to a contiguous source-HTML row block.
Conservative leaf headers/named indicators are retained when HTML is unavailable;
unrecoverable headers keep the existing `Column N` fallback.

Cell `row_label`, `header_path`, and `header_source` in the public result remain
the original metadata used by existing aggregation. Recovered structure enriches
BGE's scoring text, not the aggregator's inputs. Per-table `structure` audit and
configuration `cell_representation="structural_v1"` make that distinction visible.
Malformed candidates produce explicit partial/error results without discarding
valid tables; unreadable optional HTML produces warnings and corpus fallback.
Logging uses Python's standard logging module; applications may configure its
level (for example `logging.basicConfig(level=logging.DEBUG)`).

## Test

```bash
python -m unittest discover -s tests -v
```

These tests need installed project dependencies but no internet, GPU, real index,
large checkpoint, or training. The encoder test creates a tiny temporary random
checkpoint solely to exercise load/inference compatibility.

Additional existing local regressions:

```bash
python -m unittest discover -s table_retrieval/tests -v
python -m unittest discover -s table_retrieval/experiments/stage1/tests -v
```

## Refactor verification

116 targeted tests passed: 33 new focused tests, 48 existing runtime regressions, and
35 existing offline/legacy compatibility tests. CLI help, compilation, and
`git diff --check` also passed. No training or checkpoint writes were performed.
Broader repository discovery also passed (84 tests, including local experiment
tests). Verification used the existing local environment. The pre-existing
`transformers<5` dependency constraint was left unchanged. A clean-install check
under that declared range is still needed.

Fixed queries 33, 46 and 127 were replayed on the existing 63,021-table index
and unchanged table-aware BGE checkpoint, using the same GPU precision, batch
size and candidate order before/after:

| Query | Pooled cells | Stage 1 IDs/scores | Legacy cell rankings/scores | Structural rankings/scores | Inputs >512 tokens |
| --- | ---: | --- | --- | --- | ---: |
| 33 | 686 | identical | identical | identical to reference formatter | 261 |
| 46 | 257 | identical | identical | identical to reference formatter | 0 |
| 127 | 263 | identical | identical | identical to reference formatter | 0 |

Structural context was also checked against frozen validation/test contexts:
**894 unique tables, 47,586 cells**, exactly matching. All 11 protected
index/checkpoint artifacts kept the same SHA-256 hashes. The new structural
Top-5 intentionally differs from the old local-text Top-5; this is a compatibility
check, not a new quality evaluation or a claim of improved metrics.

## Evidence aggregation and RAG output

`EvidencePipeline.search()` runs the pure `aggregation.aggregate_evidence()`
function after Stage 2. Schema 2.1 adds these fields without changing `cells`:

- `evidence_groups`: seed references, explicit pairwise relationship reasons,
  context references, and `evidence_rank`. Groups are ordered by the maximum
  member BGE score (`ordering_score`), then original rank; no score bonuses.
- `context_cells`: source-backed cells with coordinates and `expanded_from`
  seed references/reasons. `stage2_bge_score` and `global_rank` are null because
  scores for these cells are not available in Stage 2's public result.
- `aggregation`: policy version, status, coordinate convention, budgets,
  warnings, and allowlisted source metadata keyed by corpus table ID in `sources`.

References always contain both `table_id` and `cell_id`. Original `global_rank`
continues to mean Stage 2 rank. Downstream RAG can consume groups, resolve their
members against `cells`, and attach `context_cells` and source metadata for
citations. Context is supporting material, not additional ranked predictions.

Same-table seeds group only when they share a row. Cross-table grouping requires
informative matching row labels and full header paths (including qualifiers),
plus at least two shared informative caption terms. Fallback headers never
establish cross-table relationships. Every pair in a group must qualify; a
transitive chain cannot merge unrelated endpoints. These are possible
relationships, not verified equivalence or agreement; conflicting values remain.

Policy `query_structural_v2` limits expansion to three cells per seed, eight
unique cells overall, and 4,000 characters of added value/label/header text.
`aggregate_evidence` accepts `max_context_per_seed`, `max_context_cells`, and
`max_context_chars`; zero disables additions under that budget. These limits
apply to context, not original seeds, metadata, or serialized JSON overhead.
No model tokenizer is needed for the character cap.

Query-matched header-to-value and comparator-row traversal take priority over
same-row companions and headers. Header traversal requires known first-row
alignment; comparator traversal uses a matching metric column and query-matched
row labels. Ordinary row/header expansion requires a query-matched seed, and
same-row companions must match the query or supply the row stub. Proposals are
selected globally, with Stage 1 rank providing deterministic source ordering
within a traversal tier; no model scores are modified.

Cross-table links include `interchangeable: false`, a condition warning, and
per-source captions, sub-captions, header paths, and explicit dataset/conditions/
units metadata when present (null otherwise). Captions preserve contextual
differences without guessing missing metadata or asserting equivalence.

The standalone Stage 2 CLI remains a cell-ranking command; aggregation is part
of the combined pipeline only. Empty/error combined results include empty
aggregation fields. Context warnings have their own aggregation status and do
not replace retrieval errors. The aggregator makes no model calls and reads no
labels or judgments. Existing Top-3 evaluation metrics do not measure this new
configuration; no quality improvement is claimed without a new evaluation.
