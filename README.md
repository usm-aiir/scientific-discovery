# Scientific Discovery

A multimodal scientific discovery system that integrates figure search, table search, and text search into a single agentic RAG pipeline. Given a natural-language query, an LLM-driven controller routes to the appropriate modality, formulates sub-queries, invokes the retrieval systems, and combines the retrieved evidence into a grounded answer with explicit citations back to the source figures, tables, and text.

## Project Context

This repository contains the full system across all modalities. Currently implemented: **figure and table retrieval** — scraping arXiv content, sampling representative figures and tables, generating retrieval queries, and evaluating baseline retrieval performance. Text retrieval and the agentic controller are under active development.

## Repository Structure

scrapers/ Tools for collecting figures, tables, captions, and metadata
from ar5ive (the web-rendered version of arXiv papers)

query_generator/ Utilities for generating retrieval queries from collected figures,
using a local Gemma model with a two-agent Author/Reviewer loop

bin/ Shell scripts for running the full pipeline
evaluate_clip_retrieval.py Evaluates CLIP text-to-image retrieval against ground-truth queries


## Setup

With Conda installed, clone the repository and create the shared table-retrieval
environment for live retrieval, Stage 1 indexing/evaluation and Stage 2 BGE
training/evaluation:

```bash
git clone <repository-url>
cd scientific-discovery
bash setup.sh
conda activate table-retrieval
```

All Python dependencies are pinned in `requirements.txt`. `pyproject.toml`
contains package/build metadata only; setup installs the package in editable mode.
Setup leaves an existing `table-retrieval` environment untouched. Model weights,
indexes, corpus and source HTML are external assets, not installed dependencies;
see [table retrieval](table_retrieval/README.md) for the required paths.

## Usage

### Scrape arXiv figures and metadata

Scrapes figures, captions, references, and metadata for a given month in a single pass:

```bash
bin/run_figure_scrape <year> <month> [--max-papers N] [--start-id N]
```

```bash
bin/run_figure_scrape 24 10               # scrape all of October 2024
bin/run_figure_scrape 24 10 --max-papers 5  # first 5 papers only (for testing)
bin/run_figure_scrape 24 10 --start-id 200  # resume from paper 00200
```

Output:

arxiv_data/figures/<yy>/<mm>/<paper_id>/*.png
arxiv_data/captions/<year><month>.tsv
arxiv_data/references/ref<year><month>.tsv
arxiv_data/figure_metadata<year>_<month>.tsv


### Scrape arXiv tables

Scrapes tables, captions, cell grids, footnotes, and in-text references:

```bash
bin/run_table_scrape <year> <month> [--max-papers N] [--start-id N]
```

```bash
bin/run_table_scrape 24 10
bin/run_table_scrape 24 10 --max-papers 5
```

Output:

arxiv_data/tables/<year><month>.jsonl
arxiv_data/table_metadata<year>_<month>.tsv


### Sample figures

Draws a reproducible stratified random sample of 200 figures across arXiv categories:

```bash
bin/run_figure_sample <year> <month> [--data_dir DIR] [--output_tsv FILE]
```

```bash
bin/run_figure_sample 24 10
bin/run_figure_sample 24 10 --output_tsv results/sampled_24_10.tsv
```

Output: `<year><month>_sampled_figures.tsv` (default)

### Sample tables

Draws a reproducible stratified random sample of 200 tables, merging table and metadata outputs:

```bash
bin/run_table_sample <year> <month> [--data_dir DIR] [--output_tsv FILE]
```

```bash
bin/run_table_sample 24 10
bin/run_table_sample 24 10 --output_tsv results/sampled_24_10.tsv
```

Output: `<year><month>_sampled_tables.tsv` (default)

### Generate figure queries

Generates a retrieval query per figure using a local Gemma model. Uses a two-agent loop: an **Author** agent drafts a query grounded in the figure image and context; a **Reviewer** agent accepts it or returns feedback for revision. Runs up to 3 rounds per figure, writing accepted queries incrementally so the run is safely resumable.

```bash
bin/run_query_gen <captions_tsv> <metadata_tsv> <ref_tsv> <figures_dir> \
    [output_tsv] [--load_in_4bit] [--model_name NAME]
```

```bash
bin/run_query_gen \
  arxiv_data/captions/25_01.tsv \
  arxiv_data/figure_metadata_25_01.tsv \
  arxiv_data/references/ref_25_01.tsv \
  arxiv_data/figures \
  queries_25_01.tsv \
  --model_name google/gemma-3-4b-it
```

Output: `queries_output.tsv` (default) — one accepted query per figure, with `paper_id`, `figure_id`, `sub_id`, and `query` columns. A timestamped log file is written alongside the output for long-running jobs.

### Evaluate CLIP retrieval (baseline)

Evaluates CLIP text-to-image retrieval against ground-truth queries, reporting Recall@1/5/10 and MRR:

```bash
python evaluate_clip_retrieval.py \
    --queries_tsv claude_figure_queries.tsv \
    --cache /path/to/clip_embeddings.pkl
```

**Baseline results (October 2024, 187,851 figures, openai/clip-vit-large-patch14):**

| Metric     | Score  |
|------------|--------|
| Recall@1   | 0.0%   |
| Recall@5   | 0.3%   |
| Recall@10  | 0.5%   |
| MRR        | 0.0023 |

Vanilla CLIP performs near-randomly on scientific figures, establishing the baseline this project aims to improve on.

## Environments

Table retrieval and both table experiments use `table-retrieval`, created by
`bash setup.sh`. There is no `environment.yml` and no additional experiment
requirements file.

**Legacy tools:** the existing scraping, sampling and query-generation launchers
still target `scidiscovery`. The deprecated installer retains this compatibility
route, using the same root `requirements.txt`:

```bash
bash bin/install --tools
```

The historical `llm2vecenv` CLIP environment on the HPC cluster is unrelated to
table retrieval and is not created by `setup.sh`.

## Purpose

To provide the figure and table retrieval foundation for a multimodal scientific discovery system — a RAG pipeline that routes natural-language queries across text, table, and image modalities and returns grounded, attributable answers.
ENDOFFILE

