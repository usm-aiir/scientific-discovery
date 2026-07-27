# Scientific Discovery

This repository contains tools and resources for supporting scientific discovery workflows, including scientific document collection and query generation.

## Repository Structure

### `ar5ive_scraper/`
Tools for collecting scientific content from ar5ive, the web-rendered version of arXiv papers. The scraper extracts paper content and metadata that can be used for downstream analysis, retrieval, and question generation tasks.

### `query_generator/`
Utilities for generating scientific queries from collected documents. These queries can be used for information retrieval, question answering, benchmark creation, and evaluation of scientific discovery systems.

### `bin/`
Shell scripts for installing dependencies and running the scraper pipeline.

## Getting Started

Clone the repository:

```bash
git clone <repository-url>
cd <repository-name>
```
Install required dependencies (creates a conda environment and installs all libraries):

```bash
bin/install
```
## Usage

### Scrape arXiv figures and metadata
Scrapes figures, captions, and metadata for a given month:

```bash
bin/run_scrape <year> <month>
```
Example (October 2024):

```bash
bin/run_scrape 24 10
```

Output is written to `arxiv_data/figures/`, `arxiv_data/captions/`, `arxiv_data/references/`, and `arxiv_data/metadata_<year>_<month>.tsv`.

### Scrape arXiv tables
Scrapes tables, captions, cell grids, footnotes, and in-text references:

```bash
bin/run_table_scrape <year> <month>
```

Example:

```bash
bin/run_table_scrape 24 10
```

Output is written to `arxiv_data/tables/<year>_<month>.jsonl`.

### Sample tables
Builds a reproducible stratified random sample of 200 unique tables across categories, merging table and metadata outputs:

```bash
bin/run_sample [output_dir] [output_tsv]
```

Example:

```bash
bin/run_sample arxiv_data sampled_tables.tsv
```
Output is written to `sampled_tables.tsv` by default.

### Generate figure queries
Generates figure-grounded queries using a local Gemma 31B model:

```bash
bin/run_query_gen <captions_tsv> <metadata_tsv> <ref_tsv> <figures_dir> [output_tsv]
```

Example:

```bash
bin/run_query_gen arxiv_data/captions/24_10.tsv arxiv_data/metadata_24_10.tsv arxiv_data/references/ref_24_10.tsv arxiv_data/figures queries_output.tsv
```

Output is written to `queries_output.tsv` by default.

### Sample figures
Samples 200 figures across arXiv categories for a given month, outputting a TSV with empty query columns ready to be filled in:

```bash
bin/run_figure_sample <year> <month> [--data_dir] [--output_tsv]
```

Example:

```bash
bin/run_figure_sample 24 10
```

Output is written to `200_sampled_figures.tsv` by default.

Refer to the README or documentation within each subdirectory for setup instructions and usage examples.

## Purpose

The goal of this project is to provide foundational tools for building and evaluating systems that assist researchers in discovering, exploring, and understanding scientific knowledge.

