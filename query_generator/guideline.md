# Getting Started: Running `generate_figure_queries.py`

## Overview

This guide explains how to:

1. Install Miniconda (you can skip this already install on cs servers)
2. Create a Conda environment
3. Install required libraries
4. Authenticate with Hugging Face
5. Verify GPU access
6. Run the script successfully
7. Export and reproduce environments

---

## 1. Install Miniconda

Download and install Miniconda:

https://docs.conda.io/en/latest/miniconda.html

Verify installation:

```bash
conda --version
```

---

## 2. Create a New Environment

Create a dedicated environment:

```bash
conda create -n figure-query python=3.11
```

Activate it:

```bash
conda activate figure-query
```

Verify Python:

```bash
python --version
```

---

## 3. Install Required Libraries

Inspect the imports in the script and determine the required packages.

Core dependencies:

```bash
pip install pandas
pip install transformers
pip install torch
pip install accelerate
pip install sentencepiece
pip install bitsandbytes
```

Optional:

```bash
pip install torchvision torchaudio
```

Verify installation:

```bash
python -c "import pandas, transformers, torch; print('All imports successful')"
```

---

## 4. Hugging Face Authentication

If the model is gated, login first:

```bash
huggingface-cli login
```

Paste your Hugging Face access token.

Test access:

```bash
python -c "from transformers import AutoTokenizer; AutoTokenizer.from_pretrained('google/gemma-4-31b-it')"
```

---

## 5. Verify CUDA Availability

Check GPU visibility:

```bash
python -c "import torch; print(torch.cuda.is_available())"
```

Expected output:

```text
True
```

Check available GPUs:

```bash
nvidia-smi
```

---

## 6. Running the Script

Example:

```bash
python generate_figure_queries.py \
  --captions_tsv captions.tsv \
  --metadata_tsv metadata.tsv \
  --ref_tsv references.tsv \
  --figures_dir figures \
  --output_tsv queries.tsv \
  --target 200 \
  --model_name google/gemma-4-31b-it \
```

---

## 7. Understanding the Script

The script:

1. Loads figure captions, metadata, and references.
2. Resolves figure image paths.
3. Selects a diverse set of candidate figures.
4. Uses a Gemma model as an "Author" agent.
5. Uses a second Gemma-based "Reviewer" agent.
6. Iteratively improves generated queries.
7. Saves accepted queries to a TSV file.

---

# Capturing Your Current Environment (for further reading)

## Python Version

```bash
python --version
```

or

```bash
python -c "import sys; print(sys.version)"
```

---

## Conda Environment Information

```bash
conda info
```

List environments:

```bash
conda env list
```

---

## Installed Conda Packages

```bash
conda list
```

Save to file:

```bash
conda list > conda_packages.txt
```

---

## Installed Pip Packages

```bash
pip freeze
```

Save to file:

```bash
pip freeze > requirements.txt
```

---

## Export Entire Conda Environment

Recommended for reproducibility:

```bash
conda env export > environment.yml
```

A student can recreate the environment using:

```bash
conda env create -f environment.yml
```

Then activate:

```bash
conda activate <environment-name>
```

---

# Suggested Learning Exercise

Before receiving the full environment file, try the following:

1. Create a new Conda environment.
2. Read the script imports.
3. Identify required packages.
4. Install packages one by one.
5. Resolve import errors.
6. Authenticate with Hugging Face.
7. Load a small model successfully.
8. Run the full pipeline.

This exercise develops practical skills in:
- Conda environment management
- Python package installation
- Hugging Face ecosystem
- GPU verification
- AI model deployment
- Debugging and dependency management

---