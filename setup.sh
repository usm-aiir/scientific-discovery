#!/usr/bin/env bash
# One environment for live table retrieval and both experiments.
set -euo pipefail
cd "$(dirname "$0")"
command -v conda >/dev/null || { echo 'Install Conda first, then run bash setup.sh.' >&2; exit 1; }
retrieval_envs="$(conda env list)"
if awk '$1 == "table-retrieval" {found=1} END {exit !found}' <<< "$retrieval_envs"; then
    echo 'table-retrieval already exists; leaving it untouched. Activate it with conda activate table-retrieval.' >&2
    exit 0
fi
conda create --yes --name table-retrieval --override-channels --channel conda-forge python=3.10.12 pip
conda run --no-capture-output -n table-retrieval python -m pip install -r requirements.txt
conda run --no-capture-output -n table-retrieval python -m pip install --no-deps --no-build-isolation -e .
echo 'Setup complete. Run: conda activate table-retrieval'
