#!/usr/bin/env bash
# Build the table index and evaluate validation and test questions.
set -euo pipefail

script_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
experiment_dir="$(cd -- "$script_dir/.." && pwd)"
repo_root="$(cd -- "$experiment_dir/../.." && pwd)"
python="$experiment_dir/.venv/bin/python"
if [[ ! -x "$python" ]]; then
    echo "Environment missing. Run bash $script_dir/setup.sh first." >&2
    exit 1
fi
cd "$repo_root"

results_dir="${EXPERIMENT_RESULTS_DIR:-$repo_root/results}"
index="$results_dir/stage1/index"
options=(--device "${EXPERIMENT_DEVICE:-auto}")
if [[ -n "${EXPERIMENT_DATA_DIR:-}" ]]; then
    options+=(--data-dir "$EXPERIMENT_DATA_DIR")
fi

if [[ -f "$index/index.json" ]]; then
    "$python" - "$index" "${EXPERIMENT_DATA_DIR:-}" <<'PY'
from pathlib import Path
import sys
from table_retrieval.data import DEFAULT_DATA_DIR, digest, load_json

index = Path(sys.argv[1])
data_dir = Path(sys.argv[2]) if sys.argv[2] else DEFAULT_DATA_DIR
if load_json(index / 'prepared.json')['corpus_sha256'] != digest(data_dir / 'Corpus.json'):
    raise SystemExit('The saved index uses another corpus. Choose a new EXPERIMENT_RESULTS_DIR.')
PY
    echo "Using completed Stage 1 index: $index"
else
    "$python" -m table_retrieval.experiments.stage1.build_index --output "$index" "${options[@]}"
fi
for split in val test; do
    "$python" -m table_retrieval.experiments.stage1.evaluate \
        --index "$index" --split "$split" --output "$results_dir/stage1/$split" "${options[@]}"
done

echo "Stage 1 complete: $results_dir/stage1"
