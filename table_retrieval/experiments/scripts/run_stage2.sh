#!/usr/bin/env bash
# Train the cell retriever, then evaluate pretrained and selected checkpoints.
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
output="$results_dir/stage2_bge"
options=(--device "${EXPERIMENT_DEVICE:-auto}")
if [[ -n "${EXPERIMENT_DATA_DIR:-}" ]]; then
    options+=(--data-dir "$EXPERIMENT_DATA_DIR")
fi

for split in val test; do
    for filename in fusion.run experiment.json; do
        if [[ ! -f "$results_dir/stage1/$split/$filename" ]]; then
            echo "Stage 1 input missing: $results_dir/stage1/$split/$filename" >&2
            echo "Run bash $script_dir/run_stage1.sh first." >&2
            exit 1
        fi
    done
done
for folder in train val_pretrained val_trained test_pretrained test_trained; do
    if [[ -e "$output/$folder" ]]; then
        echo "Output already exists: $output/$folder. Choose a new EXPERIMENT_RESULTS_DIR." >&2
        exit 1
    fi
done

"$python" -m table_retrieval.experiments.stage2_bge.train \
    --val-stage1-run "$results_dir/stage1/val/fusion.run" --output "$output/train" "${options[@]}"
for split in val test; do
    "$python" -m table_retrieval.experiments.stage2_bge.evaluate \
        --stage1-run "$results_dir/stage1/$split/fusion.run" --split "$split" \
        --pretrained --output "$output/${split}_pretrained" "${options[@]}"
    "$python" -m table_retrieval.experiments.stage2_bge.evaluate \
        --stage1-run "$results_dir/stage1/$split/fusion.run" --split "$split" \
        --checkpoint "$output/train" --output "$output/${split}_trained" "${options[@]}"
done

echo "Stage 2 complete: $output"
