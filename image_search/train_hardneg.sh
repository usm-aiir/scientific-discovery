#!/usr/bin/env bash
# QLoRA fine-tuning of ColQwen2.5 with same-paper hard negatives, using Adah's
# layout: scripts in the repo (~/scientific-discovery), data and outputs on the
# netstore (~/netstore1/finetuning). Run from the repo:
#   ./run_train_hardneg.sh
#
# Override settings via environment variables, e.g. if you hit out-of-memory:
#   HARD_NEG=2 BATCH=2 ./run_train_hardneg.sh
# Any extra arguments are passed straight to train_colqwen_qlora.py:
#   ./run_train_hardneg.sh --epochs 2

set -euo pipefail

SCRIPT_DIR="$(dirname "$(readlink -f "$0")")"
DATA_DIR="${DATA_DIR:-$HOME/netstore1/finetuning}"
[ -d "$DATA_DIR" ] || { echo "[ERROR] Data folder not found: $DATA_DIR"; exit 1; }
cd "$DATA_DIR"   # data files and OUTPUT_DIR are relative to here

IMAGE_DIR="${IMAGE_DIR:-$HOME/netstore1/figures/images}"
BATCH="${BATCH:-4}"
HARD_NEG="${HARD_NEG:-3}"
ACCUM="${ACCUM:-4}"
EVAL_EVERY="${EVAL_EVERY:-50}"
OUTPUT_DIR="${OUTPUT_DIR:-colqwen_hardneg_run}"

export HF_HOME="${HF_HOME:-$HOME/netstore1/HF}"
# Run on GPU 0 so GPU 1 stays free for the 2x2 retrieval runs.
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"

for f in Train.json Train_figure_qrels.tsv Val.json Val_figure_qrels.tsv "$SCRIPT_DIR/train_colqwen_qlora.py" "$SCRIPT_DIR/eval_utils.py"; do
  [ -e "$f" ] || { echo "[ERROR] Missing $f in $(pwd)"; exit 1; }
done
[ -d "$IMAGE_DIR" ] || { echo "[ERROR] Image folder not found: $IMAGE_DIR"; exit 1; }
if [ -e "$OUTPUT_DIR/best_adapter" ]; then
  echo "[ERROR] $OUTPUT_DIR/best_adapter already exists. Use OUTPUT_DIR=<new name> so you don't overwrite a previous run."
  exit 1
fi

mkdir -p "$OUTPUT_DIR"
echo "Scripts: $SCRIPT_DIR   Data: $DATA_DIR"
echo "Training on GPU $CUDA_VISIBLE_DEVICES: batch=$BATCH hard_negatives=$HARD_NEG accum=$ACCUM -> $OUTPUT_DIR"
echo "Each step encodes $((BATCH * (1 + HARD_NEG))) images. Watch nvidia-smi for the first few steps."
echo

python "$SCRIPT_DIR/train_colqwen_qlora.py" \
  --image_dir "$IMAGE_DIR" \
  --train_json Train.json --train_qrels Train_figure_qrels.tsv \
  --val_json   Val.json   --val_qrels   Val_figure_qrels.tsv \
  --train_batch_size "$BATCH" --hard_negatives "$HARD_NEG" --grad_accum_steps "$ACCUM" \
  --gradient_checkpointing --eval_every_steps "$EVAL_EVERY" \
  --output_dir "$OUTPUT_DIR" \
  "$@" \
  2>&1 | tee "$OUTPUT_DIR/train.log"