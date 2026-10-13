#!/usr/bin/env bash
# Runs the base and fine-tuned models in both bf16 and 4-bit with the SAME
# retrieval script, test set and qrels, then prints nDCG@10 for each.
#
#   base / bf16      base / 4-bit
#   tuned / bf16     tuned / 4-bit
#
# Defaults match Adah's layout: the scripts live in the repo
# (~/scientific-discovery) and all data, adapters and results live on the
# netstore (~/netstore1/finetuning). Run from the repo:
#   ./run_2_2.sh
# Run only some of the four runs (any of: base_bf16 finetuned_bf16 base_4bit finetuned_4bit):
#   RUNS="base_bf16 base_4bit" ./run_2_2.sh
# Override any setting via environment variables, e.g. to test the other adapter:
#   ADAPTER=colqwen-qlora-run/best_adapter ./run_2_2.sh
#
# Note: each run scans the full image corpus, so this is four full scans.

set -euo pipefail

# The Python scripts sit next to this file (in the repo); the data, adapters
# and outputs are on the netstore. Relative paths below are relative to DATA_DIR.
SCRIPT_DIR="$(dirname "$(readlink -f "$0")")"
# Netstore location differs between machines: ~/netstore1 on some (e.g. ant),
# /mnt/netstore1_home/<user> on others (e.g. turing). Use whichever exists.
if [ -z "${NETSTORE:-}" ]; then
  for cand in "$HOME/netstore1" "/mnt/netstore1_home/$USER"; do
    if [ -d "$cand/finetuning" ]; then NETSTORE="$cand"; break; fi
  done
fi
[ -n "${NETSTORE:-}" ] || { echo "[ERROR] Could not find the netstore (tried ~/netstore1 and /mnt/netstore1_home/$USER). Set NETSTORE=/path/to/it."; exit 1; }

DATA_DIR="${DATA_DIR:-$NETSTORE/finetuning}"
[ -d "$DATA_DIR" ] || { echo "[ERROR] Data folder not found: $DATA_DIR"; exit 1; }
cd "$DATA_DIR"

QUERY_JSON="${QUERY_JSON:-Test.json}"
QRELS="${QRELS:-Test_figure_qrels.tsv}"
IMAGE_DIR="${IMAGE_DIR:-$NETSTORE/figures/images}"
ADAPTER="${ADAPTER:-colqwen_qlora_run/best_adapter}"
OUT_DIR="${OUT_DIR:-runs_2x2}"
RUNS="${RUNS:-base_bf16 finetuned_bf16 base_4bit finetuned_4bit}"

# Reuse the model weights already downloaded to the netstore.
export HF_HOME="${HF_HOME:-$NETSTORE/HF}"
# Run on GPU 1 so GPU 0 stays free for training.
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-1}"
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"

# Fail early with a clear message if something is missing.
for f in "$QUERY_JSON" "$QRELS" "$SCRIPT_DIR/colqwen_retrieval_finetuned.py" "$SCRIPT_DIR/eval_utils.py"; do
  [ -e "$f" ] || { echo "[ERROR] Missing $f in $(pwd)"; exit 1; }
done
[ -d "$IMAGE_DIR" ] || { echo "[ERROR] Image folder not found: $IMAGE_DIR"; exit 1; }
if [[ " $RUNS " == *" finetuned_"* ]]; then
  [ -d "$ADAPTER" ] || { echo "[ERROR] Adapter folder not found: $ADAPTER"; exit 1; }
fi

echo "Scripts     : $SCRIPT_DIR"
echo "Data dir    : $DATA_DIR"
echo "Queries     : $QUERY_JSON"
echo "Qrels       : $QRELS"
echo "Images      : $IMAGE_DIR"
echo "Adapter     : $ADAPTER"
echo "Runs        : $RUNS"
echo "GPU         : CUDA_VISIBLE_DEVICES=$CUDA_VISIBLE_DEVICES"
echo "HF cache    : $HF_HOME"
echo

mkdir -p "$OUT_DIR"

for run in $RUNS; do
  case "$run" in
    base_bf16)      FLAGS=() ;;
    base_4bit)      FLAGS=(--quantize_inference) ;;
    finetuned_bf16) FLAGS=(--lora_adapter "$ADAPTER") ;;
    finetuned_4bit) FLAGS=(--lora_adapter "$ADAPTER" --quantize_inference) ;;
    *) echo "[ERROR] Unknown run '$run' (use base_bf16, finetuned_bf16, base_4bit, finetuned_4bit)"; exit 1 ;;
  esac
  echo "=== $run ==="
  # Re-running resumes automatically from this run's checkpoint, if one exists.
  python "$SCRIPT_DIR/colqwen_retrieval_finetuned.py" \
    --query_json "$QUERY_JSON" --image_dir "$IMAGE_DIR" --qrels "$QRELS" \
    "${FLAGS[@]}" \
    --output "$OUT_DIR/${run}.trec" \
    2>&1 | tee -a "$OUT_DIR/${run}.log"
done

echo
echo "===== nDCG@10 summary ($OUT_DIR) ====="
for run in $RUNS; do
  score=$(grep "ndcg@10:" "$OUT_DIR/$run.log" 2>/dev/null | tail -1 | awk '{print $2}' || true)
  echo "$run: ${score:-n/a}"
done
echo
echo "Also check each log for '[COVERAGE]', '[ADAPTERS]' and the 'Scan summary' section."