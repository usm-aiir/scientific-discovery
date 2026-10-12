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
#   ./run_2x2_comparison.sh
# Override any setting via environment variables, e.g. to test the other adapter:
#   ADAPTER=colqwen-qlora-run/best_adapter ./run_2x2_comparison.sh
#
# Note: each run scans the full image corpus, so this is four full scans.

set -euo pipefail

# The Python scripts sit next to this file (in the repo); the data, adapters
# and outputs are on the netstore. Relative paths below are relative to DATA_DIR.
SCRIPT_DIR="$(dirname "$(readlink -f "$0")")"
DATA_DIR="${DATA_DIR:-$HOME/netstore1/finetuning}"
[ -d "$DATA_DIR" ] || { echo "[ERROR] Data folder not found: $DATA_DIR"; exit 1; }
cd "$DATA_DIR"

QUERY_JSON="${QUERY_JSON:-Test.json}"
QRELS="${QRELS:-Test_figure_qrels.tsv}"
IMAGE_DIR="${IMAGE_DIR:-$HOME/netstore1/figures/images}"
ADAPTER="${ADAPTER:-colqwen_qlora_run/best_adapter}"
OUT_DIR="${OUT_DIR:-runs_2x2}"

# Reuse the model weights already downloaded to the netstore.
export HF_HOME="${HF_HOME:-$HOME/netstore1/HF}"
# Run on GPU 1 so GPU 0 stays free for training.
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-1}"
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"

# Fail early with a clear message if something is missing.
for f in "$QUERY_JSON" "$QRELS" "$SCRIPT_DIR/colqwen_retrieval_finetuned.py" "$SCRIPT_DIR/eval_utils.py"; do
  [ -e "$f" ] || { echo "[ERROR] Missing $f in $(pwd)"; exit 1; }
done
[ -d "$IMAGE_DIR" ] || { echo "[ERROR] Image folder not found: $IMAGE_DIR"; exit 1; }
[ -d "$ADAPTER" ]   || { echo "[ERROR] Adapter folder not found: $ADAPTER (try ADAPTER=colqwen-qlora-run/best_adapter)"; exit 1; }

echo "Scripts     : $SCRIPT_DIR"
echo "Data dir    : $DATA_DIR"
echo "Queries     : $QUERY_JSON"
echo "Qrels       : $QRELS"
echo "Images      : $IMAGE_DIR"
echo "Adapter     : $ADAPTER"
echo "GPU         : CUDA_VISIBLE_DEVICES=$CUDA_VISIBLE_DEVICES"
echo "HF cache    : $HF_HOME"
echo

mkdir -p "$OUT_DIR"

for prec in bf16 4bit; do
  QFLAG=()
  if [ "$prec" = "4bit" ]; then QFLAG=(--quantize_inference); fi

  echo "=== base / $prec ==="
  python "$SCRIPT_DIR/colqwen_retrieval_finetuned.py" \
    --query_json "$QUERY_JSON" --image_dir "$IMAGE_DIR" --qrels "$QRELS" \
    ${QFLAG[@]+"${QFLAG[@]}"} \
    --output "$OUT_DIR/base_${prec}.trec" \
    2>&1 | tee "$OUT_DIR/base_${prec}.log"

  echo "=== fine-tuned / $prec ==="
  python "$SCRIPT_DIR/colqwen_retrieval_finetuned.py" \
    --query_json "$QUERY_JSON" --image_dir "$IMAGE_DIR" --qrels "$QRELS" \
    --lora_adapter "$ADAPTER" \
    ${QFLAG[@]+"${QFLAG[@]}"} \
    --output "$OUT_DIR/finetuned_${prec}.trec" \
    2>&1 | tee "$OUT_DIR/finetuned_${prec}.log"
done

echo
echo "===== nDCG@10 summary ====="
for f in base_bf16 finetuned_bf16 base_4bit finetuned_4bit; do
  score=$(grep -m1 "ndcg@10:" "$OUT_DIR/$f.log" | awk '{print $2}')
  echo "$f: ${score:-n/a}"
done
echo
echo "Also check each log for '[COVERAGE]', '[ADAPTERS]' and the 'Scan summary' section."