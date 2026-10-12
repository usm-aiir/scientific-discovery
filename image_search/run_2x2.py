#!/usr/bin/env bash
# Runs the base and fine-tuned models in both bf16 and 4-bit with the SAME
# retrieval script, test set and qrels, then prints nDCG@10 for each.
#
#   base / bf16      base / 4-bit
#   tuned / bf16     tuned / 4-bit
#
# Override any path via environment variables, e.g.:
#   QUERY_JSON=Test.json QRELS=Test_figure_qrels.tsv ./run_2x2_comparison.sh
#
# Note: each run scans the full image corpus, so this is four full scans.

set -euo pipefail

QUERY_JSON="${QUERY_JSON:-Test.json}"
QRELS="${QRELS:-Test_figure_qrels.tsv}"
IMAGE_DIR="${IMAGE_DIR:-/mnt/netstore1_home/adah.holt/figures/images}"
ADAPTER="${ADAPTER:-colqwen_qlora_run/best_adapter}"
OUT_DIR="${OUT_DIR:-runs_2x2}"

mkdir -p "$OUT_DIR"

for prec in bf16 4bit; do
  QFLAG=()
  if [ "$prec" = "4bit" ]; then QFLAG=(--quantize_inference); fi

  echo "=== base / $prec ==="
  python colqwen_retrieval_finetuned.py \
    --query_json "$QUERY_JSON" --image_dir "$IMAGE_DIR" --qrels "$QRELS" \
    ${QFLAG[@]+"${QFLAG[@]}"} \
    --output "$OUT_DIR/base_${prec}.trec" \
    2>&1 | tee "$OUT_DIR/base_${prec}.log"

  echo "=== fine-tuned / $prec ==="
  python colqwen_retrieval_finetuned.py \
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