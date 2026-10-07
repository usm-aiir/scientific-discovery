from ranx import Qrels, Run, evaluate, compare
import pandas as pd

qrels = Qrels.from_file("/home/adah.holt/scientific-discovery/Test_figure_qrels.tsv", kind="trec")

run_pretrained = Run.from_file("/home/adah.holt/scientific-discovery/run_colqwen_pretrained.trec")
run_finetuned  = Run.from_file("/home/adah.holt/scientific-discovery/run_colqwen_finetuned.trec")

metrics = ["ndcg@10", "mrr", "recall@10", "recall@100", "precision@10"]

# ── Overall (mean across queries) ──────────────────────────────────────────────
print("=" * 60)
print("OVERALL METRICS (mean across all queries)")
print("=" * 60)

print("\n=== Pretrained ===")
print(evaluate(qrels, run_pretrained, metrics))

print("\n=== Finetuned ===")
print(evaluate(qrels, run_finetuned, metrics))

print("\n=== Comparison ===")
print(compare(qrels, runs=[run_pretrained, run_finetuned], metrics=metrics))

# ── Per-query metrics ──────────────────────────────────────────────────────────
print("\n" + "=" * 60)
print("PER-QUERY METRICS")
print("=" * 60)

# ranx returns a numpy array (one score per query) sorted by query_id
query_ids = sorted(qrels.qrels.keys())

for metric in metrics:
    print(f"\n── {metric} ──")

    pre_arr = evaluate(qrels, run_pretrained, metric, return_mean=False)
    ft_arr  = evaluate(qrels, run_finetuned,  metric, return_mean=False)

    pre_scores = dict(zip(query_ids, pre_arr))
    ft_scores  = dict(zip(query_ids, ft_arr))

    rows = []
    for qid in query_ids:
        p = pre_scores.get(qid, 0.0)
        f = ft_scores.get(qid, 0.0)
        rows.append({"query_id": qid, "pretrained": p, "finetuned": f, "delta": f - p})

    df = pd.DataFrame(rows).sort_values("delta", ascending=False)
    print(df.to_string(index=False, float_format=lambda x: f"{x:.4f}"))

# ── Save full per-query table to CSV ───────────────────────────────────────────
all_rows = []
for metric in metrics:
    pre_arr = evaluate(qrels, run_pretrained, metric, return_mean=False)
    ft_arr  = evaluate(qrels, run_finetuned,  metric, return_mean=False)
    pre_scores = dict(zip(query_ids, pre_arr))
    ft_scores  = dict(zip(query_ids, ft_arr))
    for qid in query_ids:
        all_rows.append({
            "query_id":   qid,
            "metric":     metric,
            "pretrained": pre_scores.get(qid, 0.0),
            "finetuned":  ft_scores.get(qid, 0.0),
            "delta":      ft_scores.get(qid, 0.0) - pre_scores.get(qid, 0.0),
        })

out_csv = "colqwen_per_query_metrics.csv"
pd.DataFrame(all_rows).to_csv(out_csv, index=False)
print(f"\nPer-query metrics saved to {out_csv}")