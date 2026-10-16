"""
Per-query evaluation of ColQwen retrieval runs with ranx.

Compares the runs below on the test set and writes:
  - overall mean metrics for every run
  - a significance test between runs (paired t-test, via ranx.compare)
  - a per-query CSV (one row per query, one column per run x metric, plus
    fine-tuned minus baseline deltas, and each query's difficulty label)
  - wins / losses / ties per query, and nDCG@10 broken down by difficulty

Defaults point at the post-fix runs on the netstore. Run from the repo
folder that also contains eval_utils.py:
    python evaluate_colqwen.py

Compare other runs with --run NAME=PATH (repeatable), e.g.
    python evaluate_colqwen.py --run pretrained_4bit=runs_2x2/base_4bit.trec \
                               --run lr1e5=runs_lr1e5/finetuned_4bit.trec \
                               --baseline pretrained_4bit --challenger lr1e5
Relative run paths are resolved against --data_dir.
"""

import argparse
import json
import warnings
from pathlib import Path

warnings.filterwarnings("ignore")  # silence numba performance warnings from ranx

import pandas as pd
from ranx import Qrels, Run, compare, evaluate

from eval_utils import load_qrels  # accepts both 3- and 4-column qrels files


def default_data_dir():
    for cand in (Path.home() / "netstore1" / "finetuning",
                 Path("/mnt/netstore1_home") / Path.home().name / "finetuning"):
        if cand.is_dir():
            return cand
    return Path.home() / "netstore1" / "finetuning"


DEFAULT_RUNS = {
    "pretrained_bf16": "runs_2x2/base_bf16.trec",
    "pretrained_4bit": "runs_2x2/base_4bit.trec",
    "finetuned_4bit":  "runs_hardneg/finetuned_4bit.trec",
}
METRICS = ["ndcg@10", "mrr", "recall@10", "recall@100", "precision@10"]


def load_query_info(json_path):
    """query_id -> {label, num_source_figures} from Test.json (if available)."""
    try:
        with open(json_path, encoding="utf-8") as f:
            data = json.load(f)
    except FileNotFoundError:
        print(f"[WARN] {json_path} not found; skipping difficulty breakdown.")
        return {}
    if isinstance(data, dict):
        data = data.get("queries", list(data.values()))
    return {
        str(q["query_id"]): {
            "label": q.get("label", "unknown"),
            "num_source_figures": q.get("num_source_figures"),
        }
        for q in data
    }


def main():
    p = argparse.ArgumentParser(description="Per-query comparison of ColQwen TREC runs")
    p.add_argument("--data_dir", default=str(default_data_dir()))
    p.add_argument("--qrels", default="Test_figure_qrels.tsv")
    p.add_argument("--queries", default="Test.json")
    p.add_argument("--run", action="append", default=None, metavar="NAME=PATH",
                   help="A run to include (repeatable). Default: the three post-fix test runs.")
    p.add_argument("--baseline", default="pretrained_4bit")
    p.add_argument("--challenger", default="finetuned_4bit")
    p.add_argument("--out", default="colqwen_per_query_metrics.csv")
    args = p.parse_args()

    data_dir = Path(args.data_dir)
    resolve = lambda s: Path(s) if Path(s).is_absolute() else data_dir / s

    runs_spec = dict(r.split("=", 1) for r in args.run) if args.run else DEFAULT_RUNS
    for name in (args.baseline, args.challenger):
        if name not in runs_spec:
            raise SystemExit(f"[ERROR] --baseline/--challenger '{name}' is not one of the runs: {list(runs_spec)}")

    # ---------------------------------------------------------------- load
    qrels = Qrels.from_dict(load_qrels(resolve(args.qrels)))
    print(f"Qrels: {resolve(args.qrels)}  ({len(qrels.keys())} queries)")

    runs = {}
    for name, path in runs_spec.items():
        path = resolve(path)
        if not path.exists():
            raise SystemExit(f"[ERROR] Run file not found: {path}")
        run = Run.from_file(str(path), kind="trec")
        run.name = name
        missing = set(qrels.keys()) - set(run.keys())
        print(f"Run {name:<16} {path}  ({len(run.keys())} queries"
              + (f", {len(missing)} qrels queries missing -> scored 0" if missing else "") + ")")
        runs[name] = run

    info = load_query_info(resolve(args.queries))

    # ---------------------------------------------------------------- overall
    # make_comparable=True: queries absent from a run count as 0 instead of
    # being silently dropped. Per-query scores are stored in run.scores keyed by
    # query id, so nothing depends on ordering.
    print("\n" + "=" * 70)
    print("OVERALL (mean over all test queries)")
    print("=" * 70)
    overall = {name: evaluate(qrels, run, METRICS, make_comparable=True) for name, run in runs.items()}
    print(pd.DataFrame(overall).T[METRICS].round(4).to_string())

    print("\n" + "=" * 70)
    print("SIGNIFICANCE (paired t-test; superscript letters = significantly better")
    print("than that run at p < 0.05)")
    print("=" * 70)
    print(compare(qrels, runs=list(runs.values()), metrics=METRICS,
                  stat_test="student", max_p=0.05, make_comparable=True))

    # ---------------------------------------------------------------- per query
    rows = []
    for qid in sorted(qrels.keys()):
        row = {"query_id": qid,
               "label": info.get(qid, {}).get("label"),
               "num_source_figures": info.get(qid, {}).get("num_source_figures")}
        for name, run in runs.items():
            for m in METRICS:
                row[f"{name}__{m}"] = float(run.scores[m].get(qid, 0.0))
        for m in METRICS:
            row[f"delta__{m}"] = row[f"{args.challenger}__{m}"] - row[f"{args.baseline}__{m}"]
        rows.append(row)
    df = pd.DataFrame(rows)
    df.to_csv(args.out, index=False)
    print(f"\nPer-query metrics saved to {args.out}  ({len(df)} queries)")

    # ---------------------------------------------------------------- summaries
    d = df["delta__ndcg@10"]
    print("\n" + "=" * 70)
    print(f"PER-QUERY nDCG@10: {args.challenger} vs {args.baseline}")
    print("=" * 70)
    print(f"  better: {(d > 1e-9).sum():>3}   worse: {(d < -1e-9).sum():>3}   same: {(d.abs() <= 1e-9).sum():>3}")
    zero = (df[[f"{n}__ndcg@10" for n in runs]] == 0).all(axis=1).sum()
    print(f"  queries where EVERY run scores 0: {zero}  "
          f"(often relevant figures missing from disk, or genuinely hard queries)")

    if df["label"].notna().any():
        cols = [f"{n}__ndcg@10" for n in runs] + ["delta__ndcg@10"]
        by_label = df.groupby("label")[cols].mean()
        by_label.insert(0, "n_queries", df.groupby("label").size())
        by_label.columns = ["n_queries"] + list(runs) + ["delta"]
        print("\nnDCG@10 by difficulty label:")
        print(by_label.round(4).to_string())

        by_nfig = df.groupby("num_source_figures")[cols].mean()
        by_nfig.insert(0, "n_queries", df.groupby("num_source_figures").size())
        by_nfig.columns = ["n_queries"] + list(runs) + ["delta"]
        print("\nnDCG@10 by number of relevant figures per query:")
        print(by_nfig.round(4).to_string())

    print("\nBiggest gains / losses for the challenger (nDCG@10):")
    show = ["query_id", "label", f"{args.baseline}__ndcg@10", f"{args.challenger}__ndcg@10", "delta__ndcg@10"]
    print(df.nlargest(5, "delta__ndcg@10")[show].round(3).to_string(index=False))
    print(df.nsmallest(5, "delta__ndcg@10")[show].round(3).to_string(index=False))


if __name__ == "__main__":
    main()