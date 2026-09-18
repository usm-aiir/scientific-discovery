# Objective: Retrieve tables for a query file and save six-column TREC rankings.

import argparse
import json
from pathlib import Path
import time

from .bm25 import BM25
from ..data import DATA, check_id, load_json


def main():
    """Build one index, search all supplied queries, and record run settings."""
    parser = argparse.ArgumentParser(description=__doc__ or "BM25 table retrieval")
    parser.add_argument("--corpus", type=Path, default=DATA / "Corpus.json")
    parser.add_argument("--queries", type=Path, default=DATA / "Val.json")
    parser.add_argument("--output", type=Path, default=Path("results/bm25_val.run"))
    parser.add_argument("--top-k", type=int, default=10)
    parser.add_argument("--k1", type=float, default=1.2)
    parser.add_argument("--b", type=float, default=0.75)
    parser.add_argument("--run-name", default="bm25_baseline")
    args = parser.parse_args()
    if args.top_k < 1:
        parser.error("--top-k must be positive")
    check_id(args.run_name)
    queries = load_json(args.queries)
    query_ids = [check_id(q["query_id"]) for q in queries]
    if not queries or len(set(query_ids)) != len(query_ids):
        raise ValueError("Queries must be nonempty and have unique IDs.")
    if any(not isinstance(q["query"], str) for q in queries):
        raise ValueError("Each query must contain a text string.")
    started = time.perf_counter()
    print("Loading corpus and building BM25 index...", flush=True)
    corpus = load_json(args.corpus)
    retriever = BM25(corpus, k1=args.k1, b=args.b)
    del corpus
    indexed = time.perf_counter()
    print(f"Indexed {len(retriever.table_ids):,} tables in {indexed - started:.1f}s.", flush=True)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    no_matches = []
    with args.output.open("w", encoding="utf-8") as output:
        for qid, query in zip(query_ids, queries):
            # Explanations/source IDs are deliberately never used for retrieval.
            results = retriever.search(query["query"], args.top_k)
            if not results:
                no_matches.append(qid)
            for rank, (uid, score) in enumerate(results, start=1):
                output.write(f"{qid} Q0 {uid} {rank} {score:.12g} {args.run_name}\n")
    finished = time.perf_counter()
    settings = {key: str(value) if isinstance(value, Path) else value for key, value in vars(args).items()}
    settings.update(table_count=len(retriever.table_ids), query_count=len(queries),
                    index_seconds=indexed - started, search_seconds=finished - indexed,
                    queries_without_matches=no_matches,
                    representation="caption + sub_caption + header-labelled rows + reference_text",
                    tokenizer="lowercase words/numbers, preserving internal dots and hyphens",
                    idf="log(1 + (N - df + 0.5) / (df + 0.5))")
    args.output.with_suffix(".settings.json").write_text(json.dumps(settings, indent=2) + "\n")
    print(f"Wrote {args.output}; {len(no_matches)} queries had no matching terms.")


if __name__ == "__main__":
    main()
