# Objective: Retrieve tables with exact cosine search over saved TAPAS table embeddings.
"""Retrieve tables with exact cosine search over saved TAPAS table embeddings."""

import argparse
import json
from pathlib import Path

import numpy as np
import torch

from ..data import check_id, load_json
from .model import DenseTableRetriever, checkpoint_fingerprint, select_device
from ..data import DATA


def rank_tables(query_vectors, table_vectors, top_k):
    """Return exact top-k indices and scores; corpus order breaks ties."""
    if top_k < 1:
        raise ValueError("top_k must be positive.")
    scores = query_vectors @ table_vectors.T
    if not np.isfinite(scores).all():
        raise ValueError("Non-finite retrieval scores.")
    indices = np.argsort(-scores, axis=1, kind="stable")[:, :top_k]
    return indices, np.take_along_axis(scores, indices, axis=1)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", type=Path, default=Path("results/dtr_model"))
    parser.add_argument("--index", type=Path, default=Path("results/dtr_index"))
    parser.add_argument("--queries", type=Path, default=DATA / "Val.json")
    parser.add_argument("--output", type=Path, default=Path("results/dtr_val.run"))
    parser.add_argument("--top-k", type=int, default=10)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--device", default="auto")
    args = parser.parse_args()
    if args.top_k < 1 or args.batch_size < 1:
        parser.error("top-k and batch-size must be positive")
    metadata = load_json(args.index / "index.json")
    if checkpoint_fingerprint(args.model) != metadata["model_fingerprint"]:
        raise ValueError("Index and model do not match. Rebuild the index with this checkpoint.")
    table_ids = load_json(args.index / "table_ids.json")
    vectors = np.load(args.index / "embeddings.npy", mmap_mode="r", allow_pickle=False)
    if (vectors.shape != (len(table_ids), metadata["dimension"]) or
            len(table_ids) != metadata["table_count"] or not table_ids or
            table_ids != sorted(set(table_ids))):
        raise ValueError("Invalid index shape or table IDs.")
    for uid in table_ids:
        check_id(uid)
    queries = load_json(args.queries)
    query_ids = [check_id(query["query_id"]) for query in queries]
    if not queries or len(set(query_ids)) != len(query_ids):
        raise ValueError("Queries must be nonempty and have unique IDs.")
    if any(not isinstance(query["query"], str) or not query["query"].strip() for query in queries):
        raise ValueError("Queries must contain nonempty text.")
    model = DenseTableRetriever.load(args.model).to(select_device(args.device)).eval()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with torch.inference_mode(), args.output.open("w") as output:
        for start in range(0, len(queries), args.batch_size):
            batch = queries[start:start + args.batch_size]
            query_vectors = model.encode_queries([q["query"] for q in batch]).cpu().numpy()
            indices, scores = rank_tables(query_vectors, vectors, args.top_k)
            for qid, hits, values in zip(query_ids[start:], indices, scores):
                for rank, (index, score) in enumerate(zip(hits, values), 1):
                    output.write(f"{qid} Q0 {table_ids[index]} {rank} {score:.12g} tapas_dtr\n")
    settings = {key: str(value) if isinstance(value, Path) else value for key, value in vars(args).items()}
    settings.update(query_count=len(queries), table_count=len(table_ids),
                    model_fingerprint=metadata["model_fingerprint"], similarity="cosine")
    args.output.with_suffix(".settings.json").write_text(json.dumps(settings, indent=2) + "\n")
    print(f"Wrote {args.output}; retrieved {min(args.top_k, len(table_ids))} tables per query.")


if __name__ == "__main__":
    main()
