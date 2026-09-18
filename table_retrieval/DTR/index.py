# Objective: Encode corpus tables once with a trained TAPAS dense retriever.
"""Encode corpus tables once with a trained TAPAS dense retriever."""

import argparse
import json
from pathlib import Path

import numpy as np
import torch

from ..data import check_id, load_json
from .model import DenseTableRetriever, checkpoint_fingerprint, select_device
from ..data import DATA


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--corpus", type=Path, default=DATA / "Corpus.json")
    parser.add_argument("--model", type=Path, default=Path("results/dtr_model"))
    parser.add_argument("--output", type=Path, default=Path("results/dtr_index"))
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--device", default="auto")
    args = parser.parse_args()
    if args.batch_size < 1:
        parser.error("batch-size must be positive")
    if args.output.exists() and (not args.output.is_dir() or any(args.output.iterdir())):
        parser.error("output must be a new or empty directory")
    corpus = load_json(args.corpus)
    if not corpus:
        raise ValueError("The corpus is empty.")
    table_ids = sorted(check_id(uid) for uid in corpus)
    device = select_device(args.device)
    model = DenseTableRetriever.load(args.model).to(device).eval()
    fingerprint = checkpoint_fingerprint(args.model)
    args.output.mkdir(parents=True, exist_ok=True)
    embeddings = np.lib.format.open_memmap(args.output / "embeddings.npy", mode="w+",
        dtype=np.float32, shape=(len(table_ids), model.table_encoder.config.hidden_size))
    with torch.inference_mode():
        for start in range(0, len(table_ids), args.batch_size):
            ids = table_ids[start:start + args.batch_size]
            vectors = model.encode_tables([corpus[uid] for uid in ids]).cpu().numpy()
            if not np.isfinite(vectors).all():
                raise ValueError(f"Non-finite embeddings for batch starting at {ids[0]}.")
            embeddings[start:start + len(ids)] = vectors
            if start == 0 or start // args.batch_size % 25 == 0:
                print(f"Encoded {start + len(ids):,}/{len(table_ids):,} tables.", flush=True)
    embeddings.flush()
    (args.output / "table_ids.json").write_text(json.dumps(table_ids) + "\n")
    # Written last: an interrupted index is never accepted as complete.
    (args.output / "index.json").write_text(json.dumps({
        "model_fingerprint": fingerprint, "corpus": str(args.corpus),
        "table_count": len(table_ids), "dimension": embeddings.shape[1],
        "similarity": "cosine", "model": str(args.model)}, indent=2) + "\n")
    print(f"Indexed {len(table_ids):,} tables in {args.output}.")


if __name__ == "__main__":
    main()
