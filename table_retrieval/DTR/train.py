# Objective: Fine-tune BERT + TAPAS using training queries and table relevance judgments.
"""Fine-tune BERT + TAPAS using training queries and table relevance judgments."""

import argparse
from collections import defaultdict
from dataclasses import asdict
import json
import math
from pathlib import Path
import random
import warnings

import numpy as np
import torch
from tqdm.auto import tqdm

from ..data import check_id, load_json
from .model import DenseTableRetriever, InputLimits, contrastive_loss, select_device
from ..evaluate import normalize_table_id
from ..data import DATA


def training_examples(queries, qrels_path, corpus, min_grade=1):
    """Resolve qrels to corpus IDs; never use explanations as encoder inputs."""
    lookup = {}
    for uid in corpus:
        normalized = normalize_table_id(check_id(uid))
        if normalized in lookup:
            raise ValueError(f"Ambiguous corpus table ID: {normalized}")
        lookup[normalized] = uid
    positives = defaultdict(set)
    query_ids = [check_id(query["query_id"]) for query in queries]
    if len(set(query_ids)) != len(query_ids):
        raise ValueError("Duplicate training query IDs.")
    allowed = set(query_ids)
    missing = set()
    for line in Path(qrels_path).read_text().splitlines():
        if not line.strip():
            continue
        qid, _, uid, grade = line.split()
        if qid not in allowed or int(grade) < min_grade:
            continue
        normalized = normalize_table_id(uid)
        if normalized in lookup:
            positives[qid].add(lookup[normalized])
        else:
            missing.add(uid)
    examples = []
    skipped = []
    for qid, query in zip(query_ids, queries):
        if not isinstance(query["query"], str) or not query["query"].strip():
            raise ValueError(f"Training query {qid} must contain nonempty text.")
        if not positives[qid]:
            skipped.append(qid)
        elif len(positives[qid]) == len(corpus):
            raise ValueError(f"Query {qid} has no negative tables in the corpus.")
        else:
            examples.append((qid, query["query"], positives[qid]))
    if not examples:
        raise ValueError("No training queries have positive judgments in the corpus.")
    report = {"missing_judged_table_ids": sorted(missing), "skipped_query_ids": skipped}
    if missing or skipped:
        warnings.warn(f"Training data: {len(missing)} missing judged table IDs; "
                      f"{len(skipped)} queries skipped. Details saved with the checkpoint.")
    return examples, report


def make_batch(examples, table_ids, rng):
    """Sample one positive and one random negative per query, then deduplicate.

    Other queries' sampled tables are in-batch negatives unless the qrels mark
    them relevant too. This avoids treating a known positive as a negative.
    """
    candidates = set()
    for _, _, relevant in examples:
        candidates.add(rng.choice(sorted(relevant)))
        negative = rng.choice(table_ids)
        while negative in relevant:
            negative = rng.choice(table_ids)
        candidates.add(negative)
    candidates = sorted(candidates)
    mask = torch.tensor([[uid in relevant for uid in candidates] for _, _, relevant in examples],
                        dtype=torch.bool)
    return candidates, mask


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--corpus", type=Path, default=DATA / "Corpus.json")
    parser.add_argument("--queries", type=Path, default=DATA / "Train.json")
    parser.add_argument("--qrels", type=Path, default=DATA / "Train_table_qrels.tsv")
    parser.add_argument("--output", type=Path, default=Path("results/dtr_model"))
    parser.add_argument("--query-model", default="bert-base-uncased")
    parser.add_argument("--table-model", default="google/tapas-base")
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--learning-rate", type=float, default=2e-5)
    parser.add_argument("--temperature", type=float, default=0.05)
    parser.add_argument("--min-grade", type=int, default=1)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", default="auto")
    for name, default in asdict(InputLimits()).items():
        parser.add_argument("--" + name.replace("_", "-"), type=int, default=default)
    args = parser.parse_args()
    if args.epochs < 1 or args.batch_size < 1:
        parser.error("epochs and batch-size must be positive")
    if any(not math.isfinite(x) or x <= 0 for x in (args.learning_rate, args.temperature)):
        parser.error("learning-rate and temperature must be finite and positive")
    if args.output.exists() and (not args.output.is_dir() or any(args.output.iterdir())):
        parser.error("output must be a new or empty directory")
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    rng = random.Random(args.seed)
    print("Loading training queries, judgments, and corpus...", flush=True)
    corpus = load_json(args.corpus)
    examples, data_report = training_examples(load_json(args.queries), args.qrels, corpus, args.min_grade)
    limits = InputLimits(**{name: getattr(args, name) for name in asdict(InputLimits())})
    device = select_device(args.device)
    print(f"Loading BERT and TAPAS on {device}...", flush=True)
    model = DenseTableRetriever.from_pretrained(args.query_model, args.table_model, limits).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.learning_rate, weight_decay=0.01)
    table_ids = sorted(corpus)
    history = []
    print(f"Training on {len(examples):,} queries using {device}.", flush=True)
    for epoch in range(args.epochs):
        model.train()
        rng.shuffle(examples)
        total_loss = 0.0
        with tqdm(total=math.ceil(len(examples) / args.batch_size),
                  desc=f"Epoch {epoch + 1}/{args.epochs}", unit="batch", dynamic_ncols=True) as progress:
            for start in range(0, len(examples), args.batch_size):
                batch = examples[start:start + args.batch_size]
                candidates, mask = make_batch(batch, table_ids, rng)
                optimizer.zero_grad(set_to_none=True)
                query_vectors = model.encode_queries([text for _, text, _ in batch])
                table_vectors = model.encode_tables([corpus[uid] for uid in candidates])
                loss = contrastive_loss(query_vectors, table_vectors, mask.to(device), args.temperature)
                if not torch.isfinite(loss):
                    raise ValueError("Non-finite training loss.")
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()
                batch_loss = loss.item()
                total_loss += batch_loss * len(batch)
                progress.set_postfix(loss=f"{batch_loss:.4f}",
                                     mean_loss=f"{total_loss / (start + len(batch)):.4f}", refresh=False)
                progress.update(1)
        history.append(total_loss / len(examples))
        print(f"Epoch {epoch + 1}/{args.epochs}: mean loss={history[-1]:.4f}", flush=True)
    settings = {key: str(value) if isinstance(value, Path) else value for key, value in vars(args).items()}
    settings.update(data_report, epoch_losses=history, training_queries=len(examples))
    print(f"Saving trained encoders to {args.output}...", flush=True)
    model.save(args.output, settings)
    print(f"Saved retriever to {args.output}.")


if __name__ == "__main__":
    main()
