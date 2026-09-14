#!/usr/bin/env python3
"""
colpali_retrieval.py
=====================
ColPali-based figure retrieval over the arXiv corpus produced by
arxiv_scraper.py.  Mirrors the interface of figure_retrieval.py so both
retrievers can be swapped in or compared without changing any downstream code.

ColPali vs CLIP
---------------
CLIP produces ONE vector per image and ONE vector per query, then scores via
a single dot product.

ColPali produces MANY vectors per image (one per image patch) and MANY vectors
per query (one per token), then scores via "late interaction" (MaxSim): for
each query token, find the most similar image patch, then sum those max
similarities across all query tokens.  This patch-level matching is why
ColPali is much better at fine-grained scientific content -- it can match the
query token "bar chart" to the specific patch in the figure that contains it.

Model used: vidore/colqwen2-v1.0  (Apache 2.0, best open-license performance)
Fallback:   vidore/colsmol-500m   (Apache 2.0, smaller GPU footprint)

Installation
------------
    pip install colpali-engine

    # Optional: fused MaxSim kernels for faster scoring on large corpora
    pip install colpali-engine[lik]

Usage
-----
    # Interactive search:
    python colpali_retrieval.py 25_04 25_05

    # Evaluate Recall@K / MRR against the query TSV:
    python colpali_retrieval.py 25_04 --evaluate --queries_tsv claude_figure_queries.tsv

    # Smaller model for limited GPU:
    python colpali_retrieval.py 25_04 --model vidore/colsmol-500m
"""

from __future__ import annotations

import argparse
import base64
import json
import pickle
from dataclasses import dataclass
from io import BytesIO
from pathlib import Path
from typing import List, Optional

import numpy as np
import torch
from PIL import Image

# colpali-engine must be installed: pip install colpali-engine
try:
    from colpali_engine.models import ColQwen2, ColQwen2Processor
except ImportError:
    raise ImportError(
        "colpali-engine is not installed.\n"
        "Run:  pip install colpali-engine"
    )


# ---------------------------------------------------------------------------
# Result dataclass  (identical to figure_retrieval.FigureResult so both
# retrievers can be used interchangeably by downstream code)
# ---------------------------------------------------------------------------

@dataclass
class FigureResult:
    """
    One retrieved figure with all the context the downstream LLM agent needs.

    Identical to the FigureResult in figure_retrieval.py -- whichever
    retriever you use (CLIP or ColPali), the rest of the pipeline sees the
    same object.
    """
    image: Image.Image          # PIL image, RGB
    score: float                # retrieval score (higher = better)
    paper_id: str               # e.g. "2504.12345"
    figure_id: int              # numeric figure number in the paper
    sub_id: Optional[str]       # sub-panel label e.g. "a", "b", or None
    image_id: str               # filename stem: "<paper_id>_<fig_id>[_<sub_id>]"
    caption: str                # full figure caption
    sub_caption: Optional[str]  # sub-panel caption, or None
    context_paras: List[str]    # body paragraphs that cite this figure
    title: str                  # paper title
    file_path: str              # absolute path to the image file on disk

    def to_base64(self) -> str:
        """Encode the image as base64 PNG for multimodal LLM API calls."""
        buf = BytesIO()
        self.image.save(buf, format="PNG")
        return base64.b64encode(buf.getvalue()).decode("utf-8")

    def as_text_context(self, max_paras: int = 3, max_para_chars: int = 400) -> str:
        """Compact text block for the LLM agent's prompt (citation + context)."""
        lines = [
            f"[Figure {self.image_id}]",
            f"Paper: {self.title or self.paper_id}  (id: {self.paper_id})",
            f"Retrieval score: {self.score:.3f}",
            f"Caption: {self.caption}",
        ]
        if self.sub_caption:
            lines.append(f"Sub-panel caption: {self.sub_caption}")
        if self.context_paras:
            lines.append(f"Cited in the paper "
                         f"({min(len(self.context_paras), max_paras)} of "
                         f"{len(self.context_paras)} paragraph(s) shown):")
            for para in self.context_paras[:max_paras]:
                snippet = para[:max_para_chars]
                if len(para) > max_para_chars:
                    snippet += "..."
                lines.append(f"  • {snippet}")
        return "\n".join(lines)


# ---------------------------------------------------------------------------
# Retriever
# ---------------------------------------------------------------------------

class ColPaliRetriever:
    """
    ColPali late-interaction retriever over arxiv_scraper.py output directories.

    On first run: scans the corpus, encodes every figure with ColQwen2's image
    encoder, and saves a cache.  Subsequent runs load the cache instantly.

    Cache format
    ------------
    The cache stores a list of per-image embedding arrays (one numpy array of
    shape (seq_len, dim) per figure) alongside the metadata records.  This is
    different from CLIP's flat (N, D) matrix because ColPali's sequence lengths
    can vary slightly between batches after padding is removed.

    Memory note
    -----------
    ColPali embeddings are larger than CLIP's: each image produces ~1030 patch
    vectors of dim 128 (ColQwen2) vs. one 768-dim vector (CLIP-L).  For a
    corpus of ~5,000 figures this is roughly 2.6 GB uncompressed.  Use
    --pool_factor 3 to reduce by ~67% with <2% performance loss.
    """

    def __init__(
        self,
        data_dirs: List[str | Path],
        cache_path: str | Path = "colpali_cache.pkl",
        model_name: str = "vidore/colqwen2-v1.0",
        device: Optional[str] = None,
        batch_size: int = 4,        # ColPali is memory-heavy; keep batches small
        pool_factor: int = 1,       # set to 3 for large corpora to save memory
    ) -> None:
        """
        Parameters
        ----------
        data_dirs:    one or more arxiv_scraper.py output directories.
        cache_path:   pickle cache path (built on first run).
        model_name:   ColPali model on HuggingFace Hub.
                      "vidore/colqwen2-v1.0"  -- best Apache 2.0 model
                      "vidore/colsmol-500m"   -- smaller, less GPU memory
        device:       "cuda" / "cpu" (auto-detected if None).
                      ColPali strongly prefers CUDA; CPU inference is very slow.
        batch_size:   images per forward pass.  ColQwen2 is ~4-8x heavier than
                      CLIP-L per image; start at 4 and increase if GPU allows.
        pool_factor:  token pooling compression factor (1 = no pooling).
                      pool_factor=3 cuts embedding storage by ~67% with
                      <2% score drop on retrieval benchmarks.
        """
        self.data_dirs   = [Path(d) for d in data_dirs]
        self.cache_path  = Path(cache_path)
        self.model_name  = model_name
        self.device      = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.batch_size  = batch_size
        self.pool_factor = pool_factor

        self._model: Optional[ColQwen2]         = None
        self._processor: Optional[ColQwen2Processor] = None

        # Populated by _load_or_build_cache():
        self._records:    List[dict]         = []
        self._embeddings: List[np.ndarray]   = []   # one (seq_len, dim) array per figure

        self._load_or_build_cache()

    # ------------------------------------------------------------------ model

    def _get_model(self) -> tuple[ColQwen2, ColQwen2Processor]:
        if self._model is None:
            if self.device == "cpu":
                print("WARNING: ColPali on CPU is very slow. "
                      "A GPU is strongly recommended.")
            print(f"Loading ColPali model  {self.model_name}  on {self.device} ...")
            self._model = ColQwen2.from_pretrained(
                self.model_name,
                torch_dtype=torch.bfloat16,
                device_map=self.device,
            ).eval()
            self._processor = ColQwen2Processor.from_pretrained(self.model_name)
        return self._model, self._processor

    # ----------------------------------------------------------- corpus scan
    # (identical logic to figure_retrieval.py -- same scraper output format)

    def _scan_corpus(self) -> List[dict]:
        records: List[dict] = []
        for data_dir in self.data_dirs:
            figures_dir  = data_dir / "figures"
            metadata_dir = data_dir / "metadata"
            if not figures_dir.is_dir():
                print(f"WARNING: no figures/ folder in {data_dir}, skipping.")
                continue
            for figures_json in sorted(figures_dir.glob("*.json")):
                paper_id = figures_json.stem
                try:
                    with open(figures_json, encoding="utf-8") as f:
                        fig_data = json.load(f)
                except Exception as exc:
                    print(f"WARNING: could not read {figures_json}: {exc}")
                    continue
                figures  = fig_data.get("figures", [])
                fig_refs = fig_data.get("figure_references", {})
                title = ""
                meta_path = metadata_dir / f"{paper_id}.json"
                if meta_path.exists():
                    try:
                        with open(meta_path, encoding="utf-8") as f:
                            title = json.load(f).get("title", "")
                    except Exception:
                        pass
                for fig in figures:
                    local_path = fig.get("local_path")
                    if not local_path:
                        continue
                    abs_path = (figures_dir / local_path).resolve()
                    if not abs_path.exists():
                        continue
                    figure_id     = fig.get("figure_id")
                    context_paras = fig_refs.get(f"Figure {figure_id}", [])
                    records.append({
                        "paper_id":      paper_id,
                        "figure_id":     figure_id,
                        "sub_id":        fig.get("sub_id"),
                        "image_id":      fig.get("image_id", ""),
                        "caption":       fig.get("caption", ""),
                        "sub_caption":   fig.get("sub_caption"),
                        "context_paras": context_paras,
                        "title":         title,
                        "file_path":     str(abs_path),
                    })
        print(f"Corpus scan complete: {len(records):,} figures across "
              f"{len(self.data_dirs)} director"
              f"{'y' if len(self.data_dirs) == 1 else 'ies'}.")
        return records

    # ---------------------------------------------------------- embedding

    def _embed_records(self, records: List[dict]) -> List[np.ndarray]:
        """
        Encode every figure with ColQwen2's image encoder.

        Returns a list of numpy arrays, one per figure, each of shape
        (seq_len, dim).  Failed images get a single zero row so the record
        list and embedding list stay in sync.
        """
        model, processor = self._get_model()
        all_embeddings: List[np.ndarray] = []

        for start in range(0, len(records), self.batch_size):
            batch   = records[start : start + self.batch_size]
            images  = []
            valid   = []          # indices within the batch that loaded ok
            for local_i, rec in enumerate(batch):
                try:
                    images.append(Image.open(rec["file_path"]).convert("RGB"))
                    valid.append(local_i)
                except Exception as exc:
                    print(f"WARNING: could not open {rec['file_path']}: {exc}")

            # Place-holder zero embeddings for failed images, filled in below.
            batch_result: List[Optional[np.ndarray]] = [None] * len(batch)

            if images:
                inputs = processor.process_images(images).to(self.device)
                with torch.no_grad():
                    embs = model(**inputs)   # (n_valid, seq_len, dim)

                # Optional token pooling to reduce embedding storage size.
                if self.pool_factor > 1:
                    try:
                        from colpali_engine.compression.token_pooling import (
                            HierarchicalTokenPooler,
                        )
                        pooler = HierarchicalTokenPooler()
                        embs = pooler.pool_embeddings(
                            embs,
                            pool_factor=self.pool_factor,
                            padding=True,
                            padding_side=processor.tokenizer.padding_side,
                        )
                    except ImportError:
                        print("WARNING: token pooling requested but "
                              "HierarchicalTokenPooler not available -- "
                              "skipping pooling.")

                embs_np = embs.float().cpu().numpy()  # (n_valid, seq_len, dim)
                for arr_i, batch_i in enumerate(valid):
                    batch_result[batch_i] = embs_np[arr_i]

            # Fill in zero placeholders for any image that failed to load.
            dim = batch_result[valid[0]].shape[-1] if valid else 128
            for local_i in range(len(batch)):
                if batch_result[local_i] is None:
                    batch_result[local_i] = np.zeros((1, dim), dtype=np.float32)
                all_embeddings.append(batch_result[local_i].astype(np.float32))

            done = min(start + self.batch_size, len(records))
            if done % 100 == 0 or done == len(records):
                print(f"  Embedded {done:,} / {len(records):,} figures ...")

        return all_embeddings

    # --------------------------------------------------------------- cache

    def _save_cache(self) -> None:
        with open(self.cache_path, "wb") as f:
            pickle.dump(
                {"records": self._records, "embeddings": self._embeddings},
                f, protocol=pickle.HIGHEST_PROTOCOL,
            )
        print(f"Cache saved → {self.cache_path}  ({len(self._records):,} figures).")

    def _load_or_build_cache(self) -> None:
        if self.cache_path.exists():
            print(f"Loading ColPali cache from {self.cache_path} ...")
            with open(self.cache_path, "rb") as f:
                cache = pickle.load(f)
            self._records    = cache["records"]
            self._embeddings = cache["embeddings"]
            if len(self._records) != len(self._embeddings):
                raise RuntimeError(
                    f"Cache is corrupt: {len(self._records)} records but "
                    f"{len(self._embeddings)} embeddings.\n"
                    f"Delete {self.cache_path} and re-run to rebuild."
                )
            print(f"Loaded {len(self._records):,} figure embeddings.")
        else:
            print("No ColPali cache found — building from scratch ...")
            self._records = self._scan_corpus()
            if not self._records:
                raise RuntimeError(
                    "No figures found in the specified data directories."
                )
            self._embeddings = self._embed_records(self._records)
            self._save_cache()

    # --------------------------------------------------------------- query

    def _encode_query(self, query: str) -> torch.Tensor:
        """
        Encode a text query with ColQwen2's text encoder.
        Returns a 2-D tensor of shape (n_tokens, dim) on CPU.
        """
        model, processor = self._get_model()
        inputs = processor.process_queries([query]).to(self.device)
        with torch.no_grad():
            q_emb = model(**inputs)   # (1, n_tokens, dim)
        return q_emb[0].float().cpu()   # (n_tokens, dim)

    def _maxsim_scores(self, q_emb: torch.Tensor) -> np.ndarray:
        """
        Compute the ColBERT-style MaxSim score between the query embedding
        and every figure in the corpus.

        For each query token, find the most similar image patch (max over
        patches); sum those max similarities across all query tokens.

        Parameters
        ----------
        q_emb : (n_query_tokens, dim) float32 tensor

        Returns
        -------
        scores : (N,) float32 numpy array, one score per figure
        """
        scores = np.empty(len(self._embeddings), dtype=np.float32)
        q_np   = q_emb.numpy()                          # (n_q, dim)

        # Score in chunks to avoid loading every embedding into GPU at once.
        CHUNK = 512
        for start in range(0, len(self._embeddings), CHUNK):
            chunk_embs = self._embeddings[start : start + CHUNK]
            chunk_scores = np.array([
                # (n_q, dim) @ (dim, n_patches) -> (n_q, n_patches) -> max over patches -> sum
                float(np.sum(np.max(q_np @ img_emb.T, axis=1)))
                for img_emb in chunk_embs
            ], dtype=np.float32)
            scores[start : start + len(chunk_scores)] = chunk_scores

        return scores

    def search(self, query: str, top_k: int = 3) -> List[FigureResult]:
        """
        Retrieve the top-K figures most semantically relevant to *query*.

        Uses ColBERT-style MaxSim late interaction: each query token finds its
        best-matching image patch; scores are summed across query tokens.

        Returns
        -------
        List[FigureResult] sorted by descending score.
        """
        if not self._records:
            return []

        q_emb  = self._encode_query(query)
        scores = self._maxsim_scores(q_emb)

        k       = min(top_k, len(scores))
        top_idx = np.argpartition(scores, -k)[-k:]
        top_idx = top_idx[np.argsort(scores[top_idx])[::-1]]

        results: List[FigureResult] = []
        for idx in top_idx:
            rec   = self._records[idx]
            score = float(scores[idx])

            # Skip zero-placeholder embeddings (images that failed to encode).
            if self._embeddings[idx].shape[0] == 1 and np.all(self._embeddings[idx] == 0):
                continue

            try:
                img = Image.open(rec["file_path"]).convert("RGB")
            except Exception as exc:
                print(f"WARNING: could not open {rec['file_path']}: {exc}")
                continue

            results.append(FigureResult(
                image         = img,
                score         = score,
                paper_id      = rec["paper_id"],
                figure_id     = rec["figure_id"],
                sub_id        = rec["sub_id"],
                image_id      = rec["image_id"],
                caption       = rec["caption"],
                sub_caption   = rec["sub_caption"],
                context_paras = rec["context_paras"],
                title         = rec["title"],
                file_path     = rec["file_path"],
            ))

        return results


# ---------------------------------------------------------------------------
# Evaluation (same TSV format as evaluate_clip_retrieval.py)
# ---------------------------------------------------------------------------

def evaluate(
    retriever: ColPaliRetriever,
    queries_tsv: str,
    top_k: int = 10,
) -> None:
    """
    Evaluate against a ground-truth TSV:
        Paper ID | Figure ID | Query 1 | Query 2 | Query 3
    Reports Recall@1, Recall@5, Recall@K, and MRR.
    """
    import pandas as pd

    df = pd.read_csv(queries_tsv, sep="\t", dtype=str).fillna("")
    print(f"Loaded {len(df)} figures from {queries_tsv}")

    id_to_idx: dict[str, int] = {
        rec["image_id"]: i for i, rec in enumerate(retriever._records)
    }

    QUERY_COLUMNS = ["Query 1", "Query 2", "Query 3"]
    triples: List[tuple] = []
    for _, row in df.iterrows():
        paper_id = str(row.get("Paper ID", "")).strip()
        fig_id   = str(row.get("Figure ID", "")).strip()
        for col in QUERY_COLUMNS:
            q = str(row.get(col, "")).strip()
            if q:
                triples.append((q, paper_id, fig_id))

    print(f"Evaluating {len(triples)} queries ...")

    ks             = sorted({1, 5, top_k})
    recall_hits    = {k: 0 for k in ks}
    reciprocal_ranks = []
    n_no_match     = 0

    for query, paper_id, fig_id in triples:
        target_image_id = f"{paper_id}_{fig_id}"
        target_idx = id_to_idx.get(target_image_id)
        if target_idx is None:
            fig_base = fig_id.rstrip("abcdefghij")
            target_idx = id_to_idx.get(f"{paper_id}_{fig_base}")
        if target_idx is None:
            n_no_match += 1
            continue

        q_emb  = retriever._encode_query(query)
        scores = retriever._maxsim_scores(q_emb)
        rank   = int((scores > scores[target_idx]).sum()) + 1

        for k in ks:
            if rank <= k:
                recall_hits[k] += 1
        reciprocal_ranks.append(1.0 / rank)

    n_evaluated = len(triples) - n_no_match
    mrr = float(np.mean(reciprocal_ranks)) if reciprocal_ranks else 0.0

    print("\n" + "=" * 50)
    print("ColPali Retrieval Evaluation Results")
    print("=" * 50)
    print(f"Model           : {retriever.model_name}")
    print(f"Total queries   : {len(triples)}")
    print(f"No match found  : {n_no_match}")
    print(f"Evaluated       : {n_evaluated}")
    print()
    for k in ks:
        r = recall_hits[k] / n_evaluated if n_evaluated else 0.0
        print(f"  Recall@{k:<3}     : {r:.1%}")
    print(f"  MRR           : {mrr:.4f}")
    print("=" * 50)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(
        description="ColPali figure search over arxiv_scraper.py output directories.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Models
------
  vidore/colqwen2-v1.0   Best Apache 2.0 model (default, score 89.3 on ViDoRe)
  vidore/colsmol-500m    Smaller, lower GPU footprint (score 82.3)
  vidore/colpali-v1.3    Original ColPali model (Gemma license)

Examples
--------
  python colpali_retrieval.py 25_04 25_05
  python colpali_retrieval.py 25_04 --evaluate --queries_tsv claude_figure_queries.tsv
  python colpali_retrieval.py 25_04 --model vidore/colsmol-500m --pool_factor 3
        """,
    )
    parser.add_argument(
        "data_dirs", nargs="+",
        help="One or more arxiv_scraper.py output directories (e.g. 25_04 25_05)",
    )
    parser.add_argument(
        "--cache", default="colpali_cache.pkl",
        help="Path to the ColPali embedding cache (default: colpali_cache.pkl)",
    )
    parser.add_argument(
        "--model", default="vidore/colqwen2-v1.0",
        help="ColPali model name on HuggingFace Hub",
    )
    parser.add_argument(
        "--top_k", type=int, default=3,
        help="Number of results per query (default: 3)",
    )
    parser.add_argument(
        "--pool_factor", type=int, default=1,
        help="Token pooling factor to reduce embedding size (1=off, 3=recommended for large corpora)",
    )
    parser.add_argument(
        "--batch_size", type=int, default=4,
        help="Images per forward pass (default: 4; increase if GPU memory allows)",
    )
    parser.add_argument(
        "--evaluate", action="store_true",
        help="Run evaluation mode instead of interactive search",
    )
    parser.add_argument(
        "--queries_tsv", default="claude_figure_queries.tsv",
        help="Ground-truth TSV for --evaluate",
    )
    args = parser.parse_args()

    retriever = ColPaliRetriever(
        data_dirs   = args.data_dirs,
        cache_path  = args.cache,
        model_name  = args.model,
        batch_size  = args.batch_size,
        pool_factor = args.pool_factor,
    )

    if args.evaluate:
        evaluate(retriever, args.queries_tsv, top_k=args.top_k)
        return

    print(f"\nReady — {len(retriever._records):,} figures indexed.  "
          f"Type 'quit' to exit.\n")

    while True:
        query = input("Query: ").strip()
        if not query:
            continue
        if query.lower() in ("quit", "exit", "q"):
            break

        results = retriever.search(query, top_k=args.top_k)
        if not results:
            print("  No results.\n")
            continue

        print(f"\nTop {len(results)} result(s) for: '{query}'\n")
        for i, r in enumerate(results, 1):
            print(f"  {i}. score={r.score:.1f}  [{r.image_id}]")
            print(f"     Paper  : {r.title or r.paper_id}  ({r.paper_id})")
            cap = r.caption[:120] + ("..." if len(r.caption) > 120 else "")
            print(f"     Caption: {cap}")
            print(f"     Path   : {r.file_path}")
            print()


if __name__ == "__main__":
    main()