"""Take a question, choose where to search, and collect what was found."""

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path
from time import perf_counter
from typing import Protocol

from router_src.data import MODALITIES, ROUTES
from router_src.inference import ModernBERTRouter, validate_query
from rewrite_src.rewrite import GemmaRewriter, validate_settings

DEFAULT_CONFIG = Path(__file__).with_name("settings.json")
RESULT_STATUSES = ("complete", "partial", "error", "unavailable")


def load_config(path=DEFAULT_CONFIG):
    """Read the settings and find the files they point to."""
    path = Path(path).resolve()
    config = json.loads(path.read_text(encoding="utf-8"))
    required = {
        "router_checkpoint", "stage1_index", "stage2_checkpoint", "stage1_rerank",
        "stage2_max_length", "table_top_k", "cell_top_k", "table_source_dir", "device",
    }
    if not isinstance(config, dict):
        raise ValueError("Runtime configuration must be a JSON object")
    if required - config.keys() or config.keys() - required - {"rewriter"}:
        raise ValueError(f"Configuration requires {sorted(required)} and optional rewriter")
    if type(config["stage1_rerank"]) is not bool:
        raise ValueError("stage1_rerank must be boolean")
    for key in ("stage2_max_length", "table_top_k", "cell_top_k"):
        if type(config[key]) is not int or config[key] < 1:
            raise ValueError(f"{key} must be a positive integer")
    if config["stage1_rerank"] and config["table_top_k"] > 10:
        raise ValueError("Reranking selects from ten candidate tables")
    if not isinstance(config["device"], str) or not config["device"].strip():
        raise ValueError("device must be a nonempty string")

    for key in ("router_checkpoint", "stage1_index", "stage2_checkpoint", "table_source_dir"):
        value = config[key]
        if key == "table_source_dir" and value is None:
            continue
        if not isinstance(value, str) or not value.strip():
            raise ValueError(f"{key} must be a nonempty artifact path")
        config[key] = str((path.parent / Path(value).expanduser()).resolve())
    if "rewriter" in config:
        config["rewriter"] = validate_settings(config["rewriter"], path.parent)
    return config


def validate_result(result):
    """Check that a search system returned the fields we need."""
    if not isinstance(result, dict) or result.get("status") not in RESULT_STATUSES:
        raise ValueError("Retriever returned an invalid evidence envelope")
    fields = {"evidence": list, "provenance": dict, "timings": dict, "errors": list}
    for field, expected_type in fields.items():
        if not isinstance(result.get(field), expected_type):
            raise ValueError(f"Retriever returned an invalid {field} field")


def error_result(message, error_type, status="error"):
    """Keep a record of what went wrong so it shows up in the results."""
    return {
        "status": status, "evidence": [], "provenance": {}, "timings": {},
        "errors": [{"error": message, "error_type": error_type}],
    }


class ModalityRetriever(Protocol):
    """Describe what each search system needs to provide."""

    def search(self, query: str, *, query_id: str | None = None) -> dict:
        """Find information for this question."""
        ...


class TableRetriever:
    """Use table search and keep the cells and their source details."""

    def __init__(self, pipeline, table_top_k, cell_top_k):
        """Keep the table search system and how many tables and cells to return."""
        self.pipeline = pipeline
        self.table_top_k = table_top_k
        self.cell_top_k = cell_top_k

    def search(self, query, *, query_id=None, stage1_query=None):
        """Find matching tables and cells, along with their paper details."""
        options = {}
        if stage1_query is not None:
            options["stage1_query"] = stage1_query
        native = self.pipeline.search(
            query, self.table_top_k, cell_top_k=self.cell_top_k,
            query_id=query_id, **options,
        )
        contexts = {}
        for table in native["tables"]:
            table_id = table["table_id"]
            source = self.pipeline.retrieval.corpus.get(table_id) or {}
            contexts[table_id] = {
                key: source.get(key)
                for key in ("paper_id", "paper_title", "caption", "sub_caption", "footnotes", "units")
            }
        return {
            "status": native["status"], "evidence": native["cells"],
            "table_context": contexts, "native": native,
            "provenance": native["configuration"], "timings": native["timings"],
            "errors": native["errors"],
        }


class Orchestrator:
    """Choose where to search, rewrite if needed, and collect the results."""

    def __init__(self, router, retrievers: dict[str, ModalityRetriever], *, rewriter=None):
        """Keep the router, search systems and optional query rewriter together."""
        if set(retrievers) - set(MODALITIES):
            raise ValueError("Unknown retriever modality")
        self.router = router
        self.retrievers = dict(retrievers)
        self.rewriter = rewriter

    @classmethod
    def from_config(cls, config, *, retrievers=None):
        """Load the router and table search system using the saved settings."""
        from table_retrieval.live.stage1.artifacts import resolve_device
        from table_retrieval.pipeline import EvidencePipeline

        device = resolve_device(config["device"])
        router = ModernBERTRouter(config["router_checkpoint"], device)
        pipeline = EvidencePipeline.from_artifacts(
            config["stage1_index"], config["stage2_checkpoint"], device,
            stage2_max_length=config["stage2_max_length"],
            table_source_dir=config["table_source_dir"],
            stage1_rerank=config["stage1_rerank"],
        )
        adapters = dict(retrievers or {})
        if "table" in adapters:
            raise ValueError("The configured table adapter is already registered")
        adapters["table"] = TableRetriever(pipeline, config["table_top_k"], config["cell_top_k"])
        rewriter = None
        if "rewriter" in config:
            rewriter = GemmaRewriter(config["rewriter"])
        return cls(router, adapters, rewriter=rewriter)

    def search(self, query, *, query_id=None):
        """Run the question through the system and return what each search found."""
        validate_query(query)
        started = perf_counter()
        decision = self.router.route(query)
        routed = perf_counter()
        route = decision["route"]
        if route not in ROUTES:
            raise ValueError("Router returned an invalid route")

        results, timings, reformulations = {}, {}, {}
        for modality in route:
            before = perf_counter()
            adapter = self.retrievers.get(modality)
            if adapter is None:
                result = error_result(
                    "Retriever is not registered", "UnavailableRetriever", status="unavailable"
                )
            else:
                try:
                    options = {}
                    if modality == "table" and self.rewriter is not None:
                        rewrite = self.rewriter.rewrite(query, route)
                        reformulations["table"] = rewrite
                        options["stage1_query"] = rewrite["query"]
                    result = adapter.search(query, query_id=query_id, **options)
                    validate_result(result)
                except Exception as error:
                    result = error_result(str(error), type(error).__name__)
            results[modality] = result
            timings[modality + "_seconds"] = perf_counter() - before

        statuses = [result["status"] for result in results.values()]
        if all(status == "complete" for status in statuses):
            status = "complete"
        elif any(status in ("complete", "partial") for status in statuses):
            status = "partial"
        else:
            status = "error"
        output = {
            "schema_version": "1.0", "query": query, "query_id": query_id,
            "created_at_utc": datetime.now(timezone.utc).isoformat(),
            "routing": decision, "status": status, "modalities": results,
            "timings": {
                "routing_seconds": routed - started, **timings,
                "total_seconds": perf_counter() - started,
            },
        }
        if self.rewriter is not None:
            output["reformulation"] = reformulations
        return output


def main(argv=None):
    """Read a question from the command line and save the results."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--query", required=True)
    parser.add_argument("--query-id")
    parser.add_argument("--output", type=Path, required=True, help="New JSON output file")
    args = parser.parse_args(argv)
    validate_query(args.query)
    if args.output.exists():
        parser.error("Output exists; choose a fresh path")
    engine = Orchestrator.from_config(load_config(args.config))
    result = engine.search(args.query, query_id=args.query_id)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("x", encoding="utf-8") as stream:
        json.dump(result, stream, indent=2, ensure_ascii=False, allow_nan=False)
        stream.write("\n")
    print(f"{result['status']}: {result['routing']['label']} → {args.output}")
    rewrite = result.get("reformulation", {}).get("table")
    if rewrite and rewrite["status"] == "fallback":
        print(f"Gemma used the original question: {rewrite['error']}")


if __name__ == "__main__":
    main()
