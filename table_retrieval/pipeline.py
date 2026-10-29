# Connects Stage 1 table retrieval with Stage 2 cell retrieval.
from datetime import datetime, timezone
from time import perf_counter

from .data import SCHEMA_VERSION
from .live.stage1.artifacts import resolve_device
from .live.stage1.pipeline import RetrievalPipeline
from .live.stage2.bge.pipeline import create_pipeline

# Runs the complete retrieval pipeline from query to final evidence cells.
class EvidencePipeline:

# Stores the loaded Stage 1 and Stage 2 retrieval pipelines.
    def __init__(self, retrieval, cell_retrieval):
        self.retrieval = retrieval
        self.cell_retrieval = cell_retrieval

# Loads the Stage 1 index and Stage 2 checkpoint and builds the full pipeline.
    @classmethod
    def from_artifacts(cls, stage1_index, checkpoint, device='auto', *, stage2_max_length,
                       table_source_dir=None, stage1_rerank=False):
        device = resolve_device(device)
        retrieval_options = {'rerank': True} if stage1_rerank else {}
        retrieval = RetrievalPipeline.from_index(stage1_index, device, **retrieval_options)
        options = {'table_source_dir': table_source_dir} if table_source_dir is not None else {}
        cells = create_pipeline(checkpoint, device, max_length=stage2_max_length, **options)
        return cls(retrieval, cells)

# Combines Stage 1 and Stage 2 settings used for the current retrieval run.
    def configuration(self, table_top_k, cell_top_k):
        return dict(
            self.retrieval.provenance,
            **self.cell_retrieval.provenance,
            retriever='fusion',
            table_top_k=table_top_k,
            cell_top_k=cell_top_k,
            cell_selection='global_top_k',
        )
# Runs Stage 1 table retrieval followed by global Stage 2 cell ranking.
    def search(self, query, table_top_k, *, cell_top_k, query_id=None):
        """Return Stage 1 tables and the global Stage 2 Top-k cells directly."""
        started = perf_counter()
        candidates = self.retrieval.search(query, table_top_k)
        tables_retrieved = perf_counter()
        evidence = self.cell_retrieval.rank(query, candidates, cell_top_k)
        finished = perf_counter()
        return dict(
            schema_version=SCHEMA_VERSION,
            query_id=str(query_id) if query_id is not None else None,
            query=query.strip(),
            created_at_utc=datetime.now(timezone.utc).isoformat(),
            configuration=self.configuration(table_top_k, cell_top_k),
            timings={
                'stage1_seconds': tables_retrieved - started,
                'stage2_seconds': finished - tables_retrieved,
                'total_seconds': finished - started,
            },
            status=evidence['status'],
            tables=evidence['tables'],
            cells=evidence['cells'],
            errors=evidence['errors'],
        )
