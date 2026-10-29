# Creates the Stage 2 BGE retrieval pipeline from a trained checkpoint.

from pathlib import Path

from table_retrieval.live.stage2.pipeline import BGECellRetrievalPipeline

from .model import BGECellSelector

# Loads the Stage 2 BGE checkpoint and builds the cell retrieval pipeline.
def create_pipeline(checkpoint, device='auto', *, max_length, table_source_dir=None):
    """Load the BGE checkpoint; source HTML is an optional enhancement.

    ``table_source_dir=None`` is normal and needs no HTML files. An explicitly
    configured directory must exist, but missing paper HTML uses corpus structure.
    """
    checkpoint = Path(checkpoint).expanduser().resolve()
    if not checkpoint.is_dir():
        raise FileNotFoundError(f'BGE checkpoint directory not found: {checkpoint}')
    if table_source_dir is not None:
        table_source_dir = Path(table_source_dir).expanduser().resolve()
        if not table_source_dir.is_dir():
            raise FileNotFoundError(f'Table source HTML directory not found: {table_source_dir}')
    selector = BGECellSelector(str(checkpoint), device, max_length=max_length)
    provenance = {
        'cell_model': str(checkpoint.resolve()),
        'cell_revision': selector.revision,
        'cell_device': selector.device,
        'table_source_dir': str(table_source_dir) if table_source_dir else None,
        'cell_max_length': selector.max_length,
    }
    return BGECellRetrievalPipeline(selector, provenance, table_source_dir=table_source_dir)
