"""Create the canonical pooled Stage 2 pipeline with a BGE cell selector."""

from pathlib import Path

from table_retrieval.stage2.pipeline import BGECellRetrievalPipeline
from table_retrieval.stage2.pipeline import load_layouts
from table_retrieval.data import digest

from .model import BGECellSelector


def create_pipeline(checkpoint, device='auto', layouts_path=None):
    """Load a required local BGE checkpoint for global cell ranking."""
    checkpoint = Path(checkpoint).expanduser().resolve()
    if not checkpoint.is_dir():
        raise FileNotFoundError(f'BGE checkpoint directory not found: {checkpoint}')
    layouts_path = Path(layouts_path).expanduser().resolve() if layouts_path is not None else None
    selector = BGECellSelector(str(checkpoint), device)
    layouts = load_layouts(layouts_path)
    provenance = {
        'cell_model': str(checkpoint.resolve()),
        'cell_revision': selector.revision,
        'cell_device': selector.device,
        'layouts_sha256': digest(layouts_path) if layouts_path else None,
    }
    return BGECellRetrievalPipeline(selector, layouts, provenance)
