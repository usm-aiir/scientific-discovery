#Loads and validates the resources needed for Stage 1 retrieval.
from pathlib import Path

from table_retrieval.data import digest, load_corpus, load_json

# Selects the device to use for Stage 1 retrieval.
def resolve_device(requested):
    """Resolve `auto` and reject unavailable or malformed device selections."""
    import torch
    if not isinstance(requested, str) or not requested.strip():
        raise ValueError('Stage 1 device must be a nonempty string or "auto".')
    if requested == 'auto':
        return 'cuda:1' if torch.cuda.device_count() >= 2 else 'cuda:0' if torch.cuda.is_available() else 'cpu'
    try:
        parsed = torch.device(requested)
    except (RuntimeError, TypeError) as error:
        raise ValueError(f'Invalid Stage 1 device: {requested!r}') from error
    if parsed.type == 'cuda':
        if not torch.cuda.is_available():
            raise RuntimeError(f'CUDA device requested but CUDA is unavailable: {requested}')
        if parsed.index is not None and parsed.index >= torch.cuda.device_count():
            raise RuntimeError(f'CUDA device does not exist: {requested}')
    return str(parsed)

#  Loads the corpus recorded in the selected Stage 1 index.
def load_search_corpus(index):
    """Load the exact corpus recorded by an index and verify its digest."""
    index = Path(index).expanduser().resolve()
    metadata = load_json(index / 'prepared.json')
    recorded = Path(metadata['corpus']).expanduser()
    if recorded.is_absolute() and recorded.is_file():
        corpus_path = recorded.resolve()
    else:
        relative = Path(recorded.name) if recorded.is_absolute() else recorded
        candidates = [index / relative]
        candidates.extend(parent / relative for parent in index.parents)
        if recorded.is_absolute():
            candidates.insert(0, index / recorded.name)
        corpus_path = next((path.resolve() for path in candidates if path.is_file()), None)
        if corpus_path is None:
            searched = ', '.join(str(path) for path in candidates)
            raise FileNotFoundError(
                f'Stage 1 corpus not found. Index records {recorded!s}; searched: {searched}')
    if metadata.get('corpus_sha256') and digest(corpus_path) != metadata['corpus_sha256']:
        raise ValueError(f'Corpus changed since indexing: {corpus_path}')
    return load_corpus(corpus_path)

# Loads the selected Stage 1 retriever from the configured index folder.
def load_search_backend(name, index, device='cpu'):
    index = Path(index)
    if name in ('bm25_full', 'bm25_matched'):
        from .retrievers.bm25 import load_lexical
        manifest = load_json(index / 'prepared.json')
        return load_lexical(index / f'{name}.npz', manifest), manifest
    if name == 'bge':
        from .retrievers.bge import load_search_index
        return load_search_index(index, device)
    raise ValueError(f'Unknown search backend: {name}')
