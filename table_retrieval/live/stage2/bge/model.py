# Loads the Stage 2 BGE model and uses it to rank candidate cells.
from pathlib import Path

from table_retrieval.data import load_json
from table_retrieval.live.stage1.artifacts import resolve_device
from table_retrieval.live.stage2.ranking import rank_scored_cells


DEFAULT_MODEL = 'BAAI/bge-m3'
DEFAULT_REVISION = '5617a9f61b028005a4858fdac845db406aefb181'
DEFAULT_THRESHOLD = 0.5
DEFAULT_BATCH_SIZE = 32
MODEL_CACHE = None


def resolve_checkpoint(checkpoint):
    """Accept pretrained BGE-M3, a checkpoint, or a training run's selected epoch."""
    if checkpoint == DEFAULT_MODEL:
        return checkpoint
    checkpoint = Path(checkpoint).expanduser().resolve()
    if (checkpoint / 'training.json').is_file():
        checkpoint = checkpoint / load_json(checkpoint / 'training.json')['checkpoint']
    if not checkpoint.is_dir():
        raise FileNotFoundError(f'Checkpoint directory not found: {checkpoint}')
    return checkpoint


def encode(model, tokenizer, texts, device, max_length):
    """Shared normalized CLS encoding; gradients stay available during training."""
    import torch
    tokens = tokenizer(texts, padding=True, truncation=True, max_length=max_length, return_tensors='pt')
    output = model(**{key: value.to(device) for key, value in tokens.items()})
    return torch.nn.functional.normalize(output.last_hidden_state[:, 0].float(), dim=-1)


class BGECellSelector:

    #Load BGE-M3 once and rank cells by cosine similarity to a question.
    def __init__(self, checkpoint=DEFAULT_MODEL, device='auto', batch_size=DEFAULT_BATCH_SIZE,
                 *, max_length):
        if type(max_length) is not int or max_length < 1:
            raise ValueError('BGE max length must be a positive integer.')
        try:
            import torch
            from transformers import AutoModel, AutoTokenizer
        except ImportError as error:
            raise RuntimeError('Install torch and transformers to run BGE cell retrieval.') from error

        if batch_size < 1:
            raise ValueError('BGE batch size must be positive.')
        self.torch = torch
        if not isinstance(device, str) or not device.strip():
            raise ValueError('BGE device must be a nonempty string or "auto".')
        requested = resolve_device(device) if device == 'auto' else device
        try:
            parsed_device = torch.device(requested)
        except (RuntimeError, TypeError) as error:
            raise ValueError(f'Invalid BGE device: {device!r}') from error
        if parsed_device.type == 'cuda':
            if not torch.cuda.is_available():
                raise RuntimeError(f'CUDA device requested but CUDA is unavailable: {requested}')
            if parsed_device.index is not None and parsed_device.index >= torch.cuda.device_count():
                raise RuntimeError(f'CUDA device does not exist: {requested}')
        self.device = str(parsed_device)
        checkpoint = str(resolve_checkpoint(checkpoint))
        self.checkpoint = checkpoint
        self.batch_size = batch_size
        self.max_length = max_length
        revision = DEFAULT_REVISION if checkpoint == DEFAULT_MODEL else None
        options = dict(revision=revision, cache_dir=MODEL_CACHE)
        if self.device.startswith('cuda'):
            torch.backends.cuda.matmul.allow_tf32 = False
            torch.backends.cudnn.allow_tf32 = False
        self.tokenizer = AutoTokenizer.from_pretrained(checkpoint, **options)
        self.model = AutoModel.from_pretrained(
            checkpoint, torch_dtype=torch.float32, attn_implementation='eager', **options
        ).to(self.device).eval()
        self.revision = getattr(self.model.config, '_commit_hash', revision)

# Converts text into normalized BGE embedding vectors.
    def encode(self, texts):
        vectors = []
        with self.torch.inference_mode():
            for start in range(0, len(texts), self.batch_size):
                batch = texts[start:start + self.batch_size]
                vector = encode(self.model, self.tokenizer, batch, self.device, self.max_length)
                vectors.append(vector.cpu())
        result = self.torch.cat(vectors)
        if not self.torch.isfinite(result).all():
            raise ValueError('BGE produced non-finite cell embeddings.')
        return result

# Compares the query embedding with all prepared cell embeddings and ranks the cells.
    def rank_prepared_cells(self, query, cells):
        """Rank a prepared cross-table cell pool with one query encoding."""
        if not isinstance(query, str) or not query.strip():
            raise ValueError('Enter a nonempty question.')
        if not cells:
            return []
        if any(not isinstance(cell, dict) or not isinstance(cell.get('text'), str)
               for cell in cells):
            raise ValueError('Prepared cells must contain text representations.')

        query_vector = self.encode([query.strip()])[0]
        cell_vectors = self.encode([cell['text'] for cell in cells])
        scores = cell_vectors @ query_vector
        return rank_scored_cells(cells, scores)
