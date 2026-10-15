# Loads the Stage 2 BGE model and uses it to rank candidate cells.

from table_retrieval.stage2.ranking import rank_scored_cells


DEFAULT_MODEL = 'BAAI/bge-m3'
DEFAULT_REVISION = '5617a9f61b028005a4858fdac845db406aefb181'
DEFAULT_THRESHOLD = 0.5
DEFAULT_BATCH_SIZE = 32
MODEL_CACHE = None


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
        requested = ('cuda:0' if torch.cuda.is_available() else 'cpu') if device == 'auto' else device
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
        self.checkpoint = checkpoint
        self.batch_size = batch_size
        self.max_length = max_length
        revision = DEFAULT_REVISION if checkpoint == DEFAULT_MODEL else None
        options = dict(revision=revision, cache_dir=MODEL_CACHE)
        dtype = torch.float16 if self.device.startswith('cuda') else torch.float32
        self.tokenizer = AutoTokenizer.from_pretrained(checkpoint, **options)
        self.model = AutoModel.from_pretrained(
            checkpoint, torch_dtype=dtype, attn_implementation='eager', **options
        ).to(self.device).eval()
        self.revision = getattr(self.model.config, '_commit_hash', revision)

# Converts text into normalized BGE embedding vectors.
    def encode(self, texts):
        vectors = []
        with self.torch.inference_mode():
            for start in range(0, len(texts), self.batch_size):
                batch = texts[start:start + self.batch_size]
                inputs = self.tokenizer(
                    batch, padding=True, truncation=True, max_length=self.max_length,
                    return_tensors='pt')
                output = self.model(**{key: value.to(self.device)
                                      for key, value in inputs.items()})
                vector = self.torch.nn.functional.normalize(
                    output.last_hidden_state[:, 0].float(), dim=-1)
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
