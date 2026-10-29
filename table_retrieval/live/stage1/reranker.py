"""Score the fused Top-10 tables with a pretrained BGE cross-encoder."""
from .artifacts import resolve_device
from .text import table_text


DEFAULT_MODEL = 'BAAI/bge-reranker-v2-m3'
DEFAULT_REVISION = '953dc6f6f85a1b2dbfca4c34a2796e7dde08d41e'
CANDIDATE_COUNT = 10
MAX_LENGTH = 2048


class TableReranker:
    def __init__(self, device='auto', *, model=DEFAULT_MODEL, batch_size=2):
        import torch
        from transformers import AutoModelForSequenceClassification, AutoTokenizer

        if type(batch_size) is not int or batch_size < 1:
            raise ValueError('Reranker batch_size must be a positive integer.')
        self.device = resolve_device(device)
        self.batch_size = batch_size
        revision = DEFAULT_REVISION if model == DEFAULT_MODEL else None
        self.tokenizer = AutoTokenizer.from_pretrained(model, revision=revision)
        self.model = AutoModelForSequenceClassification.from_pretrained(
            model, revision=revision, torch_dtype=torch.float32,
            attn_implementation='eager').to(self.device).eval()
        self.provenance = {
            'model': str(model), 'revision': revision, 'candidate_count': CANDIDATE_COUNT,
            'max_length': MAX_LENGTH, 'batch_size': batch_size, 'dtype': 'float32',
            'score': 'cross-encoder logit',
        }

    def rank(self, query, hits, corpus):
        """Return (table ID, logit), preserving fusion order when scores tie."""
        import torch

        if not isinstance(query, str) or not query.strip():
            raise ValueError('Enter a nonempty query.')
        hits = list(hits)[:CANDIDATE_COUNT]
        ids = [uid for uid, _ in hits]
        if len(ids) != len(set(ids)) or any(uid not in corpus for uid in ids):
            raise ValueError('Reranking requires unique candidate IDs present in the corpus.')
        scores = []
        with torch.inference_mode():
            for start in range(0, len(ids), self.batch_size):
                batch = ids[start:start + self.batch_size]
                inputs = self.tokenizer(
                    [query.strip()] * len(batch), [table_text(corpus[uid]) for uid in batch],
                    padding=True, truncation=True, max_length=MAX_LENGTH,
                    return_tensors='pt')
                logits = self.model(**{key: value.to(self.device) for key, value in inputs.items()})
                values = logits.logits.float().reshape(-1)
                if len(values) != len(batch) or not torch.isfinite(values).all():
                    raise ValueError('Expected one finite reranker score per table.')
                scores.extend(values.cpu().tolist())
        return sorted(zip(ids, scores), key=lambda hit: -hit[1])
