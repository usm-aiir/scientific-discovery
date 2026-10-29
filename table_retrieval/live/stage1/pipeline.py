#Loads the completed Stage 1 index and returns fused Top-5 table candidates.
import logging
from pathlib import Path

from table_retrieval.data import Candidate, digest, load_json
from .artifacts import resolve_device, load_search_corpus, load_search_backend
from .retrieval import BACKENDS, DEFAULT_TABLE_TOP_K, search_query


logger = logging.getLogger(__name__)

# Handles Stage 1 BM25 + BGE retrieval and fusion.
class RetrievalPipeline:
    # Stores the retriever, loaded resources, corpus, and index information.
    def __init__(self, retriever, resources, corpus, provenance=None, *, reranker=None):
        if retriever not in BACKENDS:
            raise ValueError(f'Unknown retriever: {retriever}')
        self.retriever = retriever
        self.resources = resources
        self.corpus = corpus
        self.reranker = reranker
        self.provenance = dict(provenance or {})
        if reranker is not None:
            if retriever != 'fusion':
                raise ValueError('Table reranking requires fusion retrieval.')
            self.provenance['table_reranker'] = dict(reranker.provenance)
    # Loads the completed Stage 1 index and prepares BM25 + BGE for searching.
    @classmethod
    def from_index(cls, index_path, device_name='auto', *, rerank=False):
        if type(rerank) is not bool:
            raise ValueError('rerank must be a boolean.')
        index = Path(index_path).expanduser().resolve()
        manifest_path = index / 'prepared.json'
        if not index.is_dir():
            raise FileNotFoundError(f'Stage 1 index directory not found: {index}')
        if not manifest_path.is_file():
            raise FileNotFoundError(f'Stage 1 index is incomplete: missing {manifest_path}')
        device = resolve_device(device_name)
        corpus = load_search_corpus(index)
        resources = {
            name: load_search_backend(name, index, device)
            for name in BACKENDS['fusion']
        }
        manifest = load_json(manifest_path)
        provenance = {
            'index': str(index),
            'index_manifest_sha256': digest(manifest_path),
            'corpus_sha256': manifest.get('corpus_sha256'),
            'retrieval_device': device,
        }
        reranker = None
        if rerank:
            from .reranker import TableReranker
            reranker = TableReranker(device)
        return cls('fusion', resources, corpus, provenance, reranker=reranker)

    # Runs Stage 1 retrieval and returns the top ranked table candidates.
    def search(self, query, top_k=DEFAULT_TABLE_TOP_K):
        if not isinstance(query, str) or not query.strip():
            raise ValueError('Enter a nonempty query.')
        if type(top_k) is not int or top_k < 1:
            raise ValueError('top_k must be a positive integer.')
        depth = top_k
        if self.reranker is not None:
            from .reranker import CANDIDATE_COUNT
            if top_k > CANDIDATE_COUNT:
                raise ValueError(f'Reranked top_k cannot exceed {CANDIDATE_COUNT}.')
            depth = CANDIDATE_COUNT
        hits = search_query(query.strip(), self.retriever, self.resources, depth)
        if self.reranker is not None:
            hits = self.reranker.rank(query.strip(), hits, self.corpus)[:top_k]
        logger.debug('Stage 1 fusion returned %d tables.', len(hits))
        return [
            Candidate(uid, self.corpus.get(uid), rank, float(score))
            for rank, (uid, score) in enumerate(hits, 1)
        ]
