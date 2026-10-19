import logging
import warnings
from functools import lru_cache
from typing import Any, Dict, List, Optional, Union
from langchain_core.documents import Document

warnings.filterwarnings("ignore", category=DeprecationWarning)

try:
    from langchain_chroma import Chroma
except ImportError:
    from langchain_community.vectorstores import Chroma

from app.config import get_settings
from app.rag.ingest import get_embedding_function

logger = logging.getLogger("enterprise_rag.retriever")
settings = get_settings()


# =========================================================================
# 1. Vector Store Factory (Chroma & Qdrant Switching)
# =========================================================================

@lru_cache(maxsize=2)
def get_vector_store(store_type: Optional[str] = None):
    """
    Factory to instantiate and return the configured vector store (Chroma or Qdrant).
    Pass store_type to override settings.VECTOR_STORE_TYPE at runtime.
    """
    target = (store_type or settings.VECTOR_STORE_TYPE).lower()
    embeddings = get_embedding_function()

    if target == "qdrant":
        try:
            from qdrant_client import QdrantClient
            from langchain_qdrant import QdrantVectorStore as Qdrant
        except ImportError:
            raise ImportError(
                "Qdrant integration packages are missing. Install with: pip install langchain-qdrant qdrant-client"
            )

        client = QdrantClient(
            url=settings.QDRANT_URL,
            api_key=settings.QDRANT_API_KEY or None,
        )
        return Qdrant(
            client=client,
            collection_name=settings.COLLECTION_NAME,
            # pyrefly: ignore[unexpected-keyword]
            embeddings=embeddings,
        )

    # Default: ChromaDB
    return Chroma(
        persist_directory=settings.CHROMA_PERSIST_DIR,
        collection_name=settings.COLLECTION_NAME,
        embedding_function=embeddings,
    )


# =========================================================================
# 2. Hybrid Retriever (Dense Embeddings + BM25 Keywords + Metadata Filter)
# =========================================================================

class HybridRetriever:
    """
    Hybrid Search combining:
    1. Dense Vector Search (semantic similarity via Chroma or Qdrant)
    2. BM25 Sparse Search (exact keyword and acronym matching)
    3. Reciprocal Rank Fusion (RRF) to merge and rank results
    4. Metadata filtering (e.g., by source file name, page, or chunk_index)
    """

    def __init__(
        self,
        vector_store=None,
        alpha: float = 0.5,
        default_k: int = 8,
    ):
        """
        :param vector_store: Vector store instance (Chroma or Qdrant)
        :param alpha: Weight between dense (alpha) and sparse BM25 (1 - alpha)
        :param default_k: Default number of top documents to return
        """
        self.vector_store = vector_store or get_vector_store()
        self.alpha = alpha
        self.default_k = default_k

    def _get_all_stored_documents(self) -> List[Document]:
        """Fetch all documents from the vector store for BM25 indexing (cached in RAM)."""
        if hasattr(self, "_cached_corpus") and self._cached_corpus:
            return self._cached_corpus
        try:
            if hasattr(self.vector_store, "get"):
                # ChromaDB supports .get()
                data = self.vector_store.get()
                docs = []
                contents = data.get("documents", [])
                metadatas = data.get("metadatas", [])
                for i in range(len(contents)):
                    meta = metadatas[i] if i < len(metadatas) and metadatas[i] else {}
                    docs.append(Document(page_content=contents[i], metadata=meta))
                self._cached_corpus = docs
                return self._cached_corpus
        except Exception as e:
            logger.debug("Could not extract full corpus for BM25: %s", e)
        return []

    def _dense_search(
        self,
        query: str,
        k: int,
        filter_dict: Optional[Dict[str, Any]] = None,
    ) -> List[Document]:
        """Perform dense embedding similarity search with optional metadata filter and noise removal."""
        try:
            kwargs: Dict[str, Any] = {"k": k * 2}
            if filter_dict:
                kwargs["filter"] = filter_dict
            raw_docs = self.vector_store.similarity_search(query, **kwargs)
        except Exception as e:
            logger.warning("Dense search encountered an issue: %s. Retrying without filter.", e)
            raw_docs = self.vector_store.similarity_search(query, k=k * 2)

        # Filter out tiny running headers / page numbers (< 50 characters)
        meaningful_docs = [d for d in raw_docs if len(d.page_content.strip()) >= 50]
        return meaningful_docs[:k] if meaningful_docs else raw_docs[:k]

    def _bm25_search(
        self,
        query: str,
        corpus: List[Document],
        k: int,
        filter_dict: Optional[Dict[str, Any]] = None,
    ) -> List[Document]:
        """Perform BM25 keyword search across corpus with metadata filtering and noise removal."""
        if not corpus:
            return []

        # Filter out noise chunks from BM25 corpus as well
        cleaned_corpus = [d for d in corpus if len(d.page_content.strip()) >= 50] or corpus

        # Apply metadata filter to corpus if specified
        filtered_corpus = cleaned_corpus
        if filter_dict:
            filtered_corpus = [
                doc for doc in cleaned_corpus
                if all(doc.metadata.get(kf) == vf for kf, vf in filter_dict.items())
            ]
        if not filtered_corpus:
            return []

        try:
            from rank_bm25 import BM25Okapi

            tokenized_corpus = [doc.page_content.lower().split() for doc in filtered_corpus]
            bm25 = BM25Okapi(tokenized_corpus)
            tokenized_query = query.lower().split()

            scores = bm25.get_scores(tokenized_query)
            top_indices = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)[:k]

            return [filtered_corpus[i] for i in top_indices if scores[i] > 0]
        except Exception as e:
            logger.warning("BM25 search skipped: %s", e)
            return []

    def invoke(
        self,
        query: str,
        k: Optional[int] = None,
        filter_dict: Optional[Dict[str, Any]] = None,
    ) -> List[Document]:
        """
        Execute Hybrid Search:
        Combines Dense + BM25 results using Reciprocal Rank Fusion (RRF).
        """
        target_k = k or self.default_k

        # 1. Dense retrieval
        dense_docs = self._dense_search(query, k=target_k * 2, filter_dict=filter_dict)

        # 2. Sparse BM25 retrieval
        corpus = self._get_all_stored_documents()
        sparse_docs = self._bm25_search(query, corpus=corpus, k=target_k * 2, filter_dict=filter_dict)

        # If BM25 has no separate index, use dense results directly
        if not sparse_docs:
            ranked_docs = dense_docs[:target_k]
        else:
            # 3. Reciprocal Rank Fusion (RRF)
            # RRF Score = sum( 1 / (60 + rank) )
            rrf_constant = 60
            doc_scores: Dict[str, float] = {}
            doc_map: Dict[str, Document] = {}

            # Score dense matches
            for rank, doc in enumerate(dense_docs):
                doc_id = f"{doc.metadata.get('source', '')}_{doc.metadata.get('page', '')}_{doc.page_content[:40]}"
                score = self.alpha * (1.0 / (rrf_constant + rank + 1))
                doc_scores[doc_id] = doc_scores.get(doc_id, 0.0) + score
                doc_map[doc_id] = doc

            # Score sparse matches
            for rank, doc in enumerate(sparse_docs):
                doc_id = f"{doc.metadata.get('source', '')}_{doc.metadata.get('page', '')}_{doc.page_content[:40]}"
                score = (1.0 - self.alpha) * (1.0 / (rrf_constant + rank + 1))
                doc_scores[doc_id] = doc_scores.get(doc_id, 0.0) + score
                doc_map[doc_id] = doc

            # Sort by combined RRF score
            sorted_ids = sorted(doc_scores.keys(), key=lambda x: doc_scores[x], reverse=True)
            ranked_docs = [doc_map[did] for did in sorted_ids[:target_k]]

        # Standardize metadata for caller
        for idx, doc in enumerate(ranked_docs, start=1):
            if not doc.metadata:
                doc.metadata = {}
            if "source" not in doc.metadata:
                doc.metadata["source"] = doc.metadata.get("file_name", "Unknown Document")
            if "page" not in doc.metadata:
                doc.metadata["page"] = 1
            if "chunk_id" not in doc.metadata:
                chunk_idx = doc.metadata.get("chunk_index")
                doc.metadata["chunk_id"] = f"chunk_{chunk_idx + 1}" if chunk_idx is not None else f"chunk_{idx}"

        return ranked_docs


# =========================================================================
# 3. Factory function for Graph and External Consumers
# =========================================================================

def get_retriever(
    k: int = 4,
    store_type: Optional[str] = None,
    alpha: float = 0.5,
) -> HybridRetriever:
    """
    Return the unified HybridRetriever (supporting Chroma, Qdrant, BM25, and metadata filtering).
    """
    vector_store = get_vector_store(store_type=store_type)
    return HybridRetriever(
        vector_store=vector_store,
        alpha=alpha,
        default_k=k,
    )
