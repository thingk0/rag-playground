"""구버전 vectorstore 경로 호환 래퍼."""

from rag_playground.adapters.vectorstore.qdrant import (
    COLLECTION_NAME,
    EMBEDDING_DIMENSION,
    EMBEDDING_MODEL,
    embed_texts,
    get_or_create_collection,
    get_qdrant_client,
    index_documents,
    search,
)

__all__ = [
    "EMBEDDING_MODEL",
    "EMBEDDING_DIMENSION",
    "COLLECTION_NAME",
    "embed_texts",
    "get_qdrant_client",
    "get_or_create_collection",
    "index_documents",
    "search",
]
