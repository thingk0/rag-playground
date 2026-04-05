"""Qdrant 벡터스토어 어댑터."""

from __future__ import annotations

import logging
from typing import Any

from openai import OpenAI
from qdrant_client import QdrantClient
from qdrant_client.models import (
    Distance,
    Document,
    Fusion,
    FusionQuery,
    Modifier,
    PointStruct,
    Prefetch,
    SparseVectorParams,
    VectorParams,
)

from rag_playground.config import OPENAI_API_KEY, QDRANT_API_KEY, QDRANT_URL
from rag_playground.domain.document import Document as DomainDocument

logger = logging.getLogger(__name__)

EMBEDDING_MODEL = "text-embedding-3-small"
EMBEDDING_DIMENSION = 1536
COLLECTION_NAME = "family_card_shops"
HYBRID_COLLECTION_NAME = "family_card_shops_hybrid"
DENSE_VECTOR_NAME = "dense"
SPARSE_VECTOR_NAME = "sparse"
BM25_OPTIONS = {"tokenizer": "multilingual", "language": "none"}


def get_openai_client() -> OpenAI:
    """OpenAI 클라이언트를 반환한다."""
    if not OPENAI_API_KEY:
        raise ValueError("OPENAI_API_KEY가 설정되지 않았습니다. .env 파일을 확인해주세요.")
    return OpenAI(api_key=OPENAI_API_KEY)


def embed_texts(texts: list[str]) -> list[list[float]]:
    """텍스트 리스트를 임베딩 벡터로 변환한다."""
    client = get_openai_client()
    response = client.embeddings.create(model=EMBEDDING_MODEL, input=texts)
    return [item.embedding for item in response.data]


def get_qdrant_client() -> QdrantClient:
    """Qdrant Cloud 클라이언트를 반환한다."""
    if not QDRANT_URL:
        raise ValueError("QDRANT_URL이 설정되지 않았습니다. .env 파일을 확인해주세요.")
    if not QDRANT_API_KEY:
        raise ValueError("QDRANT_API_KEY가 설정되지 않았습니다. .env 파일을 확인해주세요.")
    return QdrantClient(url=QDRANT_URL, api_key=QDRANT_API_KEY, timeout=60, cloud_inference=True)


def get_or_create_collection(
    client: QdrantClient | None = None,
    collection_name: str = COLLECTION_NAME,
) -> str:
    """컬렉션을 생성하거나 기존 컬렉션 이름을 반환한다."""
    if client is None:
        client = get_qdrant_client()

    existing = [c.name for c in client.get_collections().collections]
    if collection_name not in existing:
        client.create_collection(
            collection_name=collection_name,
            vectors_config=VectorParams(size=EMBEDDING_DIMENSION, distance=Distance.COSINE),
        )
        logger.info("Qdrant 컬렉션 '%s' 생성 완료", collection_name)
    else:
        logger.info("Qdrant 컬렉션 '%s' 이미 존재", collection_name)

    return collection_name


def get_or_create_hybrid_collection(
    client: QdrantClient | None = None,
    collection_name: str = HYBRID_COLLECTION_NAME,
) -> str:
    """Dense + Sparse(BM25) named vectors를 갖춘 하이브리드 컬렉션을 생성한다."""
    if client is None:
        client = get_qdrant_client()

    existing = [c.name for c in client.get_collections().collections]
    if collection_name not in existing:
        client.create_collection(
            collection_name=collection_name,
            vectors_config={
                DENSE_VECTOR_NAME: VectorParams(size=EMBEDDING_DIMENSION, distance=Distance.COSINE),
            },
            sparse_vectors_config={
                SPARSE_VECTOR_NAME: SparseVectorParams(modifier=Modifier.IDF),
            },
        )
        logger.info("하이브리드 Qdrant 컬렉션 '%s' 생성 완료", collection_name)
    else:
        logger.info("하이브리드 Qdrant 컬렉션 '%s' 이미 존재", collection_name)

    return collection_name


def index_documents(
    documents: list[DomainDocument],
    collection_name: str = COLLECTION_NAME,
    client: QdrantClient | None = None,
    batch_size: int = 100,
) -> int:
    """문서를 배치 단위로 Qdrant에 저장한다."""
    if client is None:
        client = get_qdrant_client()

    total = len(documents)
    for start in range(0, total, batch_size):
        batch = documents[start : start + batch_size]
        texts = [doc.page_content for doc in batch]
        vectors = embed_texts(texts)

        points = [
            PointStruct(
                id=start + idx,
                vector=vectors[idx],
                payload={"page_content": texts[idx], **batch[idx].metadata},
            )
            for idx in range(len(batch))
        ]

        client.upsert(collection_name=collection_name, points=points)
        logger.info("  인덱싱 진행: %s/%s", min(start + batch_size, total), total)

    return total


def index_documents_hybrid(
    documents: list[DomainDocument],
    collection_name: str = HYBRID_COLLECTION_NAME,
    client: QdrantClient | None = None,
    batch_size: int = 100,
) -> int:
    """문서를 Dense + Sparse(BM25) 벡터로 Qdrant에 저장한다."""
    if client is None:
        client = get_qdrant_client()

    total = len(documents)
    for start in range(0, total, batch_size):
        batch = documents[start : start + batch_size]
        texts = [doc.page_content for doc in batch]
        dense_vectors = embed_texts(texts)

        points = [
            PointStruct(
                id=start + idx,
                vector={
                    DENSE_VECTOR_NAME: dense_vectors[idx],
                    SPARSE_VECTOR_NAME: Document(
                        text=texts[idx],
                        model="qdrant/bm25",
                        options=BM25_OPTIONS,
                    ),
                },
                payload={"page_content": texts[idx], **batch[idx].metadata},
            )
            for idx in range(len(batch))
        ]

        client.upsert(collection_name=collection_name, points=points)
        logger.info("  하이브리드 인덱싱 진행: %s/%s", min(start + batch_size, total), total)

    return total


def search(
    query: str,
    collection_name: str = COLLECTION_NAME,
    client: QdrantClient | None = None,
    n_results: int = 5,
) -> list[dict[str, Any]]:
    """유사도 검색을 수행한다."""
    if client is None:
        client = get_qdrant_client()

    query_vector = embed_texts([query])[0]
    results = client.query_points(
        collection_name=collection_name,
        query=query_vector,
        limit=n_results,
        with_payload=True,
    )

    hits: list[dict[str, Any]] = []
    for point in results.points:
        payload = dict(point.payload or {})
        page_content = payload.pop("page_content", "")
        hits.append({
            "document": page_content,
            "metadata": payload,
            "distance": 1 - point.score,  # cosine similarity → distance
        })

    return hits


def search_bm25(
    query: str,
    collection_name: str = HYBRID_COLLECTION_NAME,
    client: QdrantClient | None = None,
    n_results: int = 5,
) -> list[dict[str, Any]]:
    """BM25 키워드 검색을 수행한다."""
    if client is None:
        client = get_qdrant_client()

    results = client.query_points(
        collection_name=collection_name,
        query=Document(text=query, model="qdrant/bm25", options=BM25_OPTIONS),
        using=SPARSE_VECTOR_NAME,
        limit=n_results,
        with_payload=True,
    )

    hits: list[dict[str, Any]] = []
    for point in results.points:
        payload = dict(point.payload or {})
        page_content = payload.pop("page_content", "")
        hits.append({
            "document": page_content,
            "metadata": payload,
            "score": point.score,
        })

    return hits


def search_hybrid(
    query: str,
    collection_name: str = HYBRID_COLLECTION_NAME,
    client: QdrantClient | None = None,
    n_results: int = 5,
    prefetch_limit: int = 20,
) -> list[dict[str, Any]]:
    """Dense + BM25 하이브리드 검색을 RRF로 결합한다."""
    if client is None:
        client = get_qdrant_client()

    query_vector = embed_texts([query])[0]
    results = client.query_points(
        collection_name=collection_name,
        prefetch=[
            Prefetch(
                query=query_vector,
                using=DENSE_VECTOR_NAME,
                limit=prefetch_limit,
            ),
            Prefetch(
                query=Document(text=query, model="qdrant/bm25", options=BM25_OPTIONS),
                using=SPARSE_VECTOR_NAME,
                limit=prefetch_limit,
            ),
        ],
        query=FusionQuery(fusion=Fusion.RRF),
        limit=n_results,
        with_payload=True,
    )

    hits: list[dict[str, Any]] = []
    for point in results.points:
        payload = dict(point.payload or {})
        page_content = payload.pop("page_content", "")
        hits.append({
            "document": page_content,
            "metadata": payload,
            "score": point.score,
        })

    return hits
