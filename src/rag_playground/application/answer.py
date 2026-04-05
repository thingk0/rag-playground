"""질의응답 유스케이스."""

from __future__ import annotations

from typing import Any

from rag_playground.adapters.llm.openai_chat import generate_answer
from rag_playground.adapters.reranker.novita import rerank_hits
from rag_playground.adapters.vectorstore.qdrant import (
    get_or_create_collection,
    get_or_create_hybrid_collection,
    search,
    search_bm25,
    search_hybrid,
)


def load_collection() -> str:
    """Naive RAG 컬렉션을 로드한다."""
    return get_or_create_collection()


def load_hybrid_collection() -> str:
    """하이브리드 검색 컬렉션을 로드한다."""
    return get_or_create_hybrid_collection()


def answer_query(
    query: str,
    collection_name: str,
    n_results: int = 5,
) -> tuple[list[dict[str, Any]], str]:
    """벡터 검색과 LLM 답변 생성을 실행한다."""
    hits = search(query, collection_name=collection_name, n_results=n_results)
    answer = generate_answer(query, hits)
    return hits, answer


def answer_query_bm25(
    query: str,
    collection_name: str,
    n_results: int = 5,
) -> tuple[list[dict[str, Any]], str]:
    """BM25 키워드 검색과 LLM 답변 생성을 실행한다."""
    hits = search_bm25(query, collection_name=collection_name, n_results=n_results)
    answer = generate_answer(query, hits)
    return hits, answer


def answer_query_hybrid(
    query: str,
    collection_name: str,
    n_results: int = 5,
) -> tuple[list[dict[str, Any]], str]:
    """하이브리드 검색(Dense+BM25 RRF)과 LLM 답변 생성을 실행한다."""
    hits = search_hybrid(query, collection_name=collection_name, n_results=n_results)
    answer = generate_answer(query, hits)
    return hits, answer


def answer_query_rerank(
    query: str,
    collection_name: str,
    n_results: int = 5,
    fetch_multiplier: int = 4,
) -> tuple[list[dict[str, Any]], str]:
    """Hybrid 검색 → Re-rank 2단계 파이프라인."""
    fetch_n = n_results * fetch_multiplier
    hits = search_hybrid(query, collection_name=collection_name, n_results=fetch_n)
    reranked = rerank_hits(query, hits, top_n=n_results)
    answer = generate_answer(query, reranked)
    return reranked, answer
