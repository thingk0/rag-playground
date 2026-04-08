"""질의응답 유스케이스."""

from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
from typing import Any

from rag_playground.adapters.llm.openai_chat import generate_answer
from rag_playground.adapters.query_rewriter.openai_rewriter import (
    generate_hypothetical_document,
    generate_multi_queries,
)
from rag_playground.adapters.reranker.novita import rerank_hits
from rag_playground.adapters.vectorstore.qdrant import (
    embed_texts,
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


def answer_query_hyde_rerank(
    query: str,
    collection_name: str,
    n_results: int = 5,
    fetch_multiplier: int = 4,
) -> tuple[list[dict[str, Any]], str]:
    """HyDE → Hybrid 검색 → Re-rank 3단계 파이프라인."""
    hypothetical_doc = generate_hypothetical_document(query)
    hyde_vector = embed_texts([hypothetical_doc])[0]
    fetch_n = n_results * fetch_multiplier
    hits = search_hybrid(
        query,
        collection_name=collection_name,
        n_results=fetch_n,
        dense_query_vector=hyde_vector,
    )
    reranked = rerank_hits(query, hits, top_n=n_results)
    answer = generate_answer(query, reranked)
    return reranked, answer


def answer_query_multi_rerank(
    query: str,
    collection_name: str,
    n_results: int = 5,
    fetch_multiplier: int = 4,
) -> tuple[list[dict[str, Any]], str]:
    """Multi-Query → Hybrid 검색 → Re-rank 3단계 파이프라인."""
    alt_queries = generate_multi_queries(query, n=3)
    fetch_n = n_results * fetch_multiplier
    all_queries = [query] + alt_queries
    with ThreadPoolExecutor(max_workers=len(all_queries)) as executor:
        futures = [
            executor.submit(search_hybrid, q, collection_name, n_results=fetch_n)
            for q in all_queries
        ]
        all_hits: list[dict[str, Any]] = []
        for future in futures:
            all_hits.extend(future.result())

    seen: set[str] = set()
    unique_hits: list[dict[str, Any]] = []
    for hit in all_hits:
        doc_text = hit["document"]
        if doc_text not in seen:
            seen.add(doc_text)
            unique_hits.append(hit)

    reranked = rerank_hits(query, unique_hits, top_n=n_results)
    answer = generate_answer(query, reranked)
    return reranked, answer
