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
from rag_playground.application.sources import get_source_config


def load_collection() -> str:
    """Naive RAG 컬렉션을 로드한다."""
    return get_or_create_collection()


def load_hybrid_collection() -> str:
    """하이브리드 검색 컬렉션을 로드한다."""
    return get_or_create_hybrid_collection()


def _dedupe_hits(hits: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """문서 텍스트 기준으로 검색 결과를 중복 제거한다."""
    seen: set[str] = set()
    unique_hits: list[dict[str, Any]] = []
    for hit in hits:
        doc_text = hit["document"]
        if doc_text not in seen:
            seen.add(doc_text)
            unique_hits.append(hit)
    return unique_hits


def retrieve_hits(
    query: str,
    collection_name: str,
    mode: str,
    n_results: int = 5,
    fetch_multiplier: int = 4,
    domain_context: str | None = None,
) -> list[dict[str, Any]]:
    """지정한 검색 모드로 hits만 반환한다."""
    if mode == "naive":
        return search(query, collection_name=collection_name, n_results=n_results)
    if mode == "bm25":
        return search_bm25(query, collection_name=collection_name, n_results=n_results)
    if mode == "hybrid":
        return search_hybrid(query, collection_name=collection_name, n_results=n_results)

    if domain_context is None:
        domain_context = "부산광역시 가족사랑카드 참여업체 데이터베이스"

    fetch_n = n_results * fetch_multiplier

    if mode == "rerank":
        hits = search_hybrid(query, collection_name=collection_name, n_results=fetch_n)
        return rerank_hits(query, hits, top_n=n_results)

    if mode == "hyde_rerank":
        hypothetical_doc = generate_hypothetical_document(query, domain_context=domain_context)
        hyde_vector = embed_texts([hypothetical_doc])[0]
        hits = search_hybrid(
            query,
            collection_name=collection_name,
            n_results=fetch_n,
            dense_query_vector=hyde_vector,
        )
        return rerank_hits(query, hits, top_n=n_results)

    if mode == "multi_rerank":
        alt_queries = generate_multi_queries(query, n=3, domain_context=domain_context)
        all_queries = [query] + alt_queries
        with ThreadPoolExecutor(max_workers=len(all_queries)) as executor:
            futures = [
                executor.submit(search_hybrid, q, collection_name, n_results=fetch_n)
                for q in all_queries
            ]
            all_hits: list[dict[str, Any]] = []
            for future in futures:
                all_hits.extend(future.result())

        unique_hits = _dedupe_hits(all_hits)
        return rerank_hits(query, unique_hits, top_n=n_results)

    raise ValueError(f"지원하지 않는 검색 모드입니다: {mode}")


def answer_query(
    query: str,
    collection_name: str,
    n_results: int = 5,
) -> tuple[list[dict[str, Any]], str]:
    """벡터 검색과 LLM 답변 생성을 실행한다."""
    hits = retrieve_hits(query, collection_name, mode="naive", n_results=n_results)
    answer = generate_answer(query, hits)
    return hits, answer


def answer_query_bm25(
    query: str,
    collection_name: str,
    n_results: int = 5,
) -> tuple[list[dict[str, Any]], str]:
    """BM25 키워드 검색과 LLM 답변 생성을 실행한다."""
    hits = retrieve_hits(query, collection_name, mode="bm25", n_results=n_results)
    answer = generate_answer(query, hits)
    return hits, answer


def answer_query_hybrid(
    query: str,
    collection_name: str,
    n_results: int = 5,
) -> tuple[list[dict[str, Any]], str]:
    """하이브리드 검색(Dense+BM25 RRF)과 LLM 답변 생성을 실행한다."""
    hits = retrieve_hits(query, collection_name, mode="hybrid", n_results=n_results)
    answer = generate_answer(query, hits)
    return hits, answer


def answer_query_rerank(
    query: str,
    collection_name: str,
    n_results: int = 5,
    fetch_multiplier: int = 4,
) -> tuple[list[dict[str, Any]], str]:
    """Hybrid 검색 → Re-rank 2단계 파이프라인."""
    reranked = retrieve_hits(
        query,
        collection_name,
        mode="rerank",
        n_results=n_results,
        fetch_multiplier=fetch_multiplier,
    )
    answer = generate_answer(query, reranked)
    return reranked, answer


def answer_query_hyde_rerank(
    query: str,
    collection_name: str,
    n_results: int = 5,
    fetch_multiplier: int = 4,
) -> tuple[list[dict[str, Any]], str]:
    """HyDE → Hybrid 검색 → Re-rank 3단계 파이프라인."""
    reranked = retrieve_hits(
        query,
        collection_name,
        mode="hyde_rerank",
        n_results=n_results,
        fetch_multiplier=fetch_multiplier,
    )
    answer = generate_answer(query, reranked)
    return reranked, answer


def answer_query_multi_rerank(
    query: str,
    collection_name: str,
    n_results: int = 5,
    fetch_multiplier: int = 4,
) -> tuple[list[dict[str, Any]], str]:
    """Multi-Query → Hybrid 검색 → Re-rank 3단계 파이프라인."""
    reranked = retrieve_hits(
        query,
        collection_name,
        mode="multi_rerank",
        n_results=n_results,
        fetch_multiplier=fetch_multiplier,
    )
    answer = generate_answer(query, reranked)
    return reranked, answer


def answer_query_for_source(
    query: str,
    source_id: str,
    mode: str,
    n_results: int = 5,
    fetch_multiplier: int = 4,
) -> tuple[list[dict[str, Any]], str]:
    """지정한 소스에 대해 검색 모드별 답변을 생성한다."""
    source = get_source_config(source_id)
    collection_name = (
        source.collection_name if mode == "naive" else source.hybrid_collection_name
    )
    hits = retrieve_hits(
        query,
        collection_name=collection_name,
        mode=mode,
        n_results=n_results,
        fetch_multiplier=fetch_multiplier,
        domain_context=source.domain_hint,
    )
    answer = generate_answer(query, hits)
    return hits, answer
