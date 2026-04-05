"""Novita.ai Reranker 어댑터 (BGE-reranker-v2-m3)."""

from __future__ import annotations

import logging
from typing import Any

import httpx

from rag_playground.config import NOVITA_API_KEY

logger = logging.getLogger(__name__)

RERANK_URL = "https://api.novita.ai/openai/v1/rerank"
RERANK_MODEL = "baai/bge-reranker-v2-m3"


def rerank(
    query: str,
    documents: list[str],
    top_n: int = 5,
) -> list[dict[str, Any]]:
    """문서 리스트를 Re-rank하여 관련도 순으로 정렬한다.

    Returns:
        [{"index": 원래 순서, "relevance_score": 점수}, ...]
    """
    if not NOVITA_API_KEY:
        raise ValueError("NOVITA_API_KEY가 설정되지 않았습니다. .env 파일을 확인해주세요.")
    if not documents:
        return []

    response = httpx.post(
        RERANK_URL,
        headers={
            "Content-Type": "application/json",
            "Authorization": f"Bearer {NOVITA_API_KEY}",
        },
        json={
            "model": RERANK_MODEL,
            "query": query,
            "documents": documents,
            "top_n": top_n,
        },
        timeout=30,
    )
    response.raise_for_status()

    data = response.json()
    results = data.get("results", [])
    logger.info("Re-rank 완료: %d건 → %d건", len(documents), len(results))

    return [
        {"index": r["index"], "relevance_score": r["relevance_score"]}
        for r in results
    ]


def rerank_hits(
    query: str,
    hits: list[dict[str, Any]],
    top_n: int = 5,
) -> list[dict[str, Any]]:
    """검색 결과(hit 리스트)를 Re-rank하여 재정렬한다.

    기존 검색 결과를 Re-ranker 점수로 재배열한다.
    """
    documents = [hit["document"] for hit in hits]
    ranked = rerank(query, documents, top_n=top_n)

    reranked_hits: list[dict[str, Any]] = []
    for item in ranked:
        original = hits[item["index"]]
        reranked_hits.append({
            "document": original["document"],
            "metadata": original["metadata"],
            "relevance_score": item["relevance_score"],
        })

    return reranked_hits
