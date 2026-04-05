"""검색 모드 비교 스크립트.

Usage:
    uv run python -m rag_playground.application.compare
    uv run python -m rag_playground.application.compare --query "부산진구 맛집"
    uv run python -m rag_playground.application.compare --top-k 3
"""

from __future__ import annotations

import argparse
import logging
import time

from rag_playground.adapters.reranker.novita import rerank_hits
from rag_playground.adapters.vectorstore.qdrant import (
    COLLECTION_NAME,
    HYBRID_COLLECTION_NAME,
    get_qdrant_client,
    search,
    search_bm25,
    search_hybrid,
)

DEFAULT_QUERIES = [
    "부산진구 한식 맛집",
    "해운대구에서 양식 할인",
    "미용실 저렴한 곳",
    "동래구 목욕탕",
    "아이스크림 가게",
]

MODE_SEARCH = {
    "naive": lambda q, c: search(q, collection_name=c),
    "bm25": lambda q, c: search_bm25(q, collection_name=c),
    "hybrid": lambda q, c: search_hybrid(q, collection_name=c),
    "rerank": lambda q, c: rerank_hits(q, search_hybrid(q, collection_name=c, n_results=20), top_n=5),
}

MODE_COLLECTION = {
    "naive": COLLECTION_NAME,
    "bm25": HYBRID_COLLECTION_NAME,
    "hybrid": HYBRID_COLLECTION_NAME,
    "rerank": HYBRID_COLLECTION_NAME,
}


def get_doc_count(collection_name: str) -> int:
    """컬렉션 문서 수를 반환한다."""
    client = get_qdrant_client()
    info = client.get_collection(collection_name=collection_name)
    return info.points_count or 0


def print_result(mode: str, hits: list[dict], elapsed: float) -> None:
    """단일 모드의 검색 결과를 출력한다."""
    print(f"  [{mode}] ({elapsed*1000:.0f}ms)")
    if not hits:
        print("    (검색 결과 없음)")
        return
    for idx, hit in enumerate(hits, start=1):
        meta = hit["metadata"]
        if mode == "rerank":
            score = hit.get("relevance_score", 0)
            label = "relevance"
        else:
            score = hit.get("score", 1 - hit.get("distance", 0))
            label = "score"
        print(f"    {idx}. {meta.get('shop_name', '?')} ({meta.get('district', '?')}) "
              f"— {meta.get('benefit', '')}  [{label}: {score:.4f}]")
    print()


def run_comparison(queries: list[str], n_results: int = 5) -> None:
    """여러 쿼리에 대해 세 모드의 결과를 비교한다."""
    # 컬렉션 상태 확인
    naive_count = get_doc_count(COLLECTION_NAME)
    hybrid_count = get_doc_count(HYBRID_COLLECTION_NAME)

    print("\n📊 컬렉션 상태:")
    print(f"  Naive (family_card_shops):        {naive_count}건")
    print(f"  Hybrid (family_card_shops_hybrid): {hybrid_count}건")

    if naive_count == 0:
        print("\n⚠️  Naive 컬렉션이 비어있습니다: uv run python -m rag_playground.application.index")
        return
    if hybrid_count == 0:
        print("\n⚠️  Hybrid 컬렉션이 비어있습니다: uv run python -m rag_playground.application.index --mode hybrid")
        return

    print(f"\n{'=' * 60}")

    for query in queries:
        print(f"\n🔍 질의: \"{query}\"")
        print("-" * 40)

        for mode in ("naive", "bm25", "hybrid", "rerank"):
            collection = MODE_COLLECTION[mode]
            start = time.time()
            hits = MODE_SEARCH[mode](query, collection)
            elapsed = time.time() - start
            print_result(mode, hits, elapsed)

        print(f"{'=' * 60}")


def main() -> None:
    """비교 CLI 엔트리포인트."""
    parser = argparse.ArgumentParser(description="검색 모드 비교")
    parser.add_argument("--query", type=str, help="비교할 질의 (미지정 시 기본 질의 사용)")
    parser.add_argument("--top-k", type=int, default=5, help="검색 결과 수 (기본: 5)")
    args = parser.parse_args()

    logging.basicConfig(level=logging.WARNING)

    queries = [args.query] if args.query else DEFAULT_QUERIES
    run_comparison(queries, n_results=args.top_k)


if __name__ == "__main__":
    main()
