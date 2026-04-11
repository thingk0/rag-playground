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

from rag_playground.adapters.vectorstore.qdrant import (
    get_qdrant_client,
)
from rag_playground.application.agentic import run_agentic_query
from rag_playground.application.answer import retrieve_hits
from rag_playground.application.sources import get_source_config

DEFAULT_QUERIES = [
    "부산진구 한식 맛집",
    "해운대구에서 양식 할인",
    "미용실 저렴한 곳",
    "동래구 목욕탕",
    "아이스크림 가게",
    "이번 주말에 아이들이랑 갈 만한 데",
]


def get_doc_count(collection_name: str) -> int:
    """컬렉션 문서 수를 반환한다."""
    client = get_qdrant_client()
    try:
        info = client.get_collection(collection_name=collection_name)
    except Exception:
        return 0
    return info.points_count or 0


def print_result(mode: str, hits: list[dict], elapsed: float) -> None:
    """단일 모드의 검색 결과를 출력한다."""
    print(f"  [{mode}] ({elapsed*1000:.0f}ms)")
    if not hits:
        print("    (검색 결과 없음)")
        return
    for idx, hit in enumerate(hits, start=1):
        meta = hit["metadata"]
        if mode in ("rerank", "hyde_rerank", "multi_rerank", "agentic"):
            score = hit.get("relevance_score", 0)
            label = "relevance"
        else:
            score = hit.get("score", 1 - hit.get("distance", 0))
            label = "score"
        title = meta.get("title") or meta.get("shop_name") or meta.get("name") or "?"
        source_label = meta.get("source_label", "?")
        summary = meta.get("summary") or meta.get("benefit", "")
        print(
            f"    {idx}. {title} ({meta.get('district', '?')}) / {source_label} "
            f"— {summary}  [{label}: {score:.4f}]"
        )
    print()


def run_comparison(queries: list[str], n_results: int = 5) -> None:
    """여러 쿼리에 대해 검색 모드 결과를 비교한다."""
    family = get_source_config("family_card")
    library = get_source_config("library")
    family_naive_count = get_doc_count(family.collection_name)
    family_hybrid_count = get_doc_count(family.hybrid_collection_name)
    library_hybrid_count = get_doc_count(library.hybrid_collection_name)

    print("\n📊 컬렉션 상태:")
    print(f"  Family Naive ({family.collection_name}): {family_naive_count}건")
    print(f"  Family Hybrid ({family.hybrid_collection_name}): {family_hybrid_count}건")
    print(f"  Library Hybrid ({library.hybrid_collection_name}): {library_hybrid_count}건")

    if family_naive_count == 0:
        print("\n⚠️  Family naive 컬렉션이 비어있습니다: uv run python -m rag_playground.application.index")
        return
    if family_hybrid_count == 0 or library_hybrid_count == 0:
        print("\n⚠️  하이브리드 컬렉션이 비어있습니다:")
        print("   uv run python -m rag_playground.application.index --mode hybrid --source all")
        return

    print(f"\n{'=' * 60}")

    for query in queries:
        print(f"\n🔍 질의: \"{query}\"")
        print("-" * 40)

        for mode in ("naive", "bm25", "hybrid", "rerank", "hyde_rerank", "multi_rerank", "agentic"):
            start = time.time()
            if mode == "agentic":
                hits = run_agentic_query(query, n_results=n_results).hits
            else:
                collection = family.collection_name if mode == "naive" else family.hybrid_collection_name
                hits = retrieve_hits(
                    query,
                    collection_name=collection,
                    mode=mode,
                    n_results=n_results,
                    domain_context=family.domain_hint,
                )
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
