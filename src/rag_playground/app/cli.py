"""RAG 대화형 CLI.

Usage:
    uv run python -m rag_playground.app.cli
"""

from __future__ import annotations

import logging
from typing import Any

from rag_playground.adapters.vectorstore.qdrant import get_qdrant_client
from rag_playground.application.answer import (
    answer_query,
    answer_query_bm25,
    answer_query_hybrid,
    answer_query_rerank,
    load_collection,
    load_hybrid_collection,
)

SEARCH_MODES = {
    "1": ("naive", "Naive RAG (벡터 검색)"),
    "2": ("bm25", "BM25 (키워드 검색)"),
    "3": ("hybrid", "Hybrid (벡터 + 키워드 RRF)"),
    "4": ("rerank", "Hybrid + Re-rank (BGE-reranker-v2-m3)"),
}


def get_document_count(collection_name: str) -> int:
    """컬렉션의 문서 수를 반환한다."""
    client = get_qdrant_client()
    info = client.get_collection(collection_name=collection_name)
    return info.points_count or 0


def select_mode() -> str:
    """검색 모드를 선택한다."""
    print("\n📋 검색 모드를 선택하세요:")
    for key, (_, desc) in SEARCH_MODES.items():
        print(f"  {key}. {desc}")

    while True:
        choice = input("\n선택 (1/2/3/4) [1]: ").strip()
        if not choice:
            choice = "1"
        if choice in SEARCH_MODES:
            return SEARCH_MODES[choice][0]
        print("⚠️  1, 2, 3, 4 중 하나를 선택하세요.")


def print_hits(hits: list[dict[str, Any]], mode: str) -> None:
    """검색 결과를 출력한다."""
    if not hits:
        print("\n📚 검색 결과 없음")
        return

    print(f"\n📚 검색된 문서 {len(hits)}건:")
    for index, hit in enumerate(hits, start=1):
        metadata = hit["metadata"]
        if mode == "rerank":
            score = hit.get("relevance_score", 0)
            label = "relevance"
        elif mode == "naive":
            score = hit.get("distance", 0)
            label = "거리"
        else:
            score = hit.get("score", 0)
            label = "score"
        print(
            f"  {index}. {metadata.get('shop_name', '?')} ({metadata.get('district', '?')}) "
            f"— {metadata.get('benefit', '정보 없음')}  [{label}: {score:.4f}]"
        )


def main() -> None:
    """대화형 검색 CLI."""
    logging.basicConfig(level=logging.WARNING, format="%(asctime)s - %(levelname)s - %(message)s")

    print("\n🚀 부산 가족사랑카드 업체 검색")
    print("=" * 50)

    mode = select_mode()
    mode_labels = {
        "naive": "Naive RAG",
        "bm25": "BM25",
        "hybrid": "Hybrid (Dense+BM25 RRF)",
        "rerank": "Hybrid + Re-rank",
    }
    print(f"\n✅ [{mode_labels[mode]}] 모드 선택됨")

    print("📦 벡터스토어 로딩 중...")
    if mode == "naive":
        collection_name = load_collection()
    else:
        collection_name = load_hybrid_collection()

    doc_count = get_document_count(collection_name)
    print(f"✅ {doc_count}건의 업체 정보가 준비되었습니다.\n")

    if doc_count == 0:
        if mode == "naive":
            print("⚠️  인덱싱된 문서가 없습니다. 먼저 인덱싱을 실행해주세요:")
            print("   uv run python -m rag_playground.application.index")
        else:
            print("⚠️  하이브리드 컬렉션이 비어있습니다. 먼저 인덱싱을 실행해주세요:")
            print("   uv run python -m rag_playground.application.index --mode hybrid")
        return

    answer_fn = {
        "naive": answer_query,
        "bm25": answer_query_bm25,
        "hybrid": answer_query_hybrid,
        "rerank": answer_query_rerank,
    }[mode]

    print("🔍 질문을 입력하세요 (종료: q)\n")

    while True:
        try:
            query = input("❓ > ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\n👋 종료합니다.")
            break

        if not query:
            continue
        if query.lower() == "q":
            print("👋 종료합니다.")
            break

        hits, answer = answer_fn(query=query, collection_name=collection_name, n_results=5)
        print_hits(hits, mode)

        print("\n💬 답변 생성 중...")
        print(f"\n💬 답변:\n{answer}\n")
        print("-" * 50)


if __name__ == "__main__":
    main()
