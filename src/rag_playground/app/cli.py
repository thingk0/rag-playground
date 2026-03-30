"""Naive RAG 대화형 CLI.

Usage:
    uv run python -m rag_playground.app.cli
"""

from __future__ import annotations

import logging

from rag_playground.adapters.vectorstore.qdrant import get_qdrant_client
from rag_playground.application.answer import answer_query, load_collection


def get_document_count(collection_name: str) -> int:
    """컬렉션의 문서 수를 반환한다."""
    client = get_qdrant_client()
    info = client.get_collection(collection_name=collection_name)
    return info.points_count or 0


def main() -> None:
    """대화형 검색 CLI."""
    logging.basicConfig(level=logging.WARNING, format="%(asctime)s - %(levelname)s - %(message)s")

    print("\n🚀 부산 가족사랑카드 업체 검색 (Naive RAG)")
    print("=" * 50)

    print("📦 벡터스토어 로딩 중...")
    collection_name = load_collection()
    doc_count = get_document_count(collection_name)
    print(f"✅ {doc_count}건의 업체 정보가 준비되었습니다.\n")

    if doc_count == 0:
        print("⚠️  인덱싱된 문서가 없습니다. 먼저 인덱싱을 실행해주세요:")
        print("   uv run python -m rag_playground.application.index")
        return

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

        hits, answer = answer_query(query=query, collection_name=collection_name, n_results=5)
        if hits:
            print(f"\n📚 검색된 문서 {len(hits)}건:")
            for index, hit in enumerate(hits, start=1):
                metadata = hit["metadata"]
                distance = hit["distance"]
                print(
                    f"  {index}. {metadata.get('shop_name', '?')} ({metadata.get('district', '?')}) "
                    f"— {metadata.get('benefit', '정보 없음')}  [거리: {distance:.4f}]"
                )
        else:
            print("\n📚 검색 결과 없음")

        print("\n💬 답변 생성 중...")
        print(f"\n💬 답변:\n{answer}\n")
        print("-" * 50)


if __name__ == "__main__":
    main()
