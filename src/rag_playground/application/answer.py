"""질의응답 유스케이스."""

from __future__ import annotations

from typing import Any

from rag_playground.adapters.llm.openai_chat import generate_answer
from rag_playground.adapters.vectorstore.qdrant import get_or_create_collection, search


def load_collection() -> str:
    """질의에 사용할 컬렉션 이름을 로드한다."""
    return get_or_create_collection()


def answer_query(
    query: str,
    collection_name: str,
    n_results: int = 5,
) -> tuple[list[dict[str, Any]], str]:
    """벡터 검색과 LLM 답변 생성을 실행한다."""
    hits = search(query, collection_name=collection_name, n_results=n_results)
    answer = generate_answer(query, hits)
    return hits, answer
