"""평가셋 로더 및 검색 결과 매칭."""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path


@dataclass(slots=True)
class RelevantDoc:
    """정답 문서 항목."""
    doc_text_prefix: str
    grade: int


@dataclass(slots=True)
class EvalQuery:
    """평가용 질의 항목."""
    query_id: str
    query: str
    source: str
    relevant_docs: list[RelevantDoc] = field(default_factory=list)


def load_golden_set(path: str | Path) -> list[EvalQuery]:
    """golden_set.json을 로드한다.

    Args:
        path: golden_set.json 파일 경로

    Returns:
        EvalQuery 리스트
    """
    raw = Path(path).read_text(encoding="utf-8")
    data = json.loads(raw)
    results: list[EvalQuery] = []
    for item in data:
        relevant = [
            RelevantDoc(
                doc_text_prefix=rd["doc_text_prefix"],
                grade=rd["grade"],
            )
            for rd in item.get("relevant_docs", [])
        ]
        results.append(
            EvalQuery(
                query_id=item["query_id"],
                query=item["query"],
                source=item["source"],
                relevant_docs=relevant,
            )
        )
    return results


def match_relevance(hit: dict, relevant_docs: list[RelevantDoc]) -> int:
    """검색 결과 hit가 정답 문서와 매칭되는지 확인한다.

    hit["document"] 텍스트가 doc_text_prefix로 시작하면 해당 grade 반환.
    매칭 안 되면 0 반환.
    """
    doc_text = hit.get("document", "")
    for doc in relevant_docs:
        if doc_text.startswith(doc.doc_text_prefix):
            return doc.grade
    return 0


def grade_hits(hits: list[dict], relevant_docs: list[RelevantDoc]) -> list[int]:
    """검색 결과 순서대로 각 hit의 relevance grade 리스트를 반환한다."""
    return [match_relevance(hit, relevant_docs) for hit in hits]
