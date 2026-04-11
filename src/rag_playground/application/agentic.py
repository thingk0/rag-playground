"""Quest 7 Agentic RAG 실행기."""

from __future__ import annotations

import argparse
from typing import Any

from rag_playground.adapters.llm.openai_chat import generate_answer
from rag_playground.adapters.reranker.novita import rerank_hits
from rag_playground.application.answer import retrieve_hits
from rag_playground.application.sources import get_source_config
from rag_playground.domain import AgentPlan, AgentResult, AgentStep

BROAD_HINTS = ("추천", "갈 만", "가볼", "어디", "좋은 곳", "놀", "코스", "주말")
SCENARIO_HINTS = ("아이", "아이들", "가족", "데이트", "나들이", "체험")
BUSAN_DISTRICTS = (
    "중구",
    "서구",
    "동구",
    "영도구",
    "부산진구",
    "동래구",
    "남구",
    "북구",
    "해운대구",
    "사하구",
    "금정구",
    "강서구",
    "연제구",
    "수영구",
    "사상구",
    "기장군",
)


def _contains_any(query: str, keywords: tuple[str, ...]) -> bool:
    """질의에 키워드가 포함되는지 검사한다."""
    return any(keyword in query for keyword in keywords)


def _count_keyword_matches(query: str, keywords: tuple[str, ...]) -> int:
    """질의에서 키워드가 몇 개 매칭되는지 센다."""
    return sum(1 for keyword in keywords if keyword in query)


def _select_sources(query: str) -> tuple[list[str], list[str]]:
    """질의에 맞는 데이터 소스를 선택한다."""
    reasons: list[str] = []
    family = get_source_config("family_card")
    library = get_source_config("library")

    family_score = _count_keyword_matches(query, family.query_keywords)
    library_score = _count_keyword_matches(query, library.query_keywords)
    broad_query = _contains_any(query, BROAD_HINTS)
    scenario_query = _contains_any(query, SCENARIO_HINTS)

    if family_score > 0 and library_score == 0:
        reasons.append("업체/할인 계열 키워드가 강하게 보입니다")
        return ["family_card"], reasons

    if library_score > 0 and family_score == 0:
        reasons.append("도서관/문화공간 계열 키워드가 강하게 보입니다")
        return ["library"], reasons

    if library_score > family_score:
        reasons.append("도서관 관련 키워드 비중이 더 높습니다")
        return ["library"], reasons

    if family_score > library_score:
        reasons.append("업체 관련 키워드 비중이 더 높습니다")
        return ["family_card"], reasons

    if broad_query or scenario_query:
        reasons.append("질의가 탐색형이라 두 소스를 모두 확인합니다")
        return ["library", "family_card"], reasons

    reasons.append("명확한 소스 힌트가 없어 기본 소스부터 조회합니다")
    return ["family_card", "library"], reasons


def _expand_sources(sources: list[str]) -> list[str]:
    """fallback 시 조회 소스를 확장한다."""
    if len(sources) > 1:
        return sources
    if sources == ["library"]:
        return ["library", "family_card"]
    return ["family_card", "library"]


def build_agentic_plan(query: str) -> AgentPlan:
    """질의로부터 Agentic RAG 실행 계획을 만든다."""
    reasons: list[str] = []
    score = 0

    has_broad_intent = _contains_any(query, BROAD_HINTS)
    has_scenario_intent = _contains_any(query, SCENARIO_HINTS)
    has_district = _contains_any(query, BUSAN_DISTRICTS)

    if has_broad_intent:
        score += 2
        reasons.append("추천형 표현이 포함되어 있습니다")
    if has_scenario_intent:
        score += 1
        reasons.append("상황형 조건이 포함되어 있습니다")
    if not has_district:
        score += 1
        reasons.append("구/군 제약이 없습니다")

    if score >= 3:
        ambiguity = "high"
        initial_mode = "multi_rerank"
        fallback_modes = ["hyde_rerank", "rerank"]
        sufficiency_threshold = 0.03
    elif score == 2:
        ambiguity = "medium"
        initial_mode = "multi_rerank"
        fallback_modes = ["rerank", "hyde_rerank"]
        sufficiency_threshold = 0.15
    else:
        ambiguity = "low"
        initial_mode = "rerank"
        fallback_modes = ["multi_rerank"]
        sufficiency_threshold = 0.55

    sources, source_reasons = _select_sources(query)
    rationale_parts = reasons + source_reasons
    rationale = "; ".join(rationale_parts) if rationale_parts else "지역과 의도가 비교적 구체적입니다"

    return AgentPlan(
        query=query,
        sources=sources,
        initial_mode=initial_mode,
        fallback_modes=fallback_modes,
        ambiguity=ambiguity,
        rationale=rationale,
        sufficiency_threshold=sufficiency_threshold,
    )


def _annotate_hits(hits: list[dict[str, Any]], source_id: str) -> list[dict[str, Any]]:
    """검색 결과에 source 메타데이터를 보강한다."""
    source = get_source_config(source_id)
    annotated: list[dict[str, Any]] = []
    for hit in hits:
        metadata = dict(hit.get("metadata", {}))
        metadata.setdefault("source_id", source.source_id)
        metadata.setdefault("source_label", source.label)
        metadata.setdefault("source", source.description)
        annotated.append({**hit, "metadata": metadata})
    return annotated


def _run_mode_for_sources(
    query: str,
    sources: list[str],
    mode: str,
    n_results: int,
    fetch_multiplier: int,
) -> list[dict[str, Any]]:
    """선택된 모든 소스에서 동일한 검색 전략을 실행한다."""
    aggregated_hits: list[dict[str, Any]] = []

    for source_id in sources:
        source = get_source_config(source_id)
        source_hits = retrieve_hits(
            query,
            collection_name=source.hybrid_collection_name,
            mode=mode,
            n_results=n_results,
            fetch_multiplier=fetch_multiplier,
            domain_context=source.domain_hint,
        )
        aggregated_hits.extend(_annotate_hits(source_hits, source_id))

    if not aggregated_hits:
        return []
    if len(aggregated_hits) <= n_results:
        return aggregated_hits
    return rerank_hits(query, aggregated_hits, top_n=n_results)


def _top_relevance(hits: list[dict[str, Any]]) -> float:
    """상위 relevance score를 반환한다."""
    if not hits:
        return 0.0
    return max(hit.get("relevance_score", 0.0) for hit in hits)


def should_fallback(hits: list[dict[str, Any]], threshold: float) -> bool:
    """검색 결과가 부족한지 판단한다."""
    if not hits:
        return True
    return _top_relevance(hits) < threshold


def preview_agentic_query(query: str) -> AgentResult:
    """질의에 대한 에이전트 실행 계획 preview를 반환한다."""
    plan = build_agentic_plan(query)
    steps = [
        AgentStep(
            name="analyze_query",
            reason="질의의 모호함과 제약 조건을 파악합니다",
            status="completed",
            details={"ambiguity": plan.ambiguity, "sources": plan.sources},
        ),
        AgentStep(
            name="primary_search",
            reason="초기 검색 전략을 적용합니다",
            mode=plan.initial_mode,
        ),
    ]

    for mode in plan.fallback_modes:
        steps.append(
            AgentStep(
                name="fallback_search",
                reason="초기 결과가 부족할 때 대체 전략을 적용합니다",
                mode=mode,
            )
        )

    steps.append(
        AgentStep(
            name="answer_with_original_query",
            reason="최종 답변은 항상 원본 질의를 기준으로 생성합니다",
        )
    )

    return AgentResult(plan=plan, steps=steps)


def run_agentic_query(
    query: str,
    n_results: int = 5,
    fetch_multiplier: int = 4,
) -> AgentResult:
    """Quest 7 Agentic RAG 실행."""
    plan = build_agentic_plan(query)
    steps = [
        AgentStep(
            name="analyze_query",
            reason="질의의 모호함과 제약 조건을 파악합니다",
            status="completed",
            details={"ambiguity": plan.ambiguity, "sources": plan.sources},
        )
    ]

    attempted_modes = [plan.initial_mode, *plan.fallback_modes]
    selected_hits: list[dict[str, Any]] = []
    selected_mode: str | None = None
    success = False

    for index, mode in enumerate(attempted_modes):
        step_name = "primary_search" if index == 0 else "fallback_search"
        attempt_sources = plan.sources if index == 0 else _expand_sources(plan.sources)
        mode_hits = _run_mode_for_sources(
            query=query,
            sources=attempt_sources,
            mode=mode,
            n_results=n_results,
            fetch_multiplier=fetch_multiplier,
        )
        top_relevance = _top_relevance(mode_hits)
        needs_fallback = should_fallback(mode_hits, threshold=plan.sufficiency_threshold)

        steps.append(
            AgentStep(
                name=step_name,
                reason="검색 전략을 적용해 후보 문서를 수집합니다",
                status="completed",
                mode=mode,
                details={
                    "hit_count": len(mode_hits),
                    "top_relevance": top_relevance,
                    "needs_fallback": needs_fallback,
                    "sources": attempt_sources,
                },
            )
        )

        if mode_hits and (not selected_hits or top_relevance > _top_relevance(selected_hits)):
            selected_hits = mode_hits
            selected_mode = mode

        if not needs_fallback:
            success = True
            selected_hits = mode_hits
            selected_mode = mode
            break

    answer = generate_answer(query, selected_hits)
    steps.append(
        AgentStep(
            name="answer_with_original_query",
            reason="최종 답변은 항상 원본 질의를 기준으로 생성합니다",
            status="completed",
            details={"selected_mode": selected_mode, "success": success},
        )
    )

    return AgentResult(
        plan=plan,
        steps=steps,
        hits=selected_hits,
        answer=answer,
        selected_mode=selected_mode,
        success=success,
    )


def _format_result(result: AgentResult, include_answer: bool = False) -> str:
    """CLI 출력용 result 포맷."""
    source_labels = [get_source_config(source_id).label for source_id in result.plan.sources]
    lines = [
        "",
        "Quest 7 Agentic RAG",
        "=" * 28,
        f"query: {result.plan.query}",
        f"sources: {', '.join(source_labels)}",
        f"ambiguity: {result.plan.ambiguity}",
        f"initial_mode: {result.plan.initial_mode}",
        f"fallback_modes: {', '.join(result.plan.fallback_modes) or '-'}",
        f"threshold: {result.plan.sufficiency_threshold:.2f}",
        f"rationale: {result.plan.rationale}",
        "",
        "steps:",
    ]

    for index, step in enumerate(result.steps, start=1):
        mode = f" [{step.mode}]" if step.mode else ""
        detail_suffix = ""
        if step.details:
            preview_details = ", ".join(f"{key}={value}" for key, value in step.details.items())
            detail_suffix = f" ({preview_details})"
        lines.append(f"  {index}. {step.name}{mode} - {step.reason}{detail_suffix}")

    if result.hits:
        lines.extend(["", "hits:"])
        for index, hit in enumerate(result.hits, start=1):
            metadata = hit["metadata"]
            title = metadata.get("title") or metadata.get("shop_name") or metadata.get("name") or "?"
            district = metadata.get("district", "?")
            source = metadata.get("source_label", "?")
            summary = metadata.get("summary") or metadata.get("benefit") or "정보 없음"
            score = hit.get("relevance_score", hit.get("score", 0.0))
            lines.append(
                f"  {index}. {title} ({district}) / {source} - {summary} [score: {score:.4f}]"
            )

    if include_answer:
        lines.extend(["", "answer:", result.answer])

    return "\n".join(lines)


def main() -> None:
    """Quest 7 Agentic RAG CLI."""
    parser = argparse.ArgumentParser(description="Quest 7 Agentic RAG")
    parser.add_argument("--query", required=True, help="분석할 질의")
    parser.add_argument("--preview", action="store_true", help="실행 계획만 출력")
    args = parser.parse_args()

    if args.preview:
        result = preview_agentic_query(args.query)
        print(_format_result(result, include_answer=False))
        return

    result = run_agentic_query(args.query)
    print(_format_result(result, include_answer=True))


if __name__ == "__main__":
    main()
