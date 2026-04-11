"""Agentic RAG 계획 모델."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Literal

AgentSearchMode = Literal["rerank", "hyde_rerank", "multi_rerank"]
AgentStepStatus = Literal["planned", "completed", "skipped", "failed"]
AmbiguityLevel = Literal["low", "medium", "high"]


@dataclass(slots=True)
class AgentPlan:
    """질의별 검색 전략 계획."""

    query: str
    sources: list[str]
    initial_mode: AgentSearchMode
    fallback_modes: list[AgentSearchMode] = field(default_factory=list)
    ambiguity: AmbiguityLevel = "low"
    rationale: str = ""
    sufficiency_threshold: float = 0.55


@dataclass(slots=True)
class AgentStep:
    """에이전트 실행 단계."""

    name: str
    reason: str
    status: AgentStepStatus = "planned"
    mode: AgentSearchMode | None = None
    details: dict[str, Any] = field(default_factory=dict)


@dataclass(slots=True)
class AgentResult:
    """Agentic RAG 실행 결과."""

    plan: AgentPlan
    steps: list[AgentStep] = field(default_factory=list)
    hits: list[dict[str, Any]] = field(default_factory=list)
    answer: str = ""
    selected_mode: AgentSearchMode | None = None
    success: bool = False
