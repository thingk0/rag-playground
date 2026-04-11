"""Agentic RAG planner 단위 테스트."""

from __future__ import annotations

import unittest

from rag_playground.application.agentic import build_agentic_plan, preview_agentic_query, should_fallback


class AgenticPlannerTests(unittest.TestCase):
    """Quest 7 planner 휴리스틱 검증."""

    def test_specific_business_query_uses_family_card(self) -> None:
        plan = build_agentic_plan("동래구 목욕탕")
        self.assertEqual(plan.sources, ["family_card"])
        self.assertEqual(plan.initial_mode, "rerank")
        self.assertEqual(plan.ambiguity, "low")

    def test_broad_family_query_uses_both_sources(self) -> None:
        plan = build_agentic_plan("이번 주말에 아이들이랑 갈 만한 데")
        self.assertEqual(plan.sources, ["library"])
        self.assertEqual(plan.initial_mode, "multi_rerank")
        self.assertEqual(plan.ambiguity, "high")

    def test_preview_includes_answer_step(self) -> None:
        preview = preview_agentic_query("도서관 어디가 좋아?")
        self.assertEqual(preview.steps[-1].name, "answer_with_original_query")


class AgenticFallbackTests(unittest.TestCase):
    """fallback 판단 규칙 검증."""

    def test_empty_hits_require_fallback(self) -> None:
        self.assertTrue(should_fallback([], threshold=0.55))

    def test_low_relevance_requires_fallback(self) -> None:
        hits = [{"relevance_score": 0.32, "metadata": {}, "document": "x"}]
        self.assertTrue(should_fallback(hits, threshold=0.55))

    def test_high_relevance_stops_fallback(self) -> None:
        hits = [{"relevance_score": 0.91, "metadata": {}, "document": "x"}]
        self.assertFalse(should_fallback(hits, threshold=0.55))


if __name__ == "__main__":
    unittest.main()
