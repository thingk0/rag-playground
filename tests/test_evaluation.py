"""평가 모듈 단위 테스트."""

from __future__ import annotations

import json
import tempfile
import unittest

from rag_playground.evaluation.dataset import (
    RelevantDoc,
    grade_hits,
    load_golden_set,
    match_relevance,
)
from rag_playground.evaluation.metrics import (
    dcg_at_k,
    mrr,
    ndcg_at_k,
    precision_at_k,
)


class TestDcgAtK(unittest.TestCase):
    """DCG@K 계산 테스트."""

    def test_known_values(self):
        # rels=[3,2,3,0,1], k=5
        # DCG = (7/log2(2)) + (3/log2(3)) + (7/log2(4)) + (0) + (1/log2(6))
        # = 7.0 + 1.8928 + 3.5 + 0 + 0.3869 = 12.7796
        rels = [3, 2, 3, 0, 1]
        result = dcg_at_k(rels, 5)
        self.assertAlmostEqual(result, 12.7796, places=3)

    def test_k_larger_than_list(self):
        rels = [3, 2]
        result = dcg_at_k(rels, 10)
        # Only uses available elements: (7/log2(2)) + (3/log2(3))
        self.assertAlmostEqual(result, 7.0 + 3.0 / 1.5850, places=3)

    def test_empty_relevances(self):
        self.assertEqual(dcg_at_k([], 5), 0.0)

    def test_all_zeros(self):
        self.assertEqual(dcg_at_k([0, 0, 0], 3), 0.0)


class TestNdcgAtK(unittest.TestCase):
    """NDCG@K 계산 테스트."""

    def test_perfect_ranking(self):
        # Already in ideal order: [3, 2, 1]
        self.assertAlmostEqual(ndcg_at_k([3, 2, 1], 3), 1.0)

    def test_worst_ranking(self):
        # All zeros = no relevant docs
        self.assertEqual(ndcg_at_k([0, 0, 0], 3), 0.0)

    def test_reversed(self):
        # [1, 2, 3] vs ideal [3, 2, 1] — NDCG should be less than 1.0
        result = ndcg_at_k([1, 2, 3], 3)
        self.assertGreater(result, 0.0)
        self.assertLess(result, 1.0)

    def test_empty(self):
        self.assertEqual(ndcg_at_k([], 5), 0.0)


class TestMrr(unittest.TestCase):
    """MRR 계산 테스트."""

    def test_first_position(self):
        self.assertEqual(mrr([3, 0, 0]), 1.0)

    def test_second_position(self):
        self.assertAlmostEqual(mrr([0, 2, 0]), 0.5)

    def test_third_position(self):
        self.assertAlmostEqual(mrr([0, 0, 1]), 1.0 / 3.0, places=4)

    def test_no_relevant(self):
        self.assertEqual(mrr([0, 0, 0]), 0.0)

    def test_empty(self):
        self.assertEqual(mrr([]), 0.0)


class TestPrecisionAtK(unittest.TestCase):
    """Precision@K 계산 테스트."""

    def test_all_relevant(self):
        self.assertAlmostEqual(precision_at_k([1, 2, 3], 3), 1.0)

    def test_half_relevant(self):
        self.assertAlmostEqual(precision_at_k([3, 0, 3, 0], 4), 0.5)

    def test_none_relevant(self):
        self.assertAlmostEqual(precision_at_k([0, 0, 0], 3), 0.0)

    def test_k_zero(self):
        self.assertEqual(precision_at_k([1, 2], 0), 0.0)

    def test_k_larger_than_list(self):
        # Only 2 elements, k=5 — counts relevant in available elements
        self.assertAlmostEqual(precision_at_k([1, 0], 5), 1.0 / 5.0)


class TestMatchRelevance(unittest.TestCase):
    """prefix 매칭 로직 테스트."""

    def setUp(self):
        self.docs = [
            RelevantDoc(doc_text_prefix="[동래구 / 세탁, 목욕업] 동래해수탕", grade=3),
            RelevantDoc(doc_text_prefix="[부산진구 / 세탁, 목욕업] 한일탕", grade=2),
            RelevantDoc(doc_text_prefix="[동래구 / 한의원]", grade=1),
        ]

    def test_exact_prefix_match(self):
        hit = {"document": "[동래구 / 세탁, 목욕업] 동래해수탕\n주소: ..."}
        self.assertEqual(match_relevance(hit, self.docs), 3)

    def test_partial_prefix_match(self):
        hit = {"document": "[부산진구 / 세탁, 목욕업] 한일탕\n주소: 부산진구 ..."}
        self.assertEqual(match_relevance(hit, self.docs), 2)

    def test_no_match(self):
        hit = {"document": "[해운대구 / 요식업등] 어떤식당\n주소: ..."}
        self.assertEqual(match_relevance(hit, self.docs), 0)

    def test_missing_document_key(self):
        hit = {"metadata": {}}
        self.assertEqual(match_relevance(hit, self.docs), 0)


class TestGradeHits(unittest.TestCase):
    """검색 결과 → relevance grade 리스트 변환 테스트."""

    def test_grading(self):
        docs = [
            RelevantDoc(doc_text_prefix="[동래구 / 세탁, 목욕업]", grade=3),
            RelevantDoc(doc_text_prefix="[부산진구 / 요식업등]", grade=2),
        ]
        hits = [
            {"document": "[동래구 / 세탁, 목욕업] 벽초온천\n주소: ..."},
            {"document": "[해운대구 / 미용업] 어떤미용실\n주소: ..."},
            {"document": "[부산진구 / 요식업등] 청기와\n주소: ..."},
        ]
        self.assertEqual(grade_hits(hits, docs), [3, 0, 2])

    def test_empty_hits(self):
        self.assertEqual(grade_hits([], []), [])


class TestLoadGoldenSet(unittest.TestCase):
    """golden_set.json 로더 테스트."""

    def test_load(self):
        data = [
            {
                "query_id": "q01",
                "query": "테스트 질의",
                "source": "family_card",
                "relevant_docs": [
                    {"doc_text_prefix": "[동래구 / 세탁, 목욕업]", "grade": 3},
                ],
            }
        ]
        with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False, encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=False)
            f.flush()
            loaded = load_golden_set(f.name)

        self.assertEqual(len(loaded), 1)
        self.assertEqual(loaded[0].query_id, "q01")
        self.assertEqual(loaded[0].query, "테스트 질의")
        self.assertEqual(loaded[0].source, "family_card")
        self.assertEqual(len(loaded[0].relevant_docs), 1)
        self.assertEqual(loaded[0].relevant_docs[0].doc_text_prefix, "[동래구 / 세탁, 목욕업]")
        self.assertEqual(loaded[0].relevant_docs[0].grade, 3)


if __name__ == "__main__":
    unittest.main()
