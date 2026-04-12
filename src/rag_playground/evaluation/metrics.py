"""검색 품질 평가 메트릭."""

from __future__ import annotations

import math


def dcg_at_k(relevances: list[int], k: int) -> float:
    """Discounted Cumulative Gain at K.

    DCG@k = Σ_{i=1}^{k} (2^{rel_i} - 1) / log2(i + 1)
    """
    score = 0.0
    for i in range(min(k, len(relevances))):
        score += (2 ** relevances[i] - 1) / math.log2(i + 2)
    return score


def ndcg_at_k(relevances: list[int], k: int) -> float:
    """Normalized DCG at K. Perfect ranking = 1.0, no relevant docs = 0.0."""
    actual_dcg = dcg_at_k(relevances, k)
    ideal_dcg = dcg_at_k(sorted(relevances, reverse=True), k)
    if ideal_dcg == 0.0:
        return 0.0
    return actual_dcg / ideal_dcg


def mrr(relevances: list[int]) -> float:
    """Mean Reciprocal Rank. First relevant doc (grade >= 1) → 1/rank. No relevant = 0.0."""
    for i, rel in enumerate(relevances):
        if rel >= 1:
            return 1.0 / (i + 1)
    return 0.0


def precision_at_k(relevances: list[int], k: int) -> float:
    """Precision at K. (relevant docs in top-K) / K. Grade >= 1 counts as relevant."""
    if k == 0:
        return 0.0
    count = sum(1 for rel in relevances[:k] if rel >= 1)
    return count / k
