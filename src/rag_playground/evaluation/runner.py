"""검색 품질 정량 평가 러너.

Usage:
    uv run python -m rag_playground.evaluation.runner
    uv run python -m rag_playground.evaluation.runner --k 10
    uv run python -m rag_playground.evaluation.runner --output data/eval/results.json
"""

from __future__ import annotations

import argparse
import json
import logging
import time
from pathlib import Path

from rag_playground.application.agentic import run_agentic_query
from rag_playground.application.answer import retrieve_hits
from rag_playground.application.sources import get_source_config
from rag_playground.evaluation.dataset import EvalQuery, grade_hits, load_golden_set
from rag_playground.evaluation.metrics import mrr, ndcg_at_k, precision_at_k

EVAL_MODES = ("naive", "bm25", "hybrid", "rerank", "hyde_rerank", "multi_rerank", "agentic")

DEFAULT_GOLDEN_PATH = Path(__file__).resolve().parents[3] / "data" / "eval" / "golden_set.json"
DEFAULT_OUTPUT_PATH = Path(__file__).resolve().parents[3] / "data" / "eval" / "results.json"


def get_hits_for_mode(
    query: str,
    mode: str,
    source: str,
    n_results: int = 5,
) -> list[dict]:
    """특정 모드로 검색하여 hits를 반환한다."""
    if mode == "agentic":
        return run_agentic_query(query, n_results=n_results).hits

    source_config = get_source_config(source)
    collection = (
        source_config.collection_name if mode == "naive" else source_config.hybrid_collection_name
    )
    return retrieve_hits(
        query,
        collection_name=collection,
        mode=mode,
        n_results=n_results,
        domain_context=source_config.domain_hint,
    )


def evaluate_query(
    eval_query: EvalQuery,
    modes: tuple[str, ...],
    k: int = 5,
    n_results: int = 5,
) -> dict:
    """단일 질의에 대해 모든 모드의 메트릭을 계산한다."""
    results = {"query_id": eval_query.query_id, "query": eval_query.query, "modes": {}}

    for mode in modes:
        start = time.time()
        try:
            hits = get_hits_for_mode(
                eval_query.query, mode, eval_query.source, n_results=n_results
            )
        except Exception as e:
            logging.warning("Mode %s failed for query '%s': %s", mode, eval_query.query, e)
            results["modes"][mode] = {"error": str(e)}
            continue
        elapsed = time.time() - start

        relevances = grade_hits(hits, eval_query.relevant_docs)
        results["modes"][mode] = {
            "ndcg": round(ndcg_at_k(relevances, k), 4),
            "mrr": round(mrr(relevances), 4),
            "precision": round(precision_at_k(relevances, k), 4),
            "elapsed_ms": round(elapsed * 1000, 0),
            "relevances": relevances,
        }

    return results


def run_evaluation(
    golden_path: str | Path,
    k: int = 5,
    n_results: int = 5,
    output_path: str | Path | None = None,
) -> list[dict]:
    """전체 평가를 실행한다."""
    golden = load_golden_set(golden_path)
    logging.info("평가셋 로드 완료: %d개 질의", len(golden))

    all_results = []
    for eval_query in golden:
        logging.info("평가 중: %s — '%s'", eval_query.query_id, eval_query.query)
        result = evaluate_query(eval_query, EVAL_MODES, k=k, n_results=n_results)
        all_results.append(result)

    # Print summary table
    print_summary(all_results, k=k)

    # Save results
    if output_path:
        Path(output_path).parent.mkdir(parents=True, exist_ok=True)
        Path(output_path).write_text(json.dumps(all_results, ensure_ascii=False, indent=2), encoding="utf-8")
        logging.info("결과 저장: %s", output_path)

    return all_results


def print_summary(all_results: list[dict], k: int) -> None:
    """평가 결과 요약 테이블을 출력한다."""
    # Aggregate metrics per mode
    mode_metrics: dict[str, list[dict]] = {}
    for qr in all_results:
        for mode, data in qr.get("modes", {}).items():
            if "error" in data:
                continue
            mode_metrics.setdefault(mode, []).append(data)

    header = f"{'Mode':<16} {'NDCG@' + str(k):<10} {'MRR':<10} {'P@' + str(k):<10} {'Avg ms':<10}"
    print(f"\n{'=' * len(header)}")
    print(header)
    print("-" * len(header))

    for mode in EVAL_MODES:
        if mode not in mode_metrics:
            continue
        entries = mode_metrics[mode]
        avg_ndcg = sum(e["ndcg"] for e in entries) / len(entries)
        avg_mrr = sum(e["mrr"] for e in entries) / len(entries)
        avg_p = sum(e["precision"] for e in entries) / len(entries)
        avg_ms = sum(e["elapsed_ms"] for e in entries) / len(entries)
        print(f"{mode:<16} {avg_ndcg:<10.4f} {avg_mrr:<10.4f} {avg_p:<10.4f} {avg_ms:<10.0f}")

    print("=" * len(header))

    # Per-query detail
    for qr in all_results:
        print(f"\n  {qr['query_id']}: '{qr['query']}'")
        for mode in EVAL_MODES:
            data = qr.get("modes", {}).get(mode)
            if not data or "error" in data:
                print(f"    {mode:<16} (error)")
                continue
            rels = data["relevances"]
            print(f"    {mode:<16} NDCG={data['ndcg']:.4f}  MRR={data['mrr']:.4f}  P@{k}={data['precision']:.4f}  rels={rels}")


def main() -> None:
    """평가 CLI 엔트리포인트."""
    parser = argparse.ArgumentParser(description="RAG 검색 품질 정량 평가")
    parser.add_argument("--golden", type=str, default=str(DEFAULT_GOLDEN_PATH), help="golden_set.json 경로")
    parser.add_argument("--output", type=str, default=str(DEFAULT_OUTPUT_PATH), help="결과 저장 경로")
    parser.add_argument("--k", type=int, default=5, help="평가할 top-K (기본: 5)")
    parser.add_argument("--n-results", type=int, default=5, help="검색 결과 수 (기본: 5)")
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
    run_evaluation(args.golden, k=args.k, n_results=args.n_results, output_path=args.output)


if __name__ == "__main__":
    main()
