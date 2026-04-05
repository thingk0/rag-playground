"""인덱싱 유스케이스."""

from __future__ import annotations

import argparse
import logging
import time
from pathlib import Path

from rag_playground.adapters.vectorstore.qdrant import (
    get_or_create_collection,
    get_or_create_hybrid_collection,
    index_documents,
    index_documents_hybrid,
)
from rag_playground.config import DEFAULT_FAMILY_CARD_JSON_PATH
from rag_playground.domain.document import load_shop_documents

logger = logging.getLogger(__name__)


def run_index(json_path: str | Path = DEFAULT_FAMILY_CARD_JSON_PATH) -> int:
    """JSON 문서를 로드해 벡터스토어에 인덱싱한다."""
    path = Path(json_path)
    logger.info("데이터 로드: %s", path)
    documents = load_shop_documents(path)
    logger.info("총 %s개 문서 변환 완료", len(documents))

    logger.info("Qdrant 컬렉션 생성/로드 중...")
    collection_name = get_or_create_collection()

    logger.info("임베딩 & 인덱싱 시작...")
    return index_documents(documents, collection_name)


def run_index_hybrid(json_path: str | Path = DEFAULT_FAMILY_CARD_JSON_PATH) -> int:
    """JSON 문서를 로드해 하이브리드 컬렉션에 인덱싱한다."""
    path = Path(json_path)
    logger.info("데이터 로드: %s", path)
    documents = load_shop_documents(path)
    logger.info("총 %s개 문서 변환 완료", len(documents))

    logger.info("하이브리드 Qdrant 컬렉션 생성/로드 중...")
    collection_name = get_or_create_hybrid_collection()

    logger.info("임베딩 & 하이브리드 인덱싱 시작...")
    return index_documents_hybrid(documents, collection_name)


def main() -> None:
    """인덱싱 CLI 엔트리포인트."""
    parser = argparse.ArgumentParser(description="RAG 인덱싱")
    parser.add_argument(
        "--mode",
        choices=["naive", "hybrid"],
        default="naive",
        help="인덱싱 모드 (기본: naive)",
    )
    parser.add_argument(
        "--json-path",
        type=str,
        default=str(DEFAULT_FAMILY_CARD_JSON_PATH),
        help="JSON 데이터 파일 경로",
    )
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")

    mode_label = "Hybrid" if args.mode == "hybrid" else "Naive RAG"
    logger.info("=== %s 인덱싱 시작 ===", mode_label)
    start = time.time()

    if args.mode == "hybrid":
        indexed_count = run_index_hybrid(args.json_path)
    else:
        indexed_count = run_index(args.json_path)

    elapsed = time.time() - start
    logger.info("=== 인덱싱 완료: %s건, %.1f초 소요 ===", indexed_count, elapsed)


if __name__ == "__main__":
    main()
