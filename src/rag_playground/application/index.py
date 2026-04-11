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
from rag_playground.application.sources import get_all_source_configs, get_source_config
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


def run_index_for_source(source_id: str, mode: str) -> int:
    """지정한 소스를 지정한 모드로 인덱싱한다."""
    source = get_source_config(source_id)
    path = Path(source.default_json_path)
    logger.info("데이터 로드: %s", path)
    documents = source.document_loader(path)
    logger.info("[%s] 총 %s개 문서 변환 완료", source.label, len(documents))

    if mode == "hybrid":
        logger.info("[%s] 하이브리드 컬렉션 생성/로드 중...", source.label)
        collection_name = get_or_create_hybrid_collection(collection_name=source.hybrid_collection_name)
        logger.info("[%s] 임베딩 & 하이브리드 인덱싱 시작...", source.label)
        return index_documents_hybrid(documents, collection_name=collection_name)

    logger.info("[%s] 컬렉션 생성/로드 중...", source.label)
    collection_name = get_or_create_collection(collection_name=source.collection_name)
    logger.info("[%s] 임베딩 & 인덱싱 시작...", source.label)
    return index_documents(documents, collection_name=collection_name)


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
        "--source",
        choices=["family_card", "library", "all"],
        default="family_card",
        help="인덱싱할 데이터 소스",
    )
    parser.add_argument(
        "--json-path",
        type=str,
        default=str(DEFAULT_FAMILY_CARD_JSON_PATH),
        help="family_card 전용 JSON 데이터 파일 경로",
    )
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")

    mode_label = "Hybrid" if args.mode == "hybrid" else "Naive RAG"
    logger.info("=== %s 인덱싱 시작 ===", mode_label)
    start = time.time()

    if args.source == "all":
        indexed_count = 0
        for source in get_all_source_configs():
            indexed_count += run_index_for_source(source.source_id, args.mode)
    elif args.source == "family_card" and args.json_path != str(DEFAULT_FAMILY_CARD_JSON_PATH):
        if args.mode == "hybrid":
            indexed_count = run_index_hybrid(args.json_path)
        else:
            indexed_count = run_index(args.json_path)
    else:
        indexed_count = run_index_for_source(args.source, args.mode)

    elapsed = time.time() - start
    logger.info("=== 인덱싱 완료: %s건, %.1f초 소요 ===", indexed_count, elapsed)


if __name__ == "__main__":
    main()
