"""인덱싱 유스케이스."""

from __future__ import annotations

import logging
import time
from pathlib import Path

from rag_playground.adapters.vectorstore.qdrant import get_or_create_collection, index_documents
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


def main() -> None:
    """인덱싱 CLI 엔트리포인트."""
    logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
    logger.info("=== Naive RAG 인덱싱 시작 ===")
    start = time.time()
    indexed_count = run_index()
    elapsed = time.time() - start
    logger.info("=== 인덱싱 완료: %s건, %.1f초 소요 ===", indexed_count, elapsed)


if __name__ == "__main__":
    main()
