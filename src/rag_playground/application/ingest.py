"""수집 유스케이스."""

from __future__ import annotations

import asyncio
import logging
from pathlib import Path
from typing import Any

import orjson

from rag_playground.adapters.data_go_kr.client import fetch_card_shops, parse_shops_data
from rag_playground.adapters.data_go_kr.library import fetch_libraries, parse_library_data
from rag_playground.config import DEFAULT_FAMILY_CARD_JSON_PATH, DEFAULT_LIBRARY_JSON_PATH

logger = logging.getLogger(__name__)


async def collect_family_card_shops(
    page_size: int = 100,
    sleep_seconds: float = 0.5,
) -> list[dict[str, Any]]:
    """가족사랑카드 업체 데이터를 페이지 단위로 수집한다."""
    all_shops: list[dict[str, Any]] = []
    page_no = 1

    while True:
        logger.info("Page %s 데이터 요청 중...", page_no)
        raw_data = await fetch_card_shops(page_no=page_no, num_of_rows=page_size)

        try:
            parsed_items = parse_shops_data(raw_data)
        except ValueError as error:
            logger.warning("데이터 파싱 오류 또는 마지막 페이지 도달: %s", error)
            break

        if not parsed_items:
            break

        all_shops.extend(parsed_items)

        if len(parsed_items) < page_size:
            break

        page_no += 1
        await asyncio.sleep(sleep_seconds)

    return all_shops


async def collect_libraries(
    page_size: int = 100,
    sleep_seconds: float = 0.5,
) -> list[dict[str, Any]]:
    """도서관 데이터를 페이지 단위로 수집한다."""
    all_libraries: list[dict[str, Any]] = []
    page_no = 1

    while True:
        logger.info("Library page %s 데이터 요청 중...", page_no)
        raw_data = await fetch_libraries(page_no=page_no, num_of_rows=page_size)

        try:
            parsed_items = parse_library_data(raw_data)
        except ValueError as error:
            logger.warning("도서관 데이터 파싱 오류 또는 마지막 페이지 도달: %s", error)
            break

        if not parsed_items:
            break

        all_libraries.extend(parsed_items)

        if len(parsed_items) < page_size:
            break

        page_no += 1
        await asyncio.sleep(sleep_seconds)

    return all_libraries


def save_family_card_shops(
    shops: list[dict[str, Any]],
    output_path: str | Path = DEFAULT_FAMILY_CARD_JSON_PATH,
) -> Path:
    """수집 데이터를 JSON으로 저장한다."""
    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("wb") as file:
        file.write(orjson.dumps(shops, option=orjson.OPT_INDENT_2))
    return path


def save_libraries(
    libraries: list[dict[str, Any]],
    output_path: str | Path = DEFAULT_LIBRARY_JSON_PATH,
) -> Path:
    """도서관 데이터를 JSON으로 저장한다."""
    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("wb") as file:
        file.write(orjson.dumps(libraries, option=orjson.OPT_INDENT_2))
    return path


async def run_ingest(output_path: str | Path = DEFAULT_FAMILY_CARD_JSON_PATH) -> Path:
    """수집부터 파일 저장까지 실행한다."""
    logger.info("부산광역시 가족사랑카드 참여업체 현황 데이터 수집을 시작합니다.")
    shops = await collect_family_card_shops()
    saved_path = save_family_card_shops(shops, output_path=output_path)
    logger.info("데이터 수집 완료! 총 %s개 업체 정보를 %s에 저장했습니다.", len(shops), saved_path)
    return saved_path


async def run_library_ingest(output_path: str | Path = DEFAULT_LIBRARY_JSON_PATH) -> Path:
    """도서관 수집부터 파일 저장까지 실행한다."""
    logger.info("부산광역시 도서관 정보 데이터 수집을 시작합니다.")
    libraries = await collect_libraries()
    saved_path = save_libraries(libraries, output_path=output_path)
    logger.info("데이터 수집 완료! 총 %s개 도서관 정보를 %s에 저장했습니다.", len(libraries), saved_path)
    return saved_path


def main() -> None:
    """수집 CLI 엔트리포인트."""
    import argparse

    parser = argparse.ArgumentParser(description="공공데이터 수집")
    parser.add_argument(
        "--source",
        choices=["family_card", "library", "all"],
        default="family_card",
        help="수집할 데이터 소스",
    )
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
    if args.source == "all":
        asyncio.run(run_ingest())
        asyncio.run(run_library_ingest())
    elif args.source == "library":
        asyncio.run(run_library_ingest())
    else:
        asyncio.run(run_ingest())


if __name__ == "__main__":
    main()
