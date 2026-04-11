"""Quest 7용 데이터 소스 카탈로그."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Callable

from rag_playground.config import DEFAULT_FAMILY_CARD_JSON_PATH, DEFAULT_LIBRARY_JSON_PATH
from rag_playground.domain.document import load_family_card_documents, load_library_documents

DocumentLoader = Callable[[str | Path], list]


@dataclass(frozen=True, slots=True)
class SourceConfig:
    """개별 검색 데이터 소스 설정."""

    source_id: str
    label: str
    description: str
    collection_name: str
    hybrid_collection_name: str
    default_json_path: Path
    document_loader: DocumentLoader
    domain_hint: str
    answer_hint: str
    query_keywords: tuple[str, ...]


SOURCE_CATALOG: dict[str, SourceConfig] = {
    "family_card": SourceConfig(
        source_id="family_card",
        label="가족사랑카드",
        description="부산광역시 가족사랑카드 참여업체 정보",
        collection_name="family_card_shops",
        hybrid_collection_name="family_card_shops_hybrid",
        default_json_path=DEFAULT_FAMILY_CARD_JSON_PATH,
        document_loader=load_family_card_documents,
        domain_hint="부산광역시 가족사랑카드 참여업체 데이터베이스",
        answer_hint="가족사랑카드 참여업체 정보",
        query_keywords=(
            "할인",
            "가맹점",
            "가족사랑카드",
            "맛집",
            "음식점",
            "식당",
            "미용실",
            "목욕탕",
            "학원",
            "병원",
            "업체",
            "가게",
        ),
    ),
    "library": SourceConfig(
        source_id="library",
        label="도서관",
        description="부산광역시 공공 도서관 정보",
        collection_name="busan_libraries",
        hybrid_collection_name="busan_libraries_hybrid",
        default_json_path=DEFAULT_LIBRARY_JSON_PATH,
        document_loader=load_library_documents,
        domain_hint="부산광역시 도서관 정보 데이터베이스",
        answer_hint="부산 도서관 정보",
        query_keywords=(
            "도서관",
            "책",
            "열람실",
            "공부",
            "독서",
            "문화",
            "강좌",
            "아이",
            "아이들",
            "주말",
            "가볼",
            "갈 만",
        ),
    ),
}


def get_source_config(source_id: str) -> SourceConfig:
    """소스 설정을 반환한다."""
    try:
        return SOURCE_CATALOG[source_id]
    except KeyError as error:
        raise ValueError(f"알 수 없는 source_id입니다: {source_id}") from error


def get_all_source_configs() -> list[SourceConfig]:
    """등록된 모든 소스를 반환한다."""
    return list(SOURCE_CATALOG.values())
