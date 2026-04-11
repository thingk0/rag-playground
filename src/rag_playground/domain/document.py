"""RAG 도메인 문서 모델."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import orjson


@dataclass(slots=True)
class Document:
    """벡터스토어에 저장할 단일 문서."""

    page_content: str
    metadata: dict[str, Any] = field(default_factory=dict)


def shop_to_text(shop: dict[str, Any]) -> str:
    """업체 레코드를 자연어 문서로 변환한다."""
    lines = [
        f"[{shop.get('district', '')} / {shop.get('category', '')}] {shop.get('shop_name', '')}",
        f"주소: {shop.get('address', '')}",
    ]
    if shop.get("phone"):
        lines.append(f"연락처: {shop['phone']}")
    if shop.get("benefit"):
        lines.append(f"혜택: {shop['benefit']}")
    return "\n".join(lines)


def shop_to_document(shop: dict[str, Any]) -> Document:
    """업체 레코드 하나를 Document로 만든다."""
    metadata = {
        "title": shop.get("shop_name", ""),
        "shop_name": shop.get("shop_name", ""),
        "district": shop.get("district", ""),
        "category": shop.get("category", ""),
        "address": shop.get("address", ""),
        "phone": shop.get("phone", ""),
        "benefit": shop.get("benefit", ""),
        "summary": shop.get("benefit", ""),
        "source": shop.get("source", "부산광역시_가족사랑카드 참여업체 현황"),
        "source_id": "family_card",
        "source_label": "가족사랑카드",
    }
    return Document(page_content=shop_to_text(shop), metadata=metadata)


def library_to_text(library: dict[str, Any]) -> str:
    """도서관 레코드를 자연어 문서로 변환한다."""
    benefit = library.get("benefit", "열람 및 문화 프로그램 이용 가능")
    lines = [
        f"[{library.get('district', '')} / {library.get('category', '도서관')}] {library.get('name', '')}",
        f"주소: {library.get('address', '')}",
    ]
    if library.get("phone"):
        lines.append(f"연락처: {library['phone']}")
    if library.get("homepage"):
        lines.append(f"홈페이지: {library['homepage']}")
    lines.append(f"안내: {benefit}")
    lines.append("추천 상황: 가족과 함께 방문하거나 아이들이 책을 읽고 머물기 좋은 부산 공공 문화 공간")
    return "\n".join(lines)


def library_to_document(library: dict[str, Any]) -> Document:
    """도서관 레코드 하나를 Document로 만든다."""
    metadata = {
        "title": library.get("name", ""),
        "name": library.get("name", ""),
        "district": library.get("district", ""),
        "category": library.get("category", "도서관"),
        "address": library.get("address", ""),
        "phone": library.get("phone", ""),
        "homepage": library.get("homepage", ""),
        "benefit": library.get("benefit", "열람 및 문화 프로그램 이용 가능"),
        "summary": library.get("benefit", "열람 및 문화 프로그램 이용 가능"),
        "source": library.get("source", "부산광역시_도서관 정보"),
        "source_id": "library",
        "source_label": "도서관",
    }
    return Document(page_content=library_to_text(library), metadata=metadata)


def _load_json_records(json_path: str | Path) -> list[dict[str, Any]]:
    """JSON 파일을 읽어 레코드 리스트로 반환한다."""
    path = Path(json_path)
    with path.open("rb") as file:
        return orjson.loads(file.read())


def load_family_card_documents(json_path: str | Path) -> list[Document]:
    """가족사랑카드 JSON 파일을 읽어서 Document 리스트로 변환한다."""
    shops = _load_json_records(json_path)
    return [shop_to_document(shop) for shop in shops]


def load_library_documents(json_path: str | Path) -> list[Document]:
    """도서관 JSON 파일을 읽어서 Document 리스트로 변환한다."""
    libraries = _load_json_records(json_path)
    return [library_to_document(library) for library in libraries]


def load_shop_documents(json_path: str | Path) -> list[Document]:
    """구버전 호환용 alias."""
    return load_family_card_documents(json_path)
