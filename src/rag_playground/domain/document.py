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
        "shop_name": shop.get("shop_name", ""),
        "district": shop.get("district", ""),
        "category": shop.get("category", ""),
        "address": shop.get("address", ""),
        "phone": shop.get("phone", ""),
        "benefit": shop.get("benefit", ""),
    }
    return Document(page_content=shop_to_text(shop), metadata=metadata)


def load_shop_documents(json_path: str | Path) -> list[Document]:
    """JSON 파일을 읽어서 Document 리스트로 변환한다."""
    path = Path(json_path)
    with path.open("rb") as file:
        shops: list[dict[str, Any]] = orjson.loads(file.read())
    return [shop_to_document(shop) for shop in shops]
