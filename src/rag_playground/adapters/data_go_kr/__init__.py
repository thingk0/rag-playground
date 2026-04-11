"""data.go.kr API 어댑터."""

from rag_playground.adapters.data_go_kr.client import (
    FAMILY_CARD_API_URL,
    fetch_card_shops,
    parse_shops_data,
)
from rag_playground.adapters.data_go_kr.library import (
    LIBRARY_API_URL,
    fetch_libraries,
    parse_library_data,
)

__all__ = [
    "FAMILY_CARD_API_URL",
    "LIBRARY_API_URL",
    "fetch_card_shops",
    "fetch_libraries",
    "parse_shops_data",
    "parse_library_data",
]
