"""data.go.kr API 어댑터."""

from rag_playground.adapters.data_go_kr.client import (
    FAMILY_CARD_API_URL,
    fetch_card_shops,
    parse_shops_data,
)

__all__ = ["FAMILY_CARD_API_URL", "fetch_card_shops", "parse_shops_data"]
