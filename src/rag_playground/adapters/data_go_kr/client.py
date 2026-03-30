"""부산 가족사랑카드 API 클라이언트."""

from __future__ import annotations

import datetime
from typing import Any

import httpx

from rag_playground.config import DATA_GO_KR_API_KEY

FAMILY_CARD_API_URL = "http://apis.data.go.kr/6260000/BusanFmlyLvcrInfoService/getFmlyLvcrInfo"


async def fetch_card_shops(
    page_no: int = 1,
    num_of_rows: int = 100,
    cp_compname: str = "",
    cp_hgu: str = "",
    cp_class: str = "",
) -> dict[str, Any]:
    """부산광역시 가족사랑카드 참여업체 현황을 조회한다."""
    if not DATA_GO_KR_API_KEY:
        raise ValueError("DATA_GO_KR_API_KEY가 설정되지 않았습니다. .env 파일을 확인해주세요.")

    params = {
        "serviceKey": DATA_GO_KR_API_KEY,
        "pageNo": str(page_no),
        "numOfRows": str(num_of_rows),
        "resultType": "json",
    }
    if cp_compname:
        params["cpCompname"] = cp_compname
    if cp_hgu:
        params["cpHgu"] = cp_hgu
    if cp_class:
        params["cpClass"] = cp_class

    async with httpx.AsyncClient() as client:
        response = await client.get(FAMILY_CARD_API_URL, params=params)
        response.raise_for_status()
        if "application/json" not in response.headers.get("content-type", "") and response.text.startswith("<"):
            raise ValueError(f"API 호출 실패 (XML 응답 반환): {response.text}")
        return response.json()


def parse_shops_data(raw_data: dict[str, Any]) -> list[dict[str, Any]]:
    """업체 JSON 데이터를 RAG 친화 포맷으로 변환한다."""
    try:
        header = raw_data["response"]["header"]
        if header["resultCode"] != "00":
            raise ValueError(f"API Error: {header['resultMsg']}")

        items = raw_data["response"]["body"]["items"]["item"]
        if isinstance(items, dict):
            items = [items]
    except KeyError as error:
        raise ValueError(f"예상치 못한 응답 형식입니다: {error}\nRaw Data: {raw_data}") from error

    now_iso = datetime.datetime.now(datetime.timezone(datetime.timedelta(hours=9))).isoformat()
    parsed_items: list[dict[str, Any]] = []
    for item in items:
        parsed_items.append(
            {
                "shop_name": item.get("cpCompname", ""),
                "address": item.get("cpAddr", ""),
                "district": item.get("cpHgu", ""),
                "benefit": item.get("cpWoo", "") or item.get("cpContent", ""),
                "category": item.get("cpClass", ""),
                "phone": item.get("cpTel", ""),
                "source": "부산광역시_가족사랑카드 참여업체 현황",
                "collected_at": now_iso,
            }
        )

    return parsed_items
