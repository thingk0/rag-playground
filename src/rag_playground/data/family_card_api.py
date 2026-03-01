from typing import Any, Dict, List
import httpx

from rag_playground.config import DATA_GO_KR_API_KEY

FAMILY_CARD_API_URL = "http://apis.data.go.kr/6260000/BusanFmlyLvcrInfoService/getFmlyLvcrInfo"

async def fetch_card_shops(
    page_no: int = 1, 
    num_of_rows: int = 100,
    cp_compname: str = "",
    cp_hgu: str = "",
    cp_class: str = ""
) -> Dict[str, Any]:
    """부산광역시 가족사랑카드 참여업체 현황을 조회합니다.
    
    Args:
        page_no: 페이지 번호 (기본값: 1)
        num_of_rows: 한 페이지결과 수 (기본값: 100)
        cp_compname: 참여업체명 검색어 (선택)
        cp_hgu: 지역구 검색어 (선택)
        cp_class: 업종명 검색어 (선택)
        
    Returns:
        JSON 응답 딕셔너리
    """
    if not DATA_GO_KR_API_KEY:
        raise ValueError("DATA_GO_KR_API_KEY가 설정되지 않았습니다. .env 파일을 확인해주세요.")
    
    # httpx.get()의 params 인자로 넘길 때는 기본적으로 url encoding 이 일어나므로, 
    # 공공데이터포털의 "Decoding" (디코딩) 된 키를 사용해야 + 나 = 기호가 중복 인코딩되지 않습니다.
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
        
        # 간혹 에러 메시지를 XML로 반환하는 경우가 있음
        if "application/json" not in response.headers.get("content-type", "") and response.text.startswith("<"):
            raise ValueError(f"API 호출 실패 (XML 응답 반환): {response.text}")
            
        return response.json()

def parse_shops_data(raw_data: Dict[str, Any]) -> List[Dict[str, Any]]:
    """업체 JSON 데이터를 정제하여 활용하기 좋은 형태로 변환합니다."""
    try:
        header = raw_data["response"]["header"]
        if header["resultCode"] != "00":
            raise ValueError(f"API Error: {header['resultMsg']}")

        items = raw_data["response"]["body"]["items"]["item"]
        
        # items가 단일 객체일 경우 리스트로 감싸기
        if isinstance(items, dict):
            items = [items]
            
    except KeyError as e:
        raise ValueError(f"예상치 못한 응답 형식입니다: {e}\nRaw Data: {raw_data}")

    import datetime
    now_iso = datetime.datetime.now(datetime.timezone(datetime.timedelta(hours=9))).isoformat()

    parsed_items = []
    for item in items:
        # RAG 검색에 용이하도록 설계 포맷
        parsed_items.append({
            "shop_name": item.get("cpCompname", ""),                       # 가게명
            "address": item.get("cpAddr", ""),                             # 주소
            "district": item.get("cpHgu", ""),                             # 구/군
            "benefit": item.get("cpWoo", "") or item.get("cpContent", ""), # 할인/우대내용 (샘플엔 cpWoo 로 들어옴)
            "category": item.get("cpClass", ""),                           # 업종명
            "phone": item.get("cpTel", ""),                                # 연락처
            "source": "부산광역시_가족사랑카드 참여업체 현황",                     # 출처           
            "collected_at": now_iso                                        # 수집 시간
        })

    return parsed_items
