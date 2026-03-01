import asyncio
import logging
import orjson
from pathlib import Path

from rag_playground.data.family_card_api import fetch_card_shops, parse_shops_data

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

# 수집된 데이터를 저장할 경로
_PROJECT_ROOT = Path(__file__).resolve().parents[3]
RAW_DATA_DIR = _PROJECT_ROOT / "data" / "raw"

async def main():
    logger.info("부산광역시 가족사랑카드 참여업체 현황 데이터 수집을 시작합니다.")
    
    all_shops = []
    page_no = 1
    num_of_rows = 100
    
    try:
        while True:
            logger.info(f"Page {page_no} 데이터 요청 중...")
            raw_data = await fetch_card_shops(page_no=page_no, num_of_rows=num_of_rows)
            
            try:
                # 응답에 아이템이 없으면 KeyError 거나 빈 리스트
                parsed_items = parse_shops_data(raw_data)
                
                if not parsed_items:
                    break
                    
                all_shops.extend(parsed_items)
                
                # 받아온 개수가 num_of_rows보다 적으면 마지막 페이지
                if len(parsed_items) < num_of_rows:
                    break
                    
                page_no += 1
                
                # 공공데이터 API Limit & 서버 부하 방지
                await asyncio.sleep(0.5)
                
            except ValueError as e:
                logger.warning(f"데이터 파싱 오류 또는 마지막 페이지 도달: {e}")
                break

        # 파일 저장 준비
        RAW_DATA_DIR.mkdir(parents=True, exist_ok=True)
        filename = RAW_DATA_DIR / "family_card_shops.json"
        
        with open(filename, "wb") as f:
            f.write(orjson.dumps(all_shops, option=orjson.OPT_INDENT_2))
            
        logger.info(f"데이터 수집 완료! 총 {len(all_shops)}개의 업체 정보를 {filename}에 저장했습니다.")
        
    except Exception as e:
        logger.error(f"데이터 수집 중 오류가 발생했습니다: {e}")

if __name__ == "__main__":
    asyncio.run(main())
