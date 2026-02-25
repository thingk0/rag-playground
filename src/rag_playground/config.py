"""환경 설정 — .env에서 API 키 등을 로드."""

from __future__ import annotations

import os
from pathlib import Path

from dotenv import load_dotenv

# 프로젝트 루트의 .env 파일 로드
_PROJECT_ROOT = Path(__file__).resolve().parents[2]
load_dotenv(_PROJECT_ROOT / ".env")

# 공공데이터포털 API 키
DATA_GO_KR_API_KEY: str = os.getenv("DATA_GO_KR_API_KEY", "")
