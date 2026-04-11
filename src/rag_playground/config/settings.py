"""환경 설정 로더."""

from __future__ import annotations

import os
from pathlib import Path

from dotenv import load_dotenv

PROJECT_ROOT = Path(__file__).resolve().parents[3]
load_dotenv(PROJECT_ROOT / ".env")

RAW_DATA_DIR = PROJECT_ROOT / "data" / "raw"
DEFAULT_FAMILY_CARD_JSON_PATH = RAW_DATA_DIR / "family_card_shops.json"
DEFAULT_LIBRARY_JSON_PATH = RAW_DATA_DIR / "busan_libraries.json"

DATA_GO_KR_API_KEY: str = os.getenv("DATA_GO_KR_API_KEY", "")
OPENAI_API_KEY: str = os.getenv("OPENAI_API_KEY", "")

QDRANT_URL: str = os.getenv("QDRANT_URL", "")
QDRANT_API_KEY: str = os.getenv("QDRANT_API_KEY", "")

NOVITA_API_KEY: str = os.getenv("NOVITA_API_KEY", "")
