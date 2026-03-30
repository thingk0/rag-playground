"""구버전 수집 CLI 경로 호환 래퍼."""

from rag_playground.application.ingest import main

__all__ = ["main"]

if __name__ == "__main__":
    main()
