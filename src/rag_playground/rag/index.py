"""구버전 인덱싱 경로 호환 래퍼."""

from rag_playground.application.index import main, run_index

__all__ = ["run_index", "main"]

if __name__ == "__main__":
    main()
