"""구버전 rag 패키지 호환 계층."""

from rag_playground.adapters.llm.openai_chat import generate_answer
from rag_playground.adapters.vectorstore.qdrant import get_or_create_collection, index_documents, search
from rag_playground.domain.document import Document, load_shop_documents

__all__ = [
    "Document",
    "load_shop_documents",
    "get_or_create_collection",
    "index_documents",
    "search",
    "generate_answer",
]
