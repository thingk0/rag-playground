"""도메인 모델."""

from rag_playground.domain.document import Document, load_shop_documents, shop_to_document, shop_to_text

__all__ = ["Document", "shop_to_text", "shop_to_document", "load_shop_documents"]
