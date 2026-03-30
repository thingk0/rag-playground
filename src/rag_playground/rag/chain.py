"""구버전 chain 경로 호환 래퍼."""

from rag_playground.adapters.llm.openai_chat import LLM_MODEL, SYSTEM_PROMPT, build_prompt, generate_answer

__all__ = ["LLM_MODEL", "SYSTEM_PROMPT", "build_prompt", "generate_answer"]
