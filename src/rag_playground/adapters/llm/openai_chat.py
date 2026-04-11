"""OpenAI Chat 어댑터."""

from __future__ import annotations

from typing import Any

from openai import OpenAI

from rag_playground.config import OPENAI_API_KEY

LLM_MODEL = "gpt-5.4-mini"

SYSTEM_PROMPT = """\
당신은 부산광역시 생활 정보 안내 도우미입니다.
아래 검색된 정보를 바탕으로 사용자의 질문에 정확하게 답해주세요.
검색된 정보에 없는 내용은 "해당 정보를 찾을 수 없습니다"라고 답하세요.
서로 다른 데이터 소스의 정보가 함께 들어올 수 있으니, 출처와 성격을 혼동하지 마세요.
답변은 친절하고 간결하게 작성하세요."""


def build_prompt(query: str, retrieved_docs: list[dict[str, Any]]) -> str:
    """검색 결과를 컨텍스트로 조합한 유저 프롬프트를 만든다."""
    if not retrieved_docs:
        context = "(검색 결과 없음)"
    else:
        context_parts: list[str] = []
        for index, doc in enumerate(retrieved_docs, start=1):
            metadata = doc.get("metadata", {})
            source_label = metadata.get("source_label") or metadata.get("source") or "알 수 없음"
            context_parts.append(f"--- 검색 결과 {index} / 출처: {source_label} ---\n{doc['document']}")
        context = "\n\n".join(context_parts)

    return f"[검색된 정보]\n{context}\n\n[사용자 질문]\n{query}"


def generate_answer(query: str, retrieved_docs: list[dict[str, Any]]) -> str:
    """검색 결과를 기반으로 LLM 응답을 생성한다."""
    if not OPENAI_API_KEY:
        raise ValueError("OPENAI_API_KEY가 설정되지 않았습니다. .env 파일을 확인해주세요.")

    client = OpenAI(api_key=OPENAI_API_KEY)
    user_message = build_prompt(query, retrieved_docs)

    response = client.chat.completions.create(
        model=LLM_MODEL,
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": user_message},
        ],
        temperature=0.3,
        max_completion_tokens=1024,
    )

    return response.choices[0].message.content or ""
