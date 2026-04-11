"""OpenAI 기반 질의 재작성 어댑터."""

from __future__ import annotations

from openai import OpenAI

from rag_playground.config import OPENAI_API_KEY

LLM_MODEL = "gpt-5.4-mini"

HYDE_SYSTEM_PROMPT_TEMPLATE = """\
당신은 부산광역시 가족사랑카드 참여업체 데이터베이스의 문서 생성기입니다.
사용자의 질문을 보고, 해당 질문에 대한 답변이 될 수 있는 가상의 업체 정보 문서를 하나 작성해주세요.
실제 데이터와 유사한 형식으로 작성하되, 구체적인 업체명, 주소, 혜택 정보를 포함해주세요.
예시 형식: [구 / 업종] 업체명\n주소: ...\n혜택: ...
짧고 간결하게 작성하세요."""

MULTI_QUERY_SYSTEM_PROMPT_TEMPLATE = """\
당신은 검색 질의 확장 전문가입니다.
사용자의 원래 질문을 보고, 같은 의도를 가진 다른 표현의 검색 질의를 3개 생성해주세요.
각 질의는 한 줄에 하나씩 작성하세요. 번호나 기호 없이 질의만 작성하세요.
부산 가족사랑카드 참여업체 검색에 적합하도록 작성하세요."""


def _build_hyde_system_prompt(domain_context: str) -> str:
    """HyDE 시스템 프롬프트를 도메인별로 생성한다."""
    return (
        f"당신은 {domain_context}의 문서 생성기입니다.\n"
        "사용자의 질문을 보고, 해당 질문에 대한 답변이 될 수 있는 가상의 문서 하나를 작성해주세요.\n"
        "실제 데이터와 유사한 형식으로 작성하되, 이름, 주소, 핵심 정보를 포함해주세요.\n"
        "짧고 간결하게 작성하세요."
    )


def _build_multi_query_system_prompt(domain_context: str, n: int) -> str:
    """Multi-Query 시스템 프롬프트를 도메인별로 생성한다."""
    return (
        "당신은 검색 질의 확장 전문가입니다.\n"
        "사용자의 원래 질문을 보고, 같은 의도를 가진 다른 표현의 검색 질의를 생성해주세요.\n"
        f"각 질의는 한 줄에 하나씩, 총 {n}개 작성하세요. 번호나 기호 없이 질의만 작성하세요.\n"
        f"{domain_context} 검색에 적합하도록 작성하세요."
    )


def generate_hypothetical_document(
    query: str,
    domain_context: str = "부산광역시 가족사랑카드 참여업체 데이터베이스",
) -> str:
    """HyDE: 질의에 대한 가상의 답변 문서를 생성한다."""
    if not OPENAI_API_KEY:
        raise ValueError("OPENAI_API_KEY가 설정되지 않았습니다.")

    client = OpenAI(api_key=OPENAI_API_KEY)
    response = client.chat.completions.create(
        model=LLM_MODEL,
        messages=[
            {"role": "system", "content": _build_hyde_system_prompt(domain_context)},
            {"role": "user", "content": query},
        ],
        temperature=0.7,
        max_completion_tokens=256,
    )
    return response.choices[0].message.content or ""


def generate_multi_queries(
    query: str,
    n: int = 3,
    domain_context: str = "부산광역시 가족사랑카드 참여업체 데이터베이스",
) -> list[str]:
    """Multi-Query: 질의의 대안 버전을 n개 생성한다."""
    if not OPENAI_API_KEY:
        raise ValueError("OPENAI_API_KEY가 설정되지 않았습니다.")

    client = OpenAI(api_key=OPENAI_API_KEY)
    response = client.chat.completions.create(
        model=LLM_MODEL,
        messages=[
            {"role": "system", "content": _build_multi_query_system_prompt(domain_context, n)},
            {"role": "user", "content": f"원래 질문: {query}\n{n}개의 대안 질의를 생성해주세요."},
        ],
        temperature=0.7,
        max_completion_tokens=256,
    )
    raw = response.choices[0].message.content or ""
    queries = [line.strip() for line in raw.splitlines() if line.strip()]
    return queries[:n]
