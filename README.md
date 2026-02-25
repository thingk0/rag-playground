# rag-playground

한국 공공데이터 API를 활용한 RAG 파이프라인 실험 프로젝트.

> "AI에게 얼마나 좋은 컨텍스트를 줄 수 있느냐" — 이것이 AI 에이전트 개발의 핵심이다.

## 프로젝트 목적

RAG(Retrieval-Augmented Generation) 기술을 Naive RAG부터 Agentic RAG까지 단계적으로 실험하며 실전 감각을 익히기 위한 플레이그라운드.

## 기술 스택

| 구분 | 선택 |
|------|------|
| Vector DB | ChromaDB |
| LLM | 오픈소스 소규모 모델 (RAG 효과를 명확히 관찰하기 위함) |
| Framework | LangChain / LlamaIndex (검토 예정) |

## 데이터 소스

[공공데이터포털](https://www.data.go.kr/) API 활용

- 무료 제공, 공식 API로 법적 문제 없음
- 날씨, 교통, 의료, 부동산 등 다양한 도메인
- 실전적인 데이터로 의미 있는 RAG 실험 가능

## 실험 로드맵

1. **Naive RAG** — 기본 파이프라인 구축
2. **Hybrid Search** — 키워드 + 벡터 검색 결합
3. **Re-ranking** — 검색 결과 재정렬
4. **Query Rewriting** — 질의 최적화
5. **Agentic RAG** — 에이전트 기반 자율 검색

추가로 다양한 임베딩 모델 및 벡터 DB 비교 실험도 진행 예정.

## 블로그

프로젝트 진행 과정을 날것 그대로 기록합니다.

- [공공데이터로 RAG 뚝딱거려보기 - 프롤로그](https://www.thingk0.website/blog/2026-02-19-public-data-rag-prologue/)
