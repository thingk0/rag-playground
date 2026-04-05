# rag-playground

한국 공공데이터 API를 활용한 RAG 파이프라인 실험 프로젝트.

> "AI에게 얼마나 좋은 컨텍스트를 줄 수 있느냐" — 이것이 AI 에이전트 개발의 핵심이다.

## 프로젝트 목적

RAG(Retrieval-Augmented Generation) 기술을 Naive RAG부터 Agentic RAG까지 단계적으로 실험하며 실전 감각을 익히기 위한 플레이그라운드.

## 기술 스택

| 구분 | 선택 |
|------|------|
| Vector DB | Qdrant Cloud (Dense + Sparse BM25) |
| Embedding | OpenAI `text-embedding-3-small` (1536차원) |
| LLM | OpenAI `gpt-4o-mini` |
| Re-ranker | Novita.ai `BGE-reranker-v2-m3` |
| Language | Python 3.13+ (uv + src layout) |

## 데이터 소스

[공공데이터포털](https://www.data.go.kr/) API 활용

- 무료 제공, 공식 API로 법적 문제 없음
- 날씨, 교통, 의료, 부동산 등 다양한 도메인
- 실전적인 데이터로 의미 있는 RAG 실험 가능

## 실험 로드맵

1. **Naive RAG** — 기본 파이프라인 구축 ✅
2. **Hybrid Search** — 키워드 + 벡터 검색 결합 ✅
3. **Re-ranking** — Hybrid(20개) → BGE-reranker-v2-m3 재정렬 ✅
4. **Query Rewriting** — 질의 최적화
5. **Agentic RAG** — 에이전트 기반 자율 검색

추가로 다양한 임베딩 모델 및 벡터 DB 비교 실험도 진행 예정.

## 실험 결과

부산 가족사랑카드 데이터(2,667건)로 4개 검색 모드 비교.

| 질의 | Naive | BM25 | Hybrid | **Rerank** |
|------|-------|------|--------|------------|
| "부산진구 맛집" | 지역 오류(금정구 1위) | 미용실·철물점 혼입 | 미용실 혼입 | **음식점만 5건** |
| "해운대구에서 양식 할인" | 미용실 혼입 | 병원·태권도 혼입 | 병원 혼입 | **해운대구 식당 중심** |
| "미용실 저렴한 곳" | 미용실 반환 | **미용실 정확 반환** | 미용실 반환 | relevance 낮음(0.19) |
| "동래구 목욕탕" | 목욕탕 반환(지역 혼입) | 헤어샵 혼입 | 헤어샵 혼입 | **동래구 목욕탕 3건(0.99/0.98/0.91)** |
| "아이스크림 가게" | 어린이집 반환 | 결과 없음 | 어린이집 반환 | 결과 없음(relevance 0.003) |

**인사이트:**
- Rerank의 relevance score는 신뢰도 지표로 활용 가능 — 0.9+ 신뢰, 0.5 미만은 데이터 부재 가능성
- BM25는 단순 키워드 매칭에 강하지만 의미적 필터링 불가
- Rerank는 쿼리 의도(카테고리)를 가장 잘 반영하나 데이터에 없는 개념엔 모든 모드가 한계

## 실행 방법

```bash
# Naive RAG 인덱싱
uv run python -m rag_playground.application.index

# Hybrid 인덱싱 (Dense + BM25 Sparse)
uv run python -m rag_playground.application.index --mode hybrid

# 검색 모드 비교
uv run python -m rag_playground.application.compare

# 대화형 CLI (모드 선택: Naive / BM25 / Hybrid / Hybrid+Rerank)
uv run python -m rag_playground.app.cli
```

## 블로그

프로젝트 진행 과정을 날것 그대로 기록합니다.

- [공공데이터로 RAG 뚝딱거려보기 - 프롤로그](https://www.thingk0.website/blog/2026-02-19-public-data-rag-prologue/)
