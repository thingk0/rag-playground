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
| LLM | OpenAI `gpt-5.4-mini` |
| Re-ranker | Novita.ai `BGE-reranker-v2-m3` |
| Query Rewriter | OpenAI `gpt-5.4-mini` (HyDE, Multi-Query) |
| Language | Python 3.13+ (uv + src layout) |

## 데이터 소스

[공공데이터포털](https://www.data.go.kr/) API 및 스냅샷 데이터 활용

- `family_card`: 부산광역시 가족사랑카드 참여업체 현황 API
- `library`: 부산광역시 도서관 정보 스냅샷 JSON
- 도서관 OpenAPI는 계정 승인 이슈로 `403`이 발생해, Quest 7에서는 체크인된 스냅샷으로 멀티 소스를 구성

## 실험 로드맵

1. **Naive RAG** — 기본 파이프라인 구축 ✅
2. **Hybrid Search** — 키워드 + 벡터 검색 결합 ✅
3. **Re-ranking** — Hybrid(20개) → BGE-reranker-v2-m3 재정렬 ✅
4. **Query Rewriting** — 질의 최적화 ✅ (HyDE, Multi-Query)
5. **Agentic RAG** — 에이전트 기반 자율 검색 ✅

추가로 정량 평가(NDCG/MRR)와 더 다양한 데이터 소스 확장 실험을 진행 예정.

## Quest 7 핵심 기능

- 질의 모호함을 분석해 `rerank` / `multi_rerank` / `hyde_rerank`를 선택
- 소스별 키워드 휴리스틱으로 `family_card` / `library` 라우팅
- relevance score 기반 fallback
- fallback 시 단일 소스에서 멀티 소스로 확장
- 최종 답변은 항상 원본 질의를 기준으로 생성

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
# 데이터 수집
uv run python -m rag_playground.application.ingest --source family_card

# 도서관은 저장소에 포함된 스냅샷 JSON 사용
# 필요 시 API 시도: uv run python -m rag_playground.application.ingest --source library

# Hybrid 인덱싱 (Dense + BM25 Sparse)
uv run python -m rag_playground.application.index --mode hybrid --source all

# 검색 모드 비교
uv run python -m rag_playground.application.compare

# Quest 7 planner preview
uv run python -m rag_playground.application.agentic --query "이번 주말에 아이들이랑 갈 만한 데" --preview

# Quest 7 Agentic 실행
uv run python -m rag_playground.application.agentic --query "이번 주말에 아이들이랑 갈 만한 데"

# 대화형 CLI (모드 선택: Naive / BM25 / Hybrid / Rerank / HyDE+Rerank / Multi-Query+Rerank / Agentic)
uv run python -m rag_playground.app.cli
```

## 테스트

```bash
uv run ruff check src tests
uv run python -m unittest
```

## 블로그

프로젝트 진행 과정을 날것 그대로 기록합니다.

- [공공데이터로 RAG 뚝딱거려보기 - 프롤로그](https://www.thingk0.website/blog/2026-02-19-public-data-rag-prologue/)
