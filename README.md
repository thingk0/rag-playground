# rag-playground

한국 공공데이터 API를 활용한 RAG 파이프라인 실험 프로젝트.

> "AI에게 얼마나 좋은 컨텍스트를 줄 수 있느냐" — 이것이 AI 에이전트 개발의 핵심이다.

## 프로젝트 목적

RAG(Retrieval-Augmented Generation) 기술을 **Naive RAG부터 Agentic RAG까지** 단계적으로 실험하며 실전 감각을 익히기 위한 플레이그라운드. 총 8개의 퀘스트를 통해 검색 품질을 점진적으로 개선하고 정량 평가까지 완성한 과정을 기록했다.

## 기술 스택

| 구분 | 선택 |
|------|------|
| Vector DB | Qdrant Cloud (Dense + Sparse BM25) |
| Embedding | OpenAI `text-embedding-3-small` (1536차원) |
| LLM | OpenAI `gpt-5.4-mini` |
| Re-ranker | Novita.ai `BGE-reranker-v2-m3` |
| Query Rewriter | OpenAI `gpt-5.4-mini` (HyDE, Multi-Query) |
| Language | Python 3.13+ (uv + src layout) |
| Lint | Ruff |

## 데이터 소스

[공공데이터포털](https://www.data.go.kr/) API 및 스냅샷 데이터 활용.

| 소스 | 설명 | 수집 방식 |
|------|------|-----------|
| `family_card` | 부산광역시 가족사랑카드 참여업체 현황 | 공공데이터포털 API (2,667건) |
| `library` | 부산광역시 도서관 정보 | 스냅샷 JSON (API 403 이슈로 대체) |

## 실험 로드맵

7개 퀘스트를 통해 RAG 파이프라인을 단계적으로 완성했다.

```
Quest 1  프로젝트 환경 세팅        ✅  uv + src layout + .gitignore
Quest 2  공공데이터 API 연동        ✅  가족사랑카드 API 수집 파이프라인
Quest 3  Naive RAG                  ✅  ChromaDB → 벡터 검색 → LLM 응답
Quest 4  Hybrid Search              ✅  BM25 + Dense 결합
Quest 5  Re-ranking                 ✅  BGE-reranker-v2-m3 도입
Quest 6  Query Rewriting            ✅  HyDE, Multi-Query 기법
Quest 7  Agentic RAG                ✅  자율 검색 에이전트
Quest 8  정량 평가                  ✅  NDCG / MRR / Precision@K
```

### Quest 7 핵심 기능

최종 퀘스트에서 에이전트가 스스로 검색 전략을 판단하는 자율 RAG 시스템을 완성했다.

- **Planner**: 질의 모호함 분석 → `rerank` / `hyde_rerank` / `multi_rerank` 전략 선택
- **Multi-source routing**: 키워드 휴리스틱으로 `family_card` / `library` 소스 라우팅
- **Fallback**: relevance score 기반 → 단일 소스에서 멀티 소스로 자동 확장
- **Trace**: 실행 단계별 계획/결과 추적

## 아키텍처

```
src/rag_playground/
├── adapters/                  # 외부 시스템 어댑터
│   ├── data_go_kr/            #   공공데이터포털 API 클라이언트
│   ├── llm/                   #   OpenAI Chat 어댑터
│   ├── query_rewriter/        #   HyDE / Multi-Query 재작성기
│   ├── reranker/              #   Novita.ai BGE-reranker 어댑터
│   └── vectorstore/           #   Qdrant 클라이언트
├── application/               # 실행 진입점
│   ├── agentic.py             #   Quest 7: Agentic RAG CLI
│   ├── answer.py              #   LLM 응답 생성
│   ├── compare.py             #   검색 모드 비교 스크립트
│   ├── index.py               #   Hybrid 인덱싱
│   ├── ingest.py              #   데이터 수집
│   └── sources.py             #   소스 라우팅 설정
├── app/
│   └── cli.py                 #   대화형 CLI (7개 모드)
├── config/
│   └── settings.py            #   환경 설정
├── data/                      #   데이터 수집 로직
├── domain/                    #   도메인 모델
│   ├── agent.py               #   Agentic RAG 계획/실행 모델
│   └── document.py            #   문서 모델
├── rag/                       #   RAG 코어 로직
│   ├── chain.py               #   검색 → 리랭크 → 응답 체인
│   ├── chunker.py             #   텍스트 청킹
│   ├── index.py               #   인덱싱 로직
│   └── vectorstore.py         #   벡터스토어 인터페이스
└── main.py                    #   엔트리포인트
```

## 실험 결과

### 정량 평가 (Quest 8)

10개 질의 × 7개 모드로 NDCG@5 / MRR / Precision@5를 측정했다.

| 모드 | NDCG@5 | MRR | P@5 | 평균 응답 |
|------|--------|-----|-----|-----------|
| Naive | 0.5237 | 0.4524 | 0.5143 | 1,454ms |
| BM25 | 0.2952 | 0.3000 | 0.1600 | 872ms |
| Hybrid | 0.5211 | 0.5333 | 0.3400 | 1,414ms |
| Rerank | 0.7686 | 0.7500 | 0.6600 | 2,259ms |
| **HyDE+Rerank** | **0.8177** | **0.7833** | 0.4600 | 3,714ms |
| Multi-Query+Rerank | 0.7963 | 0.7500 | **0.6000** | 3,518ms |
| Agentic | 0.7316 | 0.7000 | 0.6200 | 8,085ms |

**인사이트:**

- **HyDE+Rerank**가 NDCG(0.82)·MRR(0.78) 최고 — 가상 문서로 검색 후 재정렬이 가장 효과적
- **Rerank**가 P@5(0.66) 최고 — 비용 대비 정밀도 최강
- **BM25**는 순수 키워드 매칭 한계로 NDCG 0.30에 그침
- **Agentic**은 멀티 소스 라우팅이 유효하지만 응답 비용이 2배(8초), 단일 소스 질의에선 Rerank 대비 열세
- **q05(아이스크림), q06(주말 나들이)**는 전 모드 0점 — 데이터 자체에 관련 정보 없음

### 질의별 정성 비교

부산 가족사랑카드 데이터(2,667건)로 대표 모드를 비교했다.

| 질의 | Naive | BM25 | Hybrid | **Rerank** |
|------|-------|------|--------|------------|
| "부산진구 맛집" | 지역 오류(금정구 1위) | 미용실·철물점 혼입 | 미용실 혼입 | **음식점만 5건** |
| "해운대구에서 양식 할인" | 미용실 혼입 | 병원·태권도 혼입 | 병원 혼입 | **해운대구 식당 중심** |
| "미용실 저렴한 곳" | 미용실 반환 | **미용실 정확 반환** | 미용실 반환 | relevance 낮음(0.19) |
| "동래구 목욕탕" | 목욕탕 반환(지역 혼입) | 헤어샵 혼입 | 헤어샵 혼입 | **동래구 목욕탕 3건(0.99/0.98/0.91)** |
| "아이스크림 가게" | 어린이집 반환 | 결과 없음 | 어린이집 반환 | 결과 없음(relevance 0.003) |

## 시작하기

### 사전 준비

```bash
# Python 3.13+, uv 설치 필요
curl -LsSf https://astral.sh/uv/install.sh | sh
```

`.env.example`을 복사해 `.env`를 생성하고 API 키를 설정한다.

```bash
cp .env.example .env
```

필요한 키: `OPENAI_API_KEY`, `QDRANT_URL`, `QDRANT_API_KEY`, `NOVITA_API_KEY`, `DATA_GO_KR_API_KEY`

### 실행

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

# 대화형 CLI (7개 모드 선택)
uv run python -m rag_playground.app.cli
```

## 테스트

```bash
uv run ruff check src tests
uv run python -m unittest
```

## 회고

7개 퀘스트를 거치며 배운 핵심 교훈:

1. **Naive RAG는 출발점이다** — 벡터 검색만으로는 의미적 오류가 빈번하다
2. **Hybrid Search는 저비용 고효율** — 키워드 매칭과 의미 검색의 상호보완이 강력하다
3. **Re-ranker는 게임 체인저** — relevance score로 결과 신뢰도까지 판별 가능하다
4. **Query Rewriting은 쿼리에 따라 선택적으로** — HyDE와 Multi-Query 각각 장단점이 뚜렷하다
5. **Agentic RAG는 자율성과 비용의 트레이드오프** — Planner가 전략을 결정하지만 LLM 호출이 추가된다
6. **데이터가 전부다** — 어떤 검색 기법도 데이터에 없는 정보를 만들어낼 수 없다

## 라이선스

MIT

## 블로그

프로젝트 진행 과정을 날것 그대로 기록한다.

- [공공데이터로 RAG 뚝딱거려보기 - 프롤로그](https://www.thingk0.website/blog/2026-02-19-public-data-rag-prologue/)
