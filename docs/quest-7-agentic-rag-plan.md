# Quest 7: Agentic RAG 구현 요약

## 목표

Quest 7의 목적은 검색 기법을 하나 더 추가하는 것이 아니라, 질의를 보고 검색 전략과 데이터 소스를 스스로 고르는 오케스트레이션 계층을 만드는 것이다.

이번 구현에서 완성한 흐름은 아래와 같다.

1. 질의의 모호함과 의도를 분석한다.
2. 소스를 `family_card` 또는 `library`로 라우팅한다.
3. `rerank` 또는 `multi_rerank`로 1차 검색한다.
4. relevance score가 부족하면 `hyde_rerank` 등으로 fallback 한다.
5. 최종 답변은 항상 원본 질의를 기준으로 생성한다.

## 구현 범위

### 1. Planner / Trace

- `domain/agent.py`
  - `AgentPlan`
  - `AgentStep`
  - `AgentResult`
- `application/agentic.py`
  - 질의 분석
  - 소스 선택
  - 전략 선택
  - fallback 실행
  - trace 기록

### 2. 멀티 소스 카탈로그

- `application/sources.py`
  - `family_card`
  - `library`

각 소스는 다음 정보를 가진다.

- 컬렉션 이름
- 기본 JSON 경로
- 문서 로더
- query rewriting용 도메인 힌트
- 라우팅 키워드

### 3. 두 번째 데이터 소스

- `data/raw/busan_libraries.json`

도서관 OpenAPI는 계정 승인 이슈로 `403 Forbidden`이 발생해, Quest 7에서는 저장소에 포함된 스냅샷 JSON을 두 번째 소스로 사용한다.

### 4. 답변/검색 계층 일반화

- `application/answer.py`에 `retrieve_hits()` 추가
- 기존 파이프라인을 retrieval + answer 구조로 분리
- 소스별 query rewriting domain hint 지원

### 5. CLI / 비교 스크립트 연결

- `app/cli.py`
  - `Agentic RAG` 모드 추가
- `application/compare.py`
  - `agentic` 모드 추가

## 현재 휴리스틱

### 모호도 판단

- 추천형 표현: `추천`, `갈 만`, `가볼`, `어디`, `좋은 곳`, `주말`
- 상황형 표현: `아이`, `아이들`, `가족`, `데이트`, `체험`
- 지역 제약이 없으면 모호도를 더 높게 본다

### 전략 선택

- `low ambiguity` -> `rerank`
- `medium/high ambiguity` -> `multi_rerank`
- fallback -> `hyde_rerank`, `rerank`

### relevance threshold

- `low`: `0.55`
- `medium`: `0.15`
- `high`: `0.03`

모호한 질의는 relevance score가 낮게 형성되기 쉬워 임계치를 완화했다.

## 실행 예시

### 구체적 질의

```bash
uv run python -m rag_playground.application.agentic --query "동래구 목욕탕"
```

예상:

- source: `family_card`
- initial mode: `rerank`
- fallback 없이 종료

### 탐색형 질의

```bash
uv run python -m rag_playground.application.agentic --query "이번 주말에 아이들이랑 갈 만한 데"
```

예상:

- initial source: `library`
- initial mode: `multi_rerank`
- 결과 부족 시 `library + family_card`로 확장

## 남은 과제

- source selection을 LLM planner로 고도화
- source별 retrieval score calibration
- 정량 평가용 test set 구축
- NDCG/MRR 측정 파이프라인 추가
