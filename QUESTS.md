# Quests

RAG 파이프라인을 단계적으로 완성해 나가는 퀘스트 보드.

## Quest 1: 프로젝트 환경 세팅 ✅
> Python 프로젝트 초기 설정. 패키지 매니저, 의존성, 디렉토리 구조, .gitignore 등 기본 뼈대 구축.

## Quest 2: 공공데이터 API 연동 ✅
> 공공데이터포털에서 API 키 발급 후, 첫 번째 데이터 소스(부산 가족사랑카드) API 호출 및 데이터 수집 파이프라인 구축.

## Quest 3: Naive RAG 파이프라인 구축 ✅
> ChromaDB에 공공데이터 임베딩 저장 → 벡터 검색 → LLM 응답 생성. 가장 기본적인 RAG 파이프라인 완성.

## Quest 4: Hybrid Search 적용 ✅
> 키워드 검색(BM25 등) + 벡터 검색을 결합한 Hybrid Search 구현. Naive RAG 대비 성능 비교.

## Quest 5: Re-ranking 도입 ✅
> Hybrid 검색(상위 20개) → Novita.ai BGE-reranker-v2-m3로 재정렬 → 상위 5개 반환. relevance score가 신뢰도 지표로 활용됨 (0.9+: 신뢰, 0.5 미만: 데이터 부재 가능성).

## Quest 6: Query Rewriting 실험 ✅
> 사용자 질의를 LLM으로 재작성하여 검색 품질 향상. HyDE(가상 문서 임베딩)와 Multi-Query(대안 질의 3개 생성 후 결과 합산) 기법 구현. 두 기법 모두 Hybrid + Re-rank 파이프라인 위에 합성되며, 재작성 질의는 검색에만 사용하고 리랭킹·답변 생성에는 원본 질의 유지.

## Quest 7: Agentic RAG 구현 ⬜
> 에이전트가 스스로 검색 전략을 판단하고, 다중 데이터 소스를 활용하는 자율 RAG 시스템 완성. 최종 보스 퀘스트.

---

**범례:** ⬜ 대기 | 🔄 진행 중 | ✅ 완료
