# FRS (Fabric Retrieval System)

Gemini Vision 기반 패션 원단 소싱 RAG 시스템입니다. 벤더가 원단 스와치 이미지를 업로드하면 AI가 원단 스펙을 추출해 DB에 저장하고, 바이어는 자연어 또는 캐드 이미지로 적합한 원단을 검색할 수 있습니다.

## Tech Stack

- Gemini 3.5 Flash: 원단 이미지 분석, 검색 답변 생성, 쿼리 재작성
- BGE-M3: 원단 텍스트 임베딩
- FAISS HNSW: 벡터 검색 인덱스
- LangGraph CRAG: 검색, 관련성 평가, 쿼리 재작성, 답변 생성 파이프라인
- Neo4j GraphRAG: 원단, 카테고리, 특성, 복종, 시즌 관계 기반 추론 검색
- Streamlit: 벤더/바이어 웹 UI

## Folder Structure

```text
frs_project/
├── app.py                    # Streamlit 홈 화면
├── pages/
│   ├── 1_vendor.py           # 벤더 원단 업로드/관리 페이지
│   └── 2_buyer.py            # 바이어 원단 검색 페이지
├── utils/
│   ├── gemini_vision.py      # Gemini Vision 원단 이미지 분석
│   ├── vectorstore.py        # BGE-M3 + FAISS HNSW 벡터스토어
│   ├── rag_chain.py          # LangGraph CRAG 검색 파이프라인
│   ├── knowledge_graph.py    # Neo4j GraphRAG 구축/검색
│   ├── data_processor.py     # 원단 데이터 전처리/통계
│   └── pdf_report.py         # 바이어 검색 결과 PDF 리포트 생성
├── scripts/
│   └── build_graph.py        # fabric_db.json → Neo4j 그래프 일괄 구축
├── data/
│   ├── fabric_db.json        # 원단 메타데이터 DB
│   └── images/               # 업로드 원단 이미지, Git 제외
├── vectorstore/              # FAISS 인덱스, Git 제외
├── requirements.txt
├── .env.example
└── .gitignore
```

## Local Setup

1. 가상환경 생성 및 활성화

```powershell
python -m venv venv
.\venv\Scripts\Activate.ps1
```

2. 패키지 설치

```powershell
pip install -r requirements.txt
```

3. 환경변수 설정

```powershell
copy .env.example .env
```

`.env` 파일에 실제 API 키와 Neo4j 접속 정보를 입력합니다.

4. 앱 실행

```powershell
streamlit run app.py
```

5. Neo4j 그래프 수동 빌드

벤더 포털에서 원단 업로드 후 자동으로 그래프가 동기화됩니다. 수동으로 전체 그래프를 다시 만들려면 아래 명령을 실행합니다.

```powershell
python scripts/build_graph.py --reset
```

## Environment Variables

`.env.example`을 참고해 `.env` 파일을 생성하세요.

```env
GEMINI_API_KEY=your_gemini_api_key_here
NEO4J_URI=neo4j+s://your-neo4j-uri
NEO4J_USER=neo4j
NEO4J_PASSWORD=your_neo4j_password_here
```

## Notes

- `.env`는 Git에 올리지 않습니다.
- `data/images/`와 `vectorstore/`는 생성 데이터이므로 Git에서 제외합니다.
- `data/fabric_db.json`은 원단 메타데이터 DB입니다. 샘플 데이터 또는 실제 데이터 포함 여부는 저장소 공개 범위에 맞게 결정하세요.
