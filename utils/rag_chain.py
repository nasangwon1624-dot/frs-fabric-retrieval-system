import os
from google import genai
from langgraph.graph import StateGraph, END
from typing import TypedDict, List
from PIL import Image
from utils.vectorstore import search, fabric_to_text
from utils.knowledge_graph import graph_search
from dotenv import load_dotenv

load_dotenv()
client = genai.Client(api_key=os.getenv("GEMINI_API_KEY"))

# ── 상태 정의 ──────────────────────────────────────────
class RAGState(TypedDict):
    query: str                    # 원본 쿼리
    rewritten_query: str          # 재작성된 쿼리
    retrieved_fabrics: List[dict] # 검색된 원단 목록
    graph_fabrics: List[dict]     # Neo4j 그래프 검색 결과
    graph_error: str              # Neo4j 실패 시 fallback 사유
    grade: str                    # relevant / irrelevant
    answer: str                   # 최종 답변
    retry_count: int              # 재시도 횟수
    top_k: int                    # 검색 결과 수

# ── 시스템 프롬프트 ────────────────────────────────────
SYSTEM_PROMPT = """당신은 패션 원단 소싱 전문가 AI입니다.
디자이너의 요청에 맞는 최적의 원단을 추천하는 역할을 합니다.

답변 규칙:
1. 반드시 제공된 [참조 원단 데이터]만 근거로 답변하세요.
2. 데이터에 없는 내용은 절대 추측하거나 만들어내지 마세요.
3. 각 추천 원단에 대해 왜 적합한지 근거를 명확히 설명하세요.
4. 답변 마지막에 참조한 원단의 ID, 품번, 업체명을 반드시 출력하세요.
5. 한국어로 답변하세요.
6. 최종 소재 제안은 리포트 형식으로 작성하세요:
   - 추천 요약
   - 각 원단별 상세 추천 이유
   - 주의사항 (세탁, 관리 등)
   - 참조 출처
"""

GRADER_PROMPT = """당신은 검색 결과의 관련성을 평가하는 전문가입니다.

[사용자 쿼리]
{query}

[검색된 원단 데이터]
{context}

위 검색 결과가 사용자 쿼리에 답변하기에 충분히 관련성이 있는지 평가하세요.

평가 기준:
- 쿼리의 핵심 요구사항(소재, 기능, 시즌, 복종 등)이 검색 결과에 포함되어 있는가?
- 유사도가 0.5 이상인 결과가 2개 이상인가?

반드시 아래 중 하나만 출력하세요 (다른 텍스트 없이):
relevant
irrelevant
"""

REWRITE_PROMPT = """당신은 검색 쿼리 최적화 전문가입니다.

[원본 쿼리]
{query}

[실패 이유]
검색 결과가 쿼리와 관련성이 낮습니다.

위 쿼리를 원단 검색에 더 적합하게 재작성해주세요.
- 한국어 감성 표현을 구체적인 원단 특성으로 변환
- 영어 패션/텍스타일 용어 추가
- 핵심 키워드 강조

재작성된 쿼리만 출력하세요 (다른 텍스트 없이):
"""

def format_context(fabrics: list) -> str:
    """검색된 원단을 프롬프트용 텍스트로 변환"""
    lines = []
    for i, f in enumerate(fabrics, 1):
        lines.append(f"[원단 {i}]")
        lines.append(f"  ID: {f.get('id', '?')}")
        lines.append(f"  업체: {f.get('vendor', '?')} | 품번: {f.get('item_code', '?')}")
        lines.append(f"  {fabric_to_text(f)}")
        if f.get("similarity_score") is not None:
            lines.append(f"  벡터 유사도: {f.get('similarity_score', 0):.3f}")
        if f.get("graph_score") is not None:
            lines.append(f"  그래프 점수: {f.get('graph_score', 0)}")
        if f.get("matched_graph_terms"):
            lines.append(f"  그래프 매칭 근거: {f.get('matched_graph_terms')}")
        if f.get("retrieval_sources"):
            lines.append(f"  검색 출처: {', '.join(f.get('retrieval_sources', []))}")
        lines.append("")
    return "\n".join(lines)


def merge_retrieval_results(vector_fabrics: list, graph_fabrics: list, top_k: int) -> list:
    """FAISS 결과와 Neo4j GraphRAG 결과를 Fabric id 기준으로 병합합니다."""
    merged = {}

    for rank, fabric in enumerate(vector_fabrics, 1):
        fabric_id = fabric.get("id")
        if not fabric_id:
            continue
        item = fabric.copy()
        item["retrieval_sources"] = ["vector"]
        item["vector_rank"] = rank
        merged[fabric_id] = item

    for rank, fabric in enumerate(graph_fabrics, 1):
        fabric_id = fabric.get("id")
        if not fabric_id:
            continue
        if fabric_id in merged:
            merged[fabric_id].update({
                "graph_score": fabric.get("graph_score"),
                "graph_rank": rank,
                "matched_graph_terms": fabric.get("matched_graph_terms"),
                "query_terms": fabric.get("query_terms"),
            })
            sources = merged[fabric_id].setdefault("retrieval_sources", ["vector"])
            if "graph" not in sources:
                sources.append("graph")
        else:
            item = fabric.copy()
            item["retrieval_sources"] = ["graph"]
            item["graph_rank"] = rank
            item.setdefault("similarity_score", min(float(item.get("graph_score", 0)) / 10, 1.0))
            merged[fabric_id] = item

    def combined_rank(item: dict) -> tuple:
        source_bonus = len(item.get("retrieval_sources", []))
        vector_rank = item.get("vector_rank") or top_k + 1
        graph_rank = item.get("graph_rank") or top_k + 1
        graph_score = item.get("graph_score") or 0
        return (-source_bonus, vector_rank + graph_rank, -graph_score)

    return sorted(merged.values(), key=combined_rank)[:top_k]


# ── LangGraph 노드 정의 ────────────────────────────────

def retrieve_node(state: RAGState) -> RAGState:
    """벡터 검색과 Neo4j GraphRAG 검색을 함께 수행합니다."""
    query = state.get("rewritten_query") or state["query"]
    top_k = state.get("top_k", 5)

    vector_fabrics = search(query, top_k=top_k)
    graph_fabrics = []
    graph_error = ""

    try:
        graph_fabrics = graph_search(query, top_k=top_k)
    except Exception as e:
        graph_error = str(e)
        print(f"Neo4j GraphRAG fallback: {graph_error}")

    fabrics = merge_retrieval_results(vector_fabrics, graph_fabrics, top_k=top_k)
    return {
        **state,
        "retrieved_fabrics": fabrics,
        "graph_fabrics": graph_fabrics,
        "graph_error": graph_error,
    }


def grade_node(state: RAGState) -> RAGState:
    """검색 결과 품질 평가 노드 (CRAG 핵심)"""
    context = format_context(state["retrieved_fabrics"])

    prompt = GRADER_PROMPT.format(
        query=state["query"],
        context=context
    )

    response = client.models.generate_content(
        model="gemini-3.5-flash",
        contents=prompt
    )
    grade = response.text.strip().lower()

    # 안전장치: 명확하지 않으면 relevant로 처리
    if grade not in ["relevant", "irrelevant"]:
        grade = "relevant"

    return {**state, "grade": grade}


def rewrite_node(state: RAGState) -> RAGState:
    """쿼리 재작성 노드 (CRAG 교정)"""
    prompt = REWRITE_PROMPT.format(query=state["query"])
    response = client.models.generate_content(
        model="gemini-3.5-flash",
        contents=prompt
    )
    rewritten = response.text.strip()

    print(f"🔄 쿼리 재작성: '{state['query']}' → '{rewritten}'")

    return {
        **state,
        "rewritten_query": rewritten,
        "retry_count": state.get("retry_count", 0) + 1
    }


def generate_node(state: RAGState) -> RAGState:
    """최종 답변 생성 노드"""
    context = format_context(state["retrieved_fabrics"])

    prompt = f"""{SYSTEM_PROMPT}

[참조 원단 데이터]
{context}

[디자이너 요청]
{state['query']}

위 데이터를 바탕으로 최적의 원단을 추천해주세요."""

    response = client.models.generate_content(
        model="gemini-3.5-flash",
        contents=prompt
    )
    return {**state, "answer": response.text}


def should_retry(state: RAGState) -> str:
    """재시도 여부 결정 (최대 1회 재시도)"""
    if state["grade"] == "relevant":
        return "generate"
    if state.get("retry_count", 0) >= 1:
        # 재시도 초과 시 그냥 생성
        return "generate"
    return "rewrite"


# ── LangGraph 그래프 구축 ──────────────────────────────

def build_rag_graph():
    graph = StateGraph(RAGState)

    # 노드 등록
    graph.add_node("retrieve", retrieve_node)
    graph.add_node("grade",    grade_node)
    graph.add_node("rewrite",  rewrite_node)
    graph.add_node("generate", generate_node)

    # 엣지 연결
    graph.set_entry_point("retrieve")
    graph.add_edge("retrieve", "grade")
    graph.add_conditional_edges(
        "grade",
        should_retry,
        {
            "generate": "generate",
            "rewrite":  "rewrite",
        }
    )
    graph.add_edge("rewrite",  "retrieve")
    graph.add_edge("generate", END)

    return graph.compile()


# 그래프 싱글톤
_rag_graph = None

def get_rag_graph():
    global _rag_graph
    if _rag_graph is None:
        _rag_graph = build_rag_graph()
    return _rag_graph


# ── 공개 API ───────────────────────────────────────────

def query_rag(user_query: str, top_k: int = 5) -> dict:
    """텍스트 쿼리 → CRAG → 답변 반환"""
    graph = get_rag_graph()

    initial_state: RAGState = {
        "query": user_query,
        "rewritten_query": "",
        "retrieved_fabrics": [],
        "graph_fabrics": [],
        "graph_error": "",
        "grade": "",
        "answer": "",
        "retry_count": 0,
        "top_k": top_k,
    }

    result = graph.invoke(initial_state)

    return {
        "answer": result["answer"],
        "retrieved_fabrics": result["retrieved_fabrics"],
        "graph_fabrics": result.get("graph_fabrics", []),
        "graph_error": result.get("graph_error", ""),
        "query": user_query,
        "rewritten_query": result.get("rewritten_query", ""),
        "grade": result.get("grade", ""),
    }


def query_rag_with_image(user_query: str, image: Image.Image, top_k: int = 5) -> dict:
    """캐드 이미지 + 텍스트 → CRAG → 답변 반환"""
    # 1. 캐드 이미지 분석
    image_prompt = """이 패션 캐드(도식화) 이미지를 분석하여
필요한 원단 특성을 추출해주세요.

아래 항목을 중심으로 분석하세요:
- 실루엣과 볼륨감 (타이트/루즈, 구조적/플로이)
- 필요한 드레이프성 (흘러내리는/뻣뻣한)
- 기능성 요구사항 (방풍/방수/스트레치 등)
- 적합한 소재 특성 키워드 3-5개

한국어로 답변하세요."""

    image_response = client.models.generate_content(
        model="gemini-3.5-flash",
        contents=[image_prompt, image]
    )
    design_analysis = image_response.text

    # 2. 이미지 분석 + 쿼리 결합
    combined_query = f"{user_query} {design_analysis}"

    # 3. CRAG 실행
    result = query_rag(combined_query, top_k=top_k)
    result["design_analysis"] = design_analysis
    result["query"] = user_query

    return result
