import streamlit as st
import os
import json
from PIL import Image
from utils.rag_chain import query_rag, query_rag_with_image
from utils.pdf_report import generate_buyer_report_pdf

st.set_page_config(page_title="FRS - 바이어 검색", page_icon="🔍", layout="wide")

# ── CSS ───────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap');
html, body, [class*="css"] { font-family: 'Inter', sans-serif; }
[data-testid="stAppViewContainer"] { background-color: #0a0a0a; color: #f0f0f0; }
[data-testid="stHeader"] { background: transparent; }

.page-header {
    padding: 40px 0 32px;
    border-bottom: 1px solid #1a1a1a;
    margin-bottom: 32px;
}
.page-title { font-size: 32px; font-weight: 700; color: #fff; margin-bottom: 6px; }
.page-sub { font-size: 14px; color: #555; }

.search-box {
    background: #111;
    border: 1px solid #222;
    border-radius: 16px;
    padding: 32px;
    margin-bottom: 24px;
}
.example-tag {
    display: inline-block;
    background: #1a1a1a;
    border: 1px solid #2a2a2a;
    border-radius: 100px;
    padding: 4px 12px;
    font-size: 12px;
    color: #777;
    margin: 3px;
    cursor: pointer;
}
.answer-box {
    background: #111;
    border: 1px solid #1e1e1e;
    border-left: 3px solid #444;
    border-radius: 14px;
    padding: 28px;
    margin: 24px 0;
    font-size: 14px;
    line-height: 1.8;
    color: #ccc;
}
.result-card {
    background: #111;
    border: 1px solid #1e1e1e;
    border-radius: 14px;
    padding: 24px;
    margin-bottom: 12px;
    transition: border-color 0.2s;
}
.result-card:hover { border-color: #333; }

.score-bar-bg {
    background: #1a1a1a;
    border-radius: 100px;
    height: 4px;
    margin-top: 6px;
}
.score-bar-fill {
    height: 4px;
    border-radius: 100px;
    background: linear-gradient(90deg, #444, #888);
}
.spec-row {
    display: flex;
    gap: 8px;
    align-items: center;
    padding: 5px 0;
    border-bottom: 1px solid #1a1a1a;
    font-size: 12px;
}
.spec-key { color: #555; width: 70px; flex-shrink: 0; }
.spec-val { color: #bbb; }
.badge {
    display: inline-block;
    background: #1a1a1a;
    border: 1px solid #2a2a2a;
    border-radius: 100px;
    padding: 3px 10px;
    font-size: 11px;
    color: #888;
    margin: 2px;
}
.ref-header {
    font-size: 11px;
    color: #444;
    letter-spacing: 1px;
    text-transform: uppercase;
    margin-bottom: 16px;
    padding-bottom: 8px;
    border-bottom: 1px solid #1a1a1a;
}
.source-tag {
    font-size: 11px;
    color: #555;
    font-family: monospace;
    background: #0d0d0d;
    padding: 2px 8px;
    border-radius: 4px;
}
.stButton > button {
    background: #fff !important;
    color: #000 !important;
    border: none !important;
    border-radius: 10px !important;
    font-weight: 600 !important;
    font-size: 13px !important;
}
.back-btn > div > button {
    background: transparent !important;
    color: #666 !important;
    border: 1px solid #222 !important;
    font-size: 12px !important;
}
[data-testid="stSidebar"] {
    background: #0d0d0d !important;
    border-right: 1px solid #1a1a1a !important;
}
</style>
""", unsafe_allow_html=True)


def _display_result(result: dict):
    """검색 결과 공통 출력"""

    # AI 답변
    st.markdown("### 💡 AI 추천")
    st.markdown(f'<div class="answer-box">{result["answer"]}</div>', unsafe_allow_html=True)

    # 참조 원단
    st.markdown(f"""
    <div class="ref-header">
        참조된 원단 데이터 — {len(result['retrieved_fabrics'])}개
    </div>
    """, unsafe_allow_html=True)

    for i, fabric in enumerate(result["retrieved_fabrics"], 1):
        score = fabric.get("similarity_score", 0)
        score_pct = int(score * 100)

        with st.expander(
            f"**{fabric.get('id', '?')}** &nbsp; "
            f"{fabric.get('vendor', '')} `{fabric.get('item_code', '')}` — "
            f"{fabric.get('name', '알수없음')} &nbsp; | &nbsp; 유사도 {score_pct}%"
        ):
            col1, col2 = st.columns([1, 2])

            with col1:
                img_path = f"data/images/{fabric.get('source_image', '')}"
                if os.path.exists(img_path):
                    st.image(img_path, use_container_width=True)
                else:
                    st.markdown("""
                    <div style="background:#111; border:1px solid #222; border-radius:8px;
                    height:160px; display:flex; align-items:center; justify-content:center;
                    color:#333; font-size:12px;">이미지 없음</div>
                    """, unsafe_allow_html=True)

                # 유사도 바
                st.markdown(f"""
                <div style="margin-top:12px">
                    <div style="font-size:11px; color:#555; margin-bottom:4px;">유사도</div>
                    <div style="font-size:20px; font-weight:700; color:#fff">{score_pct}%</div>
                    <div class="score-bar-bg">
                        <div class="score-bar-fill" style="width:{score_pct}%"></div>
                    </div>
                </div>
                """, unsafe_allow_html=True)

            with col2:
                st.markdown(f"""
                <div class="spec-row"><span class="spec-key">품번</span><span class="spec-val">{fabric.get('item_code', '-')}</span></div>
                <div class="spec-row"><span class="spec-key">업체</span><span class="spec-val">{fabric.get('vendor', '-')}</span></div>
                <div class="spec-row"><span class="spec-key">혼용률</span><span class="spec-val">{fabric.get('composition', '-')}</span></div>
                <div class="spec-row"><span class="spec-key">평량</span><span class="spec-val">{fabric.get('weight', '-')}</span></div>
                <div class="spec-row"><span class="spec-key">폭</span><span class="spec-val">{fabric.get('width', '-')}</span></div>
                <div class="spec-row"><span class="spec-key">가공</span><span class="spec-val">{fabric.get('finish', '-')}</span></div>
                <div class="spec-row"><span class="spec-key">시즌</span><span class="spec-val">{', '.join(fabric.get('season', []))}</span></div>
                """, unsafe_allow_html=True)

                chars = fabric.get('characteristics', [])
                if chars:
                    badges = "".join([f'<span class="badge">{c}</span>' for c in chars])
                    st.markdown(f'<div style="margin-top:12px">{badges}</div>', unsafe_allow_html=True)

                st.markdown(f"""
                <div style="margin-top:16px; padding:12px; background:#0d0d0d;
                border-radius:8px; font-size:12px; color:#666; line-height:1.6;">
                    {fabric.get('description', '-')}
                </div>
                <div style="margin-top:8px">
                    <span class="source-tag">📎 {fabric.get('source_image', '알수없음')}</span>
                </div>
                """, unsafe_allow_html=True)

    pdf_bytes = generate_buyer_report_pdf(result)
    st.download_button(
        "📄 소재 제안 리포트 PDF 다운로드",
        data=pdf_bytes,
        file_name="frs_fabric_recommendation_report.pdf",
        mime="application/pdf",
        use_container_width=True,
    )


# ── 사이드바 ───────────────────────────────────────────
with st.sidebar:
    st.markdown("### ⚙️ 검색 설정")
    top_k = st.slider("참조 원단 수", 3, 10, 5)
    st.markdown("---")
    st.markdown("### 📌 검색 팁")
    st.markdown("""
    <div style="font-size:12px; color:#555; line-height:1.8;">
    • 감성적 표현도 OK<br>
    • 복종 + 소재 조합으로 검색<br>
    • 시즌, 기능성 함께 입력<br>
    • 캐드 이미지 업로드 시 자동 분석
    </div>
    """, unsafe_allow_html=True)
    st.markdown("---")
    st.markdown('<div class="back-btn">', unsafe_allow_html=True)
    if st.button("← 홈으로", use_container_width=True):
        st.switch_page("app.py")
    st.markdown('</div>', unsafe_allow_html=True)

# ── 벡터스토어 확인 ────────────────────────────────────
if not os.path.exists("vectorstore/faiss_index"):
    st.markdown("""
    <div style="text-align:center; padding:80px; color:#444;">
        <div style="font-size:48px; margin-bottom:16px;">⚠️</div>
        <div style="font-size:18px; color:#666; margin-bottom:8px;">원단 DB가 비어있습니다</div>
        <div style="font-size:13px;">벤더 포털에서 원단을 먼저 등록해주세요.</div>
    </div>
    """, unsafe_allow_html=True)
    if st.button("벤더 포털로 이동 →", use_container_width=True):
        st.switch_page("pages/1_vendor.py")
    st.stop()

# ── 헤더 ───────────────────────────────────────────────
st.markdown("""
<div class="page-header">
    <div class="page-title">🔍 소재 검색</div>
    <div class="page-sub">자연어 또는 캐드 이미지로 최적의 원단을 찾아드립니다.</div>
</div>
""", unsafe_allow_html=True)

# ── 검색 모드 ──────────────────────────────────────────
mode = st.radio(
    "검색 방식",
    ["📝 텍스트 검색", "🖼️ 캐드 이미지 검색"],
    horizontal=True,
    label_visibility="collapsed"
)

st.markdown("---")

# ── 텍스트 검색 ───────────────────────────────────────
if mode == "📝 텍스트 검색":
    st.markdown('<div class="search-box">', unsafe_allow_html=True)

    examples = [
        "바스락거리는 여름 아노락 소재",
        "드레이프 좋은 가을 포멀 원단",
        "친환경 스포츠웨어용 스트레치 소재",
        "탄탄한 고시감의 정장 바지 원단",
        "부드럽고 보온성 좋은 겨울 이너",
    ]
    st.markdown(
        "**예시 검색어** &nbsp;" +
        "".join([f'<span class="example-tag">{e}</span>' for e in examples]),
        unsafe_allow_html=True
    )

    st.markdown("<div style='margin-top:16px'>", unsafe_allow_html=True)
    user_query = st.text_area(
        "검색어",
        placeholder="예) 바스락거리는 여름 아노락에 어울리는 방풍 소재가 필요해요",
        height=100,
        label_visibility="collapsed"
    )
    st.markdown('</div>', unsafe_allow_html=True)
    st.markdown('</div>', unsafe_allow_html=True)

    if st.button("🔍 AI 검색", type="primary", use_container_width=True):
        if not user_query.strip():
            st.warning("검색어를 입력해주세요.")
        else:
            with st.spinner("AI가 최적 원단을 분석 중입니다..."):
                result = query_rag(user_query, top_k=top_k)
            _display_result(result)

# ── 캐드 이미지 검색 ──────────────────────────────────
else:
    col1, col2 = st.columns([1, 1])

    with col1:
        st.markdown("**캐드(도식화) 이미지 업로드**")
        uploaded_cad = st.file_uploader(
            "캐드 업로드",
            type=["jpg", "jpeg", "png"],
            label_visibility="collapsed"
        )
        if uploaded_cad:
            st.image(uploaded_cad, caption="업로드된 캐드", use_container_width=True)

    with col2:
        st.markdown("**추가 요청사항**")
        user_query = st.text_area(
            "추가 요청",
            placeholder="예) 예산 m당 1만원 이하, 국내 재고 있는 소재로 추천해주세요",
            height=120,
            label_visibility="collapsed"
        )

        if st.button("🤖 이미지 분석 + 검색", type="primary", use_container_width=True):
            if not uploaded_cad:
                st.warning("캐드 이미지를 업로드해주세요.")
            else:
                image = Image.open(uploaded_cad)
                with st.spinner("캐드 이미지 분석 중..."):
                    result = query_rag_with_image(
                        user_query or "이 디자인에 적합한 원단을 추천해주세요",
                        image,
                        top_k=top_k
                    )

                if "design_analysis" in result:
                    with st.expander("🔎 캐드 이미지 분석 결과 보기"):
                        st.markdown(f"""
                        <div style="font-size:13px; color:#888; line-height:1.7; padding:12px;
                        background:#0d0d0d; border-radius:8px;">
                            {result['design_analysis']}
                        </div>
                        """, unsafe_allow_html=True)

                _display_result(result)
