import streamlit as st
import json
import os

try:
    from utils.supabase_client import load_all_fabrics
except Exception:
    load_all_fabrics = None

st.set_page_config(
    page_title="FRS — Fabric Retrieval System",
    page_icon="🧵",
    layout="wide",
    initial_sidebar_state="collapsed"
)

st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap');
html, body, [class*="css"] { font-family: 'Inter', sans-serif; }

[data-testid="stAppViewContainer"] { background-color: #0a0a0a; color: #f0f0f0; }
[data-testid="stHeader"] { background: transparent; }
[data-testid="stSidebar"] { display: none; }

.hero { text-align: center; padding: 80px 20px 60px; }
.hero-badge {
    display: inline-block;
    background: rgba(255,255,255,0.08);
    border: 1px solid rgba(255,255,255,0.15);
    border-radius: 100px;
    padding: 6px 18px;
    font-size: 12px;
    font-weight: 500;
    letter-spacing: 2px;
    text-transform: uppercase;
    color: #aaa;
    margin-bottom: 28px;
}
.hero-title {
    font-size: 56px;
    font-weight: 700;
    letter-spacing: -1.5px;
    line-height: 1.1;
    margin-bottom: 20px;
    background: linear-gradient(135deg, #ffffff 0%, #888888 100%);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
}
.hero-sub {
    font-size: 18px;
    color: #666;
    font-weight: 400;
    margin-bottom: 60px;
    line-height: 1.6;
}
.stats-bar {
    display: flex;
    justify-content: center;
    gap: 60px;
    padding: 40px 0;
    border-top: 1px solid #1a1a1a;
    border-bottom: 1px solid #1a1a1a;
    margin: 40px 0;
}
.stat-item { text-align: center; }
.stat-number { font-size: 32px; font-weight: 700; color: #fff; display: block; }
.stat-label { font-size: 12px; color: #555; letter-spacing: 1px; text-transform: uppercase; }
.role-card {
    background: #111111;
    border: 1px solid #222;
    border-radius: 20px;
    padding: 48px 40px;
    text-align: center;
    height: 100%;
    position: relative;
    overflow: hidden;
}
.role-card::before {
    content: '';
    position: absolute;
    top: 0; left: 0; right: 0;
    height: 1px;
    background: linear-gradient(90deg, transparent, rgba(255,255,255,0.1), transparent);
}
.role-icon { font-size: 48px; margin-bottom: 20px; display: block; }
.role-title { font-size: 24px; font-weight: 600; color: #fff; margin-bottom: 12px; }
.role-desc { font-size: 14px; color: #666; line-height: 1.7; margin-bottom: 32px; }
.role-features { text-align: left; margin-bottom: 36px; }
.feature-item {
    display: flex;
    align-items: center;
    gap: 10px;
    font-size: 13px;
    color: #888;
    padding: 8px 0;
    border-bottom: 1px solid #1a1a1a;
}
.feature-dot { width: 6px; height: 6px; border-radius: 50%; background: #444; flex-shrink: 0; }
.stButton > button {
    width: 100%;
    background: #fff !important;
    color: #000 !important;
    border: none !important;
    border-radius: 12px !important;
    padding: 14px 28px !important;
    font-size: 14px !important;
    font-weight: 600 !important;
}
.vendor-btn > div > button {
    background: #111 !important;
    color: #fff !important;
    border: 1px solid #333 !important;
}
.footer {
    text-align: center;
    padding: 40px 0 20px;
    color: #333;
    font-size: 12px;
    letter-spacing: 0.5px;
}
</style>
""", unsafe_allow_html=True)

def get_category(fabric):
    cat = fabric.get("category", "")
    if isinstance(cat, list):
        return cat[0] if cat else ""
    return cat


def get_stats():
    db = []
    if load_all_fabrics is not None:
        try:
            db = load_all_fabrics()
        except Exception:
            db = []

    if not db:
        db_path = "data/fabric_db.json"
        if os.path.exists(db_path):
            with open(db_path, "r", encoding="utf-8") as f:
                db = json.load(f)

    categories = set(get_category(f) for f in db if f.get("category"))
    return len(db), len(categories)

fabric_count, category_count = get_stats()
vs_ready = os.path.exists("vectorstore/faiss_index")

st.markdown("""
<div class="hero">
    <div class="hero-badge">AI-Powered Fabric Intelligence</div>
    <div class="hero-title">Fabric Retrieval<br>System</div>
    <div class="hero-sub">
        Gemini Vision 기반 원단 소싱 플랫폼<br>
        벤더는 원단을 등록하고, 바이어는 AI로 최적 소재를 찾습니다.
    </div>
</div>
""", unsafe_allow_html=True)

st.markdown(f"""
<div class="stats-bar">
    <div class="stat-item">
        <span class="stat-number">{fabric_count}</span>
        <span class="stat-label">등록 원단</span>
    </div>
    <div class="stat-item">
        <span class="stat-number">{category_count}</span>
        <span class="stat-label">카테고리</span>
    </div>
    <div class="stat-item">
        <span class="stat-number">{"✓" if vs_ready else "—"}</span>
        <span class="stat-label">AI 검색 {"준비됨" if vs_ready else "대기중"}</span>
    </div>
</div>
""", unsafe_allow_html=True)

col_left, col_gap, col_right = st.columns([1, 0.08, 1])

with col_left:
    st.markdown("""
    <div class="role-card">
        <span class="role-icon">🏭</span>
        <div class="role-title">벤더 포털</div>
        <div class="role-desc">
            원단 사진을 업로드하면 Gemini AI가<br>
            자동으로 소재를 분석하고 DB에 등록합니다.
        </div>
        <div class="role-features">
            <div class="feature-item"><div class="feature-dot"></div>원단 사진 업로드 (다중 업로드 지원)</div>
            <div class="feature-item"><div class="feature-dot"></div>Gemini Vision 자동 스펙 추출</div>
            <div class="feature-item"><div class="feature-dot"></div>등록 원단 현황 조회 및 관리</div>
            <div class="feature-item"><div class="feature-dot"></div>AI 검색 인덱스 자동 업데이트</div>
        </div>
    </div>
    """, unsafe_allow_html=True)
    st.markdown('<div class="vendor-btn">', unsafe_allow_html=True)
    if st.button("벤더 포털 입장", key="vendor", use_container_width=True):
        st.switch_page("pages/1_vendor.py")
    st.markdown('</div>', unsafe_allow_html=True)

with col_right:
    st.markdown("""
    <div class="role-card">
        <span class="role-icon">🔍</span>
        <div class="role-title">바이어 검색</div>
        <div class="role-desc">
            텍스트 또는 캐드 이미지로 원하는 소재를<br>
            AI가 실시간으로 매칭하고 추천합니다.
        </div>
        <div class="role-features">
            <div class="feature-item"><div class="feature-dot"></div>자연어 텍스트 소재 검색</div>
            <div class="feature-item"><div class="feature-dot"></div>캐드(도식화) 이미지 업로드 검색</div>
            <div class="feature-item"><div class="feature-dot"></div>AI 추천 + 참조 원단 출처 제공</div>
            <div class="feature-item"><div class="feature-dot"></div>유사도 기반 Top-K 결과 제공</div>
        </div>
    </div>
    """, unsafe_allow_html=True)
    if st.button("바이어 검색 입장", key="buyer", use_container_width=True):
        st.switch_page("pages/2_buyer.py")

st.markdown("""
<div class="footer">
    FRS v1.0 &nbsp;|&nbsp; 인공지능응용 기말 프로젝트 &nbsp;|&nbsp; 나상원 &nbsp;|&nbsp; Powered by Gemini &amp; FAISS
</div>
""", unsafe_allow_html=True)
