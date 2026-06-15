import streamlit as st
import json
import os
import time
from PIL import Image
from utils.gemini_vision import analyze_fabric_image
from utils.vectorstore import build_vectorstore
from utils.knowledge_graph import build_knowledge_graph
from utils.data_processor import get_quality_report

st.set_page_config(page_title="FRS - 벤더 포털", page_icon="🏭", layout="wide")

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

.stat-card {
    background: #111;
    border: 1px solid #222;
    border-radius: 14px;
    padding: 20px 24px;
    text-align: center;
}
.stat-num { font-size: 28px; font-weight: 700; color: #fff; display: block; }
.stat-label { font-size: 11px; color: #555; letter-spacing: 1px; text-transform: uppercase; }

.upload-zone {
    border: 1px dashed #333;
    border-radius: 16px;
    padding: 40px;
    text-align: center;
    background: #0d0d0d;
    margin-bottom: 24px;
}
.fabric-card {
    background: #111;
    border: 1px solid #1e1e1e;
    border-radius: 14px;
    padding: 20px;
    margin-bottom: 12px;
}
.fabric-card:hover { border-color: #333; }

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
.spec-row {
    display: flex;
    gap: 8px;
    align-items: center;
    padding: 6px 0;
    border-bottom: 1px solid #1a1a1a;
    font-size: 13px;
}
.spec-key { color: #555; width: 80px; flex-shrink: 0; }
.spec-val { color: #ccc; }

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

DB_PATH = "data/fabric_db.json"
os.makedirs("data", exist_ok=True)
os.makedirs("data/images", exist_ok=True)

def load_db():
    if os.path.exists(DB_PATH):
        with open(DB_PATH, "r", encoding="utf-8") as f:
            return json.load(f)
    return []

def save_db(data):
    with open(DB_PATH, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)


def get_category(fabric):
    cat = fabric.get("category", "")
    if isinstance(cat, list):
        return cat[0] if cat else ""
    return cat


def format_top_stats(stats, limit=3):
    if not stats:
        return '<div style="font-size:12px; color:#444;">데이터 없음</div>'
    rows = []
    for name, count in list(stats.items())[:limit]:
        rows.append(
            f'<div style="display:flex; justify-content:space-between; font-size:12px; '
            f'color:#666; padding:3px 0;"><span>{name}</span><span>{count}</span></div>'
        )
    return "".join(rows)


# ── 사이드바 ───────────────────────────────────────────
with st.sidebar:
    st.markdown("### 📊 DB 현황")
    db = load_db()
    valid = [f for f in db if f.get("name") != "분석 실패"]
    categories = set(get_category(f) for f in valid if f.get("category"))

    st.markdown(f"""
    <div class="stat-card" style="margin-bottom:12px">
        <span class="stat-num">{len(valid)}</span>
        <span class="stat-label">등록 원단</span>
    </div>
    <div class="stat-card" style="margin-bottom:12px">
        <span class="stat-num">{len(categories)}</span>
        <span class="stat-label">카테고리</span>
    </div>
    <div class="stat-card" style="margin-bottom:24px">
        <span class="stat-num">{"✓" if os.path.exists("vectorstore/faiss_index") else "—"}</span>
        <span class="stat-label">검색 인덱스</span>
    </div>
    """, unsafe_allow_html=True)

    st.markdown("### 🧹 데이터 품질 리포트")
    quality = get_quality_report(DB_PATH)
    st.markdown(f"""
    <div class="stat-card" style="margin-bottom:12px; text-align:left">
        <div style="display:flex; justify-content:space-between; margin-bottom:6px;">
            <span class="stat-label">중복 품번</span>
            <span style="color:#fff; font-weight:700;">{quality["duplicate_count"]}</span>
        </div>
        <div style="display:flex; justify-content:space-between; margin-bottom:6px;">
            <span class="stat-label">정제 후 원단</span>
            <span style="color:#fff; font-weight:700;">{quality["deduped_count"]}</span>
        </div>
        <div style="display:flex; justify-content:space-between; margin-bottom:6px;">
            <span class="stat-label">판독불가 항목</span>
            <span style="color:#fff; font-weight:700;">{quality["unreadable_count"]}</span>
        </div>
        <div style="display:flex; justify-content:space-between;">
            <span class="stat-label">결측치</span>
            <span style="color:#fff; font-weight:700;">{quality["missing_count"]}</span>
        </div>
    </div>
    <div style="font-size:11px; color:#555; letter-spacing:1px; text-transform:uppercase; margin:12px 0 6px;">Top Categories</div>
    {format_top_stats(quality["category_stats"])}
    <div style="font-size:11px; color:#555; letter-spacing:1px; text-transform:uppercase; margin:12px 0 6px;">Top Vendors</div>
    {format_top_stats(quality["vendor_stats"])}
    <div style="font-size:11px; color:#555; letter-spacing:1px; text-transform:uppercase; margin:12px 0 6px;">Top Seasons</div>
    {format_top_stats(quality["season_stats"])}
    """, unsafe_allow_html=True)

    if st.button("🔄 검색 인덱스 재빌드", use_container_width=True):
        if len(valid) > 0:
            with st.spinner("인덱스 구축 중..."):
                count = build_vectorstore(DB_PATH)
            st.success(f"✓ {count}개 원단 인덱싱 완료")
        else:
            st.warning("등록된 원단이 없습니다.")

    st.markdown("---")
    st.markdown('<div class="back-btn">', unsafe_allow_html=True)
    if st.button("← 홈으로", use_container_width=True):
        st.switch_page("app.py")
    st.markdown('</div>', unsafe_allow_html=True)

# ── 헤더 ───────────────────────────────────────────────
st.markdown("""
<div class="page-header">
    <div class="page-title">🏭 벤더 포털</div>
    <div class="page-sub">원단 사진을 업로드하면 Gemini AI가 자동으로 스펙을 분석하고 DB에 등록합니다.</div>
</div>
""", unsafe_allow_html=True)

# ── 탭 ────────────────────────────────────────────────
tab1, tab2 = st.tabs(["📤 원단 업로드", "📋 등록 현황"])

# ── 탭1: 업로드 ───────────────────────────────────────
with tab1:
    st.markdown('<div class="upload-zone">', unsafe_allow_html=True)
    uploaded_files = st.file_uploader(
        "원단 스와치 시트 사진을 업로드하세요",
        type=["jpg", "jpeg", "png"],
        accept_multiple_files=True,
        help="여러 장 동시 업로드 가능합니다."
    )
    st.markdown('</div>', unsafe_allow_html=True)

    if uploaded_files:
        st.markdown(f"**{len(uploaded_files)}장** 선택됨")

        # 미리보기
        preview_cols = st.columns(min(len(uploaded_files), 5))
        for i, f in enumerate(uploaded_files[:5]):
            with preview_cols[i]:
                st.image(f, caption=f.name[:15], use_container_width=True)
        if len(uploaded_files) > 5:
            st.caption(f"... 외 {len(uploaded_files)-5}장 더")

        st.markdown("---")

        if st.button("🤖 AI 분석 시작", type="primary", use_container_width=True):
            db = load_db()
            start_id = len(db) + 1
            results = []

            progress = st.progress(0)
            status = st.empty()
            total = len(uploaded_files)

            for i, uploaded_file in enumerate(uploaded_files):
                fabric_id = f"FAB-{str(start_id + i).zfill(3)}"
                status.markdown(f"⏳ **[{i+1}/{total}]** 분석 중: `{uploaded_file.name}`")

                # 임시 저장 후 분석
                tmp_path = f"data/tmp_{fabric_id}.jpg"
                with open(tmp_path, "wb") as f:
                    f.write(uploaded_file.getbuffer())

                result = analyze_fabric_image(tmp_path, fabric_id)
                os.remove(tmp_path)

                # 이미지 영구 저장
                img_filename = f"{fabric_id}_{uploaded_file.name}"
                img_save_path = f"data/images/{img_filename}"
                with open(img_save_path, "wb") as f:
                    uploaded_file.seek(0)
                    f.write(uploaded_file.getbuffer())
                result["source_image"] = img_filename

                results.append(result)
                progress.progress((i + 1) / total)

                if i < total - 1:
                    time.sleep(1.0)

            db.extend(results)
            save_db(db)
            status.empty()
            progress.empty()

            # 성공/실패 집계
            success = [r for r in results if r.get("name") != "분석 실패"]
            failed = [r for r in results if r.get("name") == "분석 실패"]

            st.success(f"✅ 분석 완료 — 성공 **{len(success)}개** / 실패 **{len(failed)}개**")

            # 결과 미리보기
            st.markdown("#### 분석 결과")
            for r in results:
                is_fail = r.get("name") == "분석 실패"
                with st.expander(
                    f"{'❌' if is_fail else '✅'} **{r.get('id')}** | "
                    f"{r.get('vendor', '')} {r.get('item_code', '')} — {r.get('name', '분석실패')}"
                ):
                    if not is_fail:
                        col1, col2 = st.columns([1, 2])
                        with col1:
                            img_path = f"data/images/{r.get('source_image', '')}"
                            if os.path.exists(img_path):
                                st.image(img_path, use_container_width=True)
                        with col2:
                            st.markdown(f"""
                            <div class="spec-row"><span class="spec-key">품번</span><span class="spec-val">{r.get('item_code', '-')}</span></div>
                            <div class="spec-row"><span class="spec-key">업체</span><span class="spec-val">{r.get('vendor', '-')}</span></div>
                            <div class="spec-row"><span class="spec-key">혼용률</span><span class="spec-val">{r.get('composition', '-')}</span></div>
                            <div class="spec-row"><span class="spec-key">평량</span><span class="spec-val">{r.get('weight', '-')}</span></div>
                            <div class="spec-row"><span class="spec-key">폭</span><span class="spec-val">{r.get('width', '-')}</span></div>
                            <div class="spec-row"><span class="spec-key">가공</span><span class="spec-val">{r.get('finish', '-')}</span></div>
                            """, unsafe_allow_html=True)
                            chars = r.get('characteristics', [])
                            if chars:
                                badges = "".join([f'<span class="badge">{c}</span>' for c in chars])
                                st.markdown(f'<div style="margin-top:12px">{badges}</div>', unsafe_allow_html=True)
                    else:
                        st.error(f"분석 실패: {r.get('error', '알 수 없는 오류')}")

            # 벡터스토어 자동 재빌드
            with st.spinner("🔍 검색 인덱스 업데이트 중..."):
                build_vectorstore(DB_PATH)
            st.success("🔍 검색 인덱스 업데이트 완료!")

            # Neo4j Knowledge Graph 자동 동기화
            try:
                with st.spinner("🕸️ 지식 그래프 업데이트 중..."):
                    graph_count = build_knowledge_graph(DB_PATH)
                st.success(f"🕸️ 지식 그래프 업데이트 완료! ({graph_count}개 원단)")
            except Exception as e:
                st.warning(f"Neo4j 지식 그래프 업데이트는 건너뛰었습니다: {e}")

# ── 탭2: 등록 현황 ────────────────────────────────────
with tab2:
    db = load_db()
    valid_db = [f for f in db if f.get("name") != "분석 실패"]

    if not valid_db:
        st.markdown("""
        <div style="text-align:center; padding:60px; color:#444;">
            <div style="font-size:48px; margin-bottom:16px;">📭</div>
            <div style="font-size:16px;">아직 등록된 원단이 없습니다.</div>
            <div style="font-size:13px; margin-top:8px; color:#333;">업로드 탭에서 원단 사진을 등록해주세요.</div>
        </div>
        """, unsafe_allow_html=True)
    else:
        col_search, col_filter = st.columns([3, 1])
        with col_search:
            keyword = st.text_input("🔎 검색", placeholder="원단명, 품번, 업체명, 카테고리...")
        with col_filter:
            all_cats = sorted(set(get_category(f) or "기타" for f in valid_db))
            selected_cat = st.selectbox("카테고리", ["전체"] + all_cats)

        filtered = valid_db
        if keyword:
            filtered = [f for f in filtered if keyword.lower() in json.dumps(f, ensure_ascii=False).lower()]
        if selected_cat != "전체":
            filtered = [f for f in filtered if get_category(f) == selected_cat]

        st.markdown(f"**{len(filtered)}개** 원단")
        st.markdown("---")

        for fabric in filtered:
            with st.expander(
                f"**{fabric.get('id', '?')}** | "
                f"{fabric.get('vendor', '')} `{fabric.get('item_code', '')}` — "
                f"{fabric.get('name', '알수없음')} | {get_category(fabric)}"
            ):
                col1, col2 = st.columns([1, 2])
                with col1:
                    img_path = f"data/images/{fabric.get('source_image', '')}"
                    if os.path.exists(img_path):
                        st.image(img_path, use_container_width=True)
                    else:
                        st.caption("이미지 없음")
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
                    <div style="margin-top:16px; padding:12px; background:#0d0d0d; border-radius:8px; font-size:13px; color:#666; line-height:1.6;">
                        {fabric.get('description', '-')}
                    </div>
                    """, unsafe_allow_html=True)
