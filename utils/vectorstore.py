import json
import os
import faiss
import numpy as np
import pickle
from sentence_transformers import SentenceTransformer

VECTORSTORE_PATH = "vectorstore/faiss_index"
METADATA_PATH = "vectorstore/metadata.pkl"
MODEL_NAME = "BAAI/bge-m3"

_embedder = None

def get_embedder():
    global _embedder
    if _embedder is None:
        print("BGE-M3 모델 로딩 중...")
        _embedder = SentenceTransformer(MODEL_NAME)
    return _embedder


def fabric_to_text(fabric: dict) -> str:
    """원단 데이터를 검색용 텍스트로 변환"""
    parts = []

    fields = [
        ("vendor",              "업체"),
        ("item_code",           "품번"),
        ("name",                "원단명"),
        ("category",            "카테고리"),
        ("composition",         "혼용률"),
        ("weight",              "평량"),
        ("width",               "폭"),
        ("finish",              "가공"),
        ("texture_description", "질감"),
        ("description",         "설명"),
    ]

    for key, label in fields:
        val = fabric.get(key)
        if val and val != "판독불가":
            parts.append(f"{label}: {val}")

    # 리스트 필드
    if fabric.get("characteristics"):
        parts.append(f"특성: {', '.join(fabric['characteristics'])}")
    if fabric.get("suitable_for"):
        parts.append(f"적합복종: {', '.join(fabric['suitable_for'])}")
    if fabric.get("season"):
        parts.append(f"시즌: {', '.join(fabric['season'])}")
    if fabric.get("color_options"):
        parts.append(f"색상: {', '.join(fabric['color_options'])}")

    return " | ".join(parts)


def build_vectorstore(fabric_db_path: str = "data/fabric_db.json") -> int:
    """fabric_db.json → BGE-M3 임베딩 → FAISS HNSW 인덱스 구축"""
    os.makedirs("vectorstore", exist_ok=True)

    with open(fabric_db_path, "r", encoding="utf-8") as f:
        fabrics = json.load(f)

    # 오류 항목 제외
    valid = [f for f in fabrics if f.get("name") != "분석 실패" and "error" not in f]
    print(f"유효한 원단: {len(valid)}개")

    if not valid:
        return 0

    embedder = get_embedder()
    texts = [fabric_to_text(f) for f in valid]

    # BGE-M3 임베딩 생성
    print("임베딩 생성 중...")
    embeddings = embedder.encode(
        texts,
        show_progress_bar=True,
        normalize_embeddings=True,  # 코사인 유사도용 정규화
        batch_size=16
    )
    embeddings = np.array(embeddings, dtype=np.float32)

    dim = embeddings.shape[1]  # BGE-M3: 1024차원

    # FAISS HNSW 인덱스 구축
    # M=32: 각 노드당 연결 수 (높을수록 정확하지만 메모리 사용 증가)
    # efConstruction=200: 인덱스 구축 시 탐색 범위
    print("HNSW 인덱스 구축 중...")
    index = faiss.IndexHNSWFlat(dim, 32)
    index.hnsw.efConstruction = 200
    index.hnsw.efSearch = 50  # 검색 시 탐색 범위
    index.add(embeddings)

    # 저장
    faiss.write_index(index, VECTORSTORE_PATH)
    with open(METADATA_PATH, "wb") as f:
        pickle.dump(valid, f)

    print(f"✅ HNSW 인덱스 저장 완료: {len(valid)}개 원단, {dim}차원")
    return len(valid)


def search(query: str, top_k: int = 5) -> list:
    """쿼리 → BGE-M3 임베딩 → HNSW 검색 → 유사 원단 반환"""
    if not os.path.exists(VECTORSTORE_PATH):
        raise FileNotFoundError("벡터스토어가 없습니다. build_vectorstore()를 먼저 실행하세요.")

    index = faiss.read_index(VECTORSTORE_PATH)
    with open(METADATA_PATH, "rb") as f:
        fabrics = pickle.load(f)

    embedder = get_embedder()
    query_vec = embedder.encode(
        [query],
        normalize_embeddings=True,
        show_progress_bar=False
    )
    query_vec = np.array(query_vec, dtype=np.float32)

    scores, indices = index.search(query_vec, top_k)

    results = []
    for score, idx in zip(scores[0], indices[0]):
        if 0 <= idx < len(fabrics):
            fabric = fabrics[idx].copy()
            fabric["similarity_score"] = float(score)
            results.append(fabric)

    return results


def add_fabric(fabric: dict, db_path: str = "data/fabric_db.json"):
    """새 원단 추가 후 인덱스 재빌드"""
    if os.path.exists(db_path):
        with open(db_path, "r", encoding="utf-8") as f:
            fabrics = json.load(f)
    else:
        fabrics = []

    fabrics.append(fabric)

    with open(db_path, "w", encoding="utf-8") as f:
        json.dump(fabrics, f, ensure_ascii=False, indent=2)

    return build_vectorstore(db_path)
