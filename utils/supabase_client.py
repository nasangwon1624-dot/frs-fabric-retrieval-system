import mimetypes
import os
from typing import Any, Dict, List

from dotenv import load_dotenv

try:
    from supabase import Client, create_client
except ImportError:  # pragma: no cover - shown as a runtime fallback in Streamlit
    Client = None
    create_client = None


load_dotenv()

SUPABASE_URL = os.getenv("SUPABASE_URL")
SUPABASE_KEY = os.getenv("SUPABASE_KEY")
BUCKET_NAME = "fabric-images"

_client = None


def get_supabase_client():
    """Supabase client singleton."""
    global _client
    if create_client is None:
        raise ImportError("supabase 패키지가 설치되어 있지 않습니다. `pip install supabase`를 실행하세요.")
    if not SUPABASE_URL or not SUPABASE_KEY:
        raise ValueError("SUPABASE_URL, SUPABASE_KEY 환경변수를 설정하세요.")
    if _client is None:
        _client = create_client(SUPABASE_URL, SUPABASE_KEY)
    return _client


def _as_list(value: Any) -> List[str]:
    if value is None:
        return []
    if isinstance(value, list):
        return value
    if isinstance(value, tuple):
        return list(value)
    text = str(value).strip()
    return [text] if text else []


def _fabric_payload(fabric: Dict[str, Any]) -> Dict[str, Any]:
    payload = fabric.copy()
    for key in ["characteristics", "suitable_for", "color_options", "season"]:
        payload[key] = _as_list(payload.get(key))
    return payload


def save_fabric(fabric: Dict[str, Any]) -> Dict[str, Any]:
    """fabrics 테이블에 원단 1개를 upsert합니다."""
    client = get_supabase_client()
    payload = _fabric_payload(fabric)
    response = client.table("fabrics").upsert(payload, on_conflict="id").execute()
    if response.data:
        return response.data[0]
    return payload


def load_all_fabrics() -> List[Dict[str, Any]]:
    """fabrics 테이블 전체를 조회해 list[dict]로 반환합니다."""
    client = get_supabase_client()
    response = client.table("fabrics").select("*").order("id").execute()
    return response.data or []


def upload_image(file_bytes: bytes, filename: str) -> str:
    """fabric-images 버킷에 이미지를 업로드하고 public URL을 반환합니다."""
    client = get_supabase_client()
    content_type = mimetypes.guess_type(filename)[0] or "application/octet-stream"
    client.storage.from_(BUCKET_NAME).upload(
        filename,
        file_bytes,
        file_options={
            "content-type": content_type,
            "upsert": "true",
        },
    )
    return get_image_url(filename)


def get_image_url(filename: str) -> str:
    """fabric-images 버킷의 public URL을 반환합니다."""
    client = get_supabase_client()
    return client.storage.from_(BUCKET_NAME).get_public_url(filename)
