import json
import os
from typing import Any, Dict, List

import pandas as pd


DEFAULT_DB_PATH = "data/fabric_db.json"
UNREADABLE_VALUES = {"판독불가", "분석 실패", "알수없음", "알 수 없음", ""}

CATEGORY_ALIASES = {
    "아웃도어": "아웃도어/기능성",
    "기능성": "아웃도어/기능성",
    "아웃도어 기능성": "아웃도어/기능성",
    "outdoor": "아웃도어/기능성",
    "functional": "아웃도어/기능성",
    "sports": "스포츠",
    "sport": "스포츠",
    "formal": "포멀",
    "casual": "캐주얼",
    "denim": "데님",
    "knit": "니트",
}

SEASON_ALIASES = {
    "spring": "봄",
    "summer": "여름",
    "fall": "가을",
    "autumn": "가을",
    "winter": "겨울",
    "s/s": "봄/여름",
    "ss": "봄/여름",
    "f/w": "가을/겨울",
    "fw": "가을/겨울",
}


def _as_list(value: Any) -> List[str]:
    if value is None:
        return []
    if isinstance(value, list):
        return [str(item).strip() for item in value if str(item).strip()]
    text = str(value).strip()
    return [text] if text else []


def _first_value(value: Any) -> str:
    values = _as_list(value)
    return values[0] if values else ""


def _is_unreadable(value: Any) -> bool:
    values = _as_list(value)
    if not values:
        return True
    return all(item.strip() in UNREADABLE_VALUES for item in values)


def normalize_category(value: Any) -> str:
    """카테고리 값을 대표 문자열로 정규화합니다."""
    category = _first_value(value)
    if not category or category in UNREADABLE_VALUES:
        return "기타"

    normalized_key = category.lower().replace("-", " ").replace("_", " ").strip()
    if normalized_key in CATEGORY_ALIASES:
        return CATEGORY_ALIASES[normalized_key]
    if "아웃도어" in category or "기능성" in category:
        return "아웃도어/기능성"
    return category


def normalize_season(value: Any) -> List[str]:
    """시즌 값을 봄/여름/가을/겨울 중심 리스트로 정규화합니다."""
    seasons = []
    for item in _as_list(value):
        key = item.lower().replace("-", "/").strip()
        mapped = SEASON_ALIASES.get(key, item)
        if mapped == "봄/여름":
            seasons.extend(["봄", "여름"])
        elif mapped == "가을/겨울":
            seasons.extend(["가을", "겨울"])
        elif mapped not in UNREADABLE_VALUES:
            seasons.append(mapped)
    return list(dict.fromkeys(seasons))


def load_fabric_dataframe(db_path: str = DEFAULT_DB_PATH) -> pd.DataFrame:
    """fabric_db.json을 pandas DataFrame으로 읽습니다."""
    if not os.path.exists(db_path):
        return pd.DataFrame()
    with open(db_path, "r", encoding="utf-8") as f:
        data = json.load(f)
    if not isinstance(data, list):
        return pd.DataFrame()
    return pd.DataFrame(data)


def clean_fabric_dataframe(df: pd.DataFrame) -> pd.DataFrame:
    """판독불가/결측치 처리와 시즌/카테고리 정규화를 적용합니다."""
    if df.empty:
        return df.copy()

    cleaned = df.copy()
    for column in cleaned.columns:
        cleaned[column] = cleaned[column].apply(
            lambda value: "" if _is_unreadable(value) else value
        )

    if "category" in cleaned.columns:
        cleaned["category"] = cleaned["category"].apply(normalize_category)
    else:
        cleaned["category"] = "기타"

    if "season" in cleaned.columns:
        cleaned["season"] = cleaned["season"].apply(normalize_season)
    else:
        cleaned["season"] = [[] for _ in range(len(cleaned))]

    if "vendor" not in cleaned.columns:
        cleaned["vendor"] = ""
    if "item_code" not in cleaned.columns:
        cleaned["item_code"] = ""

    return cleaned


def remove_duplicate_fabrics(df: pd.DataFrame) -> pd.DataFrame:
    """item_code 기준 중복 원단을 제거합니다. 빈 item_code는 유지합니다."""
    if df.empty or "item_code" not in df.columns:
        return df.copy()

    has_code = df["item_code"].astype(str).str.strip() != ""
    with_code = df[has_code].drop_duplicates(subset=["item_code"], keep="first")
    without_code = df[~has_code]
    return pd.concat([with_code, without_code], ignore_index=True)


def get_category_stats(df: pd.DataFrame) -> Dict[str, int]:
    cleaned = clean_fabric_dataframe(df)
    if cleaned.empty:
        return {}
    return cleaned["category"].value_counts().to_dict()


def get_season_stats(df: pd.DataFrame) -> Dict[str, int]:
    cleaned = clean_fabric_dataframe(df)
    if cleaned.empty:
        return {}
    exploded = cleaned.explode("season")
    exploded = exploded[exploded["season"].astype(str).str.strip() != ""]
    return exploded["season"].value_counts().to_dict()


def get_vendor_stats(df: pd.DataFrame) -> Dict[str, int]:
    cleaned = clean_fabric_dataframe(df)
    if cleaned.empty:
        return {}
    vendors = cleaned["vendor"].replace("", "미상")
    return vendors.value_counts().to_dict()


def preprocess_fabric_data(db_path: str = DEFAULT_DB_PATH, remove_duplicates: bool = True) -> pd.DataFrame:
    """JSON을 읽고 정제한 DataFrame을 반환합니다."""
    df = load_fabric_dataframe(db_path)
    cleaned = clean_fabric_dataframe(df)
    if remove_duplicates:
        return remove_duplicate_fabrics(cleaned)
    return cleaned


def get_quality_report(db_path: str = DEFAULT_DB_PATH) -> Dict[str, Any]:
    """사이드바 표시용 데이터 품질 리포트를 반환합니다."""
    raw_df = load_fabric_dataframe(db_path)
    cleaned_df = clean_fabric_dataframe(raw_df)
    deduped_df = remove_duplicate_fabrics(cleaned_df)

    duplicate_count = 0
    if not cleaned_df.empty and "item_code" in cleaned_df.columns:
        has_code = cleaned_df["item_code"].astype(str).str.strip() != ""
        duplicate_count = int(cleaned_df[has_code].duplicated(subset=["item_code"]).sum())

    unreadable_count = 0
    missing_count = 0
    if not raw_df.empty:
        unreadable_mask = raw_df.map(_is_unreadable)
        unreadable_count = int(unreadable_mask.sum().sum())
        missing_count = int(raw_df.isna().sum().sum())

    return {
        "total_count": int(len(raw_df)),
        "valid_count": int(len(cleaned_df)),
        "deduped_count": int(len(deduped_df)),
        "duplicate_count": duplicate_count,
        "unreadable_count": unreadable_count,
        "missing_count": missing_count,
        "category_stats": get_category_stats(cleaned_df),
        "season_stats": get_season_stats(cleaned_df),
        "vendor_stats": get_vendor_stats(cleaned_df),
    }
