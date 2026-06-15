import json
import os
import re
from typing import Any, Dict, List, Optional

from google import genai
from dotenv import load_dotenv

try:
    from neo4j import GraphDatabase
except ImportError:  # pragma: no cover - handled at runtime for Streamlit clarity
    GraphDatabase = None

load_dotenv()
client = genai.Client(api_key=os.getenv("GEMINI_API_KEY"))

NEO4J_URI = os.getenv("NEO4J_URI")
NEO4J_USER = os.getenv("NEO4J_USER")
NEO4J_PASSWORD = os.getenv("NEO4J_PASSWORD")

DEFAULT_DB_PATH = "data/fabric_db.json"

_driver = None


# Property -> Usage 추론 규칙. fabric_db.json에 없는 Usage도 그래프에 생성됩니다.
PROPERTY_USAGE_RULES: Dict[str, List[str]] = {
    "방풍": ["아웃도어 재킷", "윈드브레이커", "트레킹 재킷"],
    "발수": ["아웃도어 재킷", "레인 재킷", "윈드브레이커"],
    "방수": ["레인 재킷", "아웃도어 재킷", "스포츠 아우터"],
    "경량": ["윈드브레이커", "여름 아우터", "패커블 재킷"],
    "스트레치": ["스포츠웨어", "레깅스", "액티브웨어"],
    "신축성": ["스포츠웨어", "레깅스", "액티브웨어"],
    "흡습속건": ["스포츠웨어", "러닝웨어", "트레이닝복"],
    "통기성": ["여름 셔츠", "스포츠웨어", "이너웨어"],
    "보온": ["겨울 아우터", "플리스", "이너웨어"],
    "기모": ["겨울 이너", "스웨트셔츠", "후디"],
    "드레이프": ["원피스", "블라우스", "스커트"],
    "광택": ["포멀웨어", "블라우스", "드레스"],
    "고시감": ["정장 바지", "재킷", "셔츠"],
    "내구성": ["워크웨어", "가방", "아웃도어 팬츠"],
    "친환경": ["지속가능 패션", "캐주얼웨어", "라이프스타일웨어"],
}

QUERY_EXTRACT_PROMPT = """패션 원단 검색 쿼리에서 그래프 검색에 사용할 키워드를 JSON으로 추출하세요.

반드시 JSON만 출력하세요.

[쿼리]
{query}

출력 형식:
{{
  "categories": ["카테고리"],
  "properties": ["원단 특성"],
  "usages": ["복종/용도"],
  "seasons": ["봄", "여름", "가을", "겨울"],
  "keywords": ["기타 핵심 키워드"]
}}
"""


def _require_driver():
    if GraphDatabase is None:
        raise ImportError(
            "neo4j 패키지가 설치되어 있지 않습니다. `pip install neo4j` 후 다시 실행하세요."
        )
    if not all([NEO4J_URI, NEO4J_USER, NEO4J_PASSWORD]):
        raise ValueError(
            "Neo4j 연결 환경변수 NEO4J_URI, NEO4J_USER, NEO4J_PASSWORD를 설정하세요."
        )


def get_driver():
    """Neo4j 드라이버 싱글톤을 반환합니다."""
    global _driver
    _require_driver()
    if _driver is None:
        _driver = GraphDatabase.driver(
            NEO4J_URI,
            auth=(NEO4J_USER, NEO4J_PASSWORD),
        )
    return _driver


def close_driver() -> None:
    """Streamlit 종료나 테스트 후 Neo4j 연결을 닫습니다."""
    global _driver
    if _driver is not None:
        _driver.close()
        _driver = None


def _as_list(value: Any) -> List[str]:
    if not value:
        return []
    if isinstance(value, list):
        return [str(v).strip() for v in value if str(v).strip()]
    return [str(value).strip()]


def _load_fabrics(fabric_db_path: str) -> List[dict]:
    if not os.path.exists(fabric_db_path):
        return []
    with open(fabric_db_path, "r", encoding="utf-8") as f:
        fabrics = json.load(f)
    return [
        fabric
        for fabric in fabrics
        if fabric.get("id") and fabric.get("name") != "분석 실패" and "error" not in fabric
    ]


def _execute_write(session, func, *args):
    if hasattr(session, "execute_write"):
        return session.execute_write(func, *args)
    return session.write_transaction(func, *args)


def _create_constraints(tx) -> None:
    constraints = [
        "CREATE CONSTRAINT fabric_id IF NOT EXISTS FOR (n:Fabric) REQUIRE n.id IS UNIQUE",
        "CREATE CONSTRAINT category_name IF NOT EXISTS FOR (n:Category) REQUIRE n.name IS UNIQUE",
        "CREATE CONSTRAINT property_name IF NOT EXISTS FOR (n:Property) REQUIRE n.name IS UNIQUE",
        "CREATE CONSTRAINT usage_name IF NOT EXISTS FOR (n:Usage) REQUIRE n.name IS UNIQUE",
        "CREATE CONSTRAINT season_name IF NOT EXISTS FOR (n:Season) REQUIRE n.name IS UNIQUE",
    ]
    for cypher in constraints:
        tx.run(cypher)


def _clear_frs_graph(tx) -> None:
    # FRS가 사용하는 라벨만 정리합니다. 다른 Neo4j 데이터는 건드리지 않습니다.
    tx.run(
        """
        MATCH (n)
        WHERE n:Fabric OR n:Category OR n:Property OR n:Usage OR n:Season
        DETACH DELETE n
        """
    )


def _merge_fabric(tx, fabric: dict) -> None:
    tx.run(
        """
        MERGE (f:Fabric {id: $id})
        SET f.name = $name,
            f.vendor = $vendor,
            f.item_code = $item_code,
            f.composition = $composition,
            f.weight = $weight,
            f.width = $width,
            f.finish = $finish,
            f.source_image = $source_image,
            f.description = $description,
            f.texture_description = $texture_description
        """,
        id=fabric.get("id"),
        name=fabric.get("name", ""),
        vendor=fabric.get("vendor", ""),
        item_code=fabric.get("item_code", ""),
        composition=fabric.get("composition", ""),
        weight=fabric.get("weight", ""),
        width=fabric.get("width", ""),
        finish=fabric.get("finish", ""),
        source_image=fabric.get("source_image", ""),
        description=fabric.get("description", ""),
        texture_description=fabric.get("texture_description", ""),
    )

    category = fabric.get("category")
    if category and category != "판독불가":
        tx.run(
            """
            MATCH (f:Fabric {id: $fabric_id})
            MERGE (c:Category {name: $name})
            MERGE (f)-[:BELONGS_TO]->(c)
            """,
            fabric_id=fabric["id"],
            name=category,
        )

    for prop in _as_list(fabric.get("characteristics")):
        if prop == "판독불가":
            continue
        tx.run(
            """
            MATCH (f:Fabric {id: $fabric_id})
            MERGE (p:Property {name: $name})
            MERGE (f)-[:HAS_PROPERTY]->(p)
            """,
            fabric_id=fabric["id"],
            name=prop,
        )

    for usage in _as_list(fabric.get("suitable_for")):
        if usage == "판독불가":
            continue
        tx.run(
            """
            MATCH (f:Fabric {id: $fabric_id})
            MERGE (u:Usage {name: $name})
            MERGE (f)-[:SUITABLE_FOR]->(u)
            """,
            fabric_id=fabric["id"],
            name=usage,
        )

    for season in _as_list(fabric.get("season")):
        if season == "판독불가":
            continue
        tx.run(
            """
            MATCH (f:Fabric {id: $fabric_id})
            MERGE (s:Season {name: $name})
            MERGE (f)-[:FOR_SEASON]->(s)
            """,
            fabric_id=fabric["id"],
            name=season,
        )


def _merge_implication_rules(tx) -> None:
    for prop, usages in PROPERTY_USAGE_RULES.items():
        for usage in usages:
            tx.run(
                """
                MERGE (p:Property {name: $property})
                MERGE (u:Usage {name: $usage})
                MERGE (p)-[:IMPLIES]->(u)
                """,
                property=prop,
                usage=usage,
            )


def build_knowledge_graph(
    fabric_db_path: str = DEFAULT_DB_PATH,
    reset: bool = False,
) -> int:
    """fabric_db.json을 Neo4j 그래프로 동기화합니다.

    rag_chain.py에서는 원단 등록 후 `build_knowledge_graph()`를 호출하면 됩니다.
    """
    fabrics = _load_fabrics(fabric_db_path)
    driver = get_driver()

    with driver.session() as session:
        _execute_write(session, _create_constraints)
        if reset:
            _execute_write(session, _clear_frs_graph)
        for fabric in fabrics:
            _execute_write(session, _merge_fabric, fabric)
        _execute_write(session, _merge_implication_rules)

    return len(fabrics)


def extract_graph_terms(query: str) -> Dict[str, List[str]]:
    """Gemini로 자연어 쿼리에서 그래프 검색 키워드를 추출합니다."""
    try:
        response = client.models.generate_content(
            model="gemini-3.5-flash",
            contents=QUERY_EXTRACT_PROMPT.format(query=query)
        )
        raw = response.text.strip()
        if raw.startswith("```"):
            raw = raw.split("```")[1]
            if raw.startswith("json"):
                raw = raw[4:]
        parsed = json.loads(raw.strip())
    except Exception:
        parsed = {}

    fallback = _fallback_extract_terms(query)
    return {
        "categories": _unique_terms(parsed.get("categories")) or fallback["categories"],
        "properties": _unique_terms(parsed.get("properties")) or fallback["properties"],
        "usages": _unique_terms(parsed.get("usages")) or fallback["usages"],
        "seasons": _unique_terms(parsed.get("seasons")) or fallback["seasons"],
        "keywords": _unique_terms(parsed.get("keywords")) or fallback["keywords"],
    }


def _unique_terms(value: Any) -> List[str]:
    seen = set()
    terms = []
    for item in _as_list(value):
        normalized = item.strip()
        if normalized and normalized not in seen:
            seen.add(normalized)
            terms.append(normalized)
    return terms


def _fallback_extract_terms(query: str) -> Dict[str, List[str]]:
    q = query.lower()
    properties = [p for p in PROPERTY_USAGE_RULES if p.lower() in q]
    usages = sorted({u for p in properties for u in PROPERTY_USAGE_RULES[p]})
    seasons = [s for s in ["봄", "여름", "가을", "겨울"] if s in query]

    category_candidates = ["아웃도어", "기능성", "캐주얼", "포멀", "스포츠", "데님", "니트"]
    categories = [c for c in category_candidates if c in query]

    keywords = [
        token
        for token in re.split(r"[\s,./+()\-\[\]]+", query)
        if len(token.strip()) >= 2
    ][:12]

    return {
        "categories": categories,
        "properties": properties,
        "usages": usages,
        "seasons": seasons,
        "keywords": keywords,
    }


def infer_usages_from_properties(properties: List[str]) -> List[str]:
    """특성 목록에서 Property-[:IMPLIES]->Usage 추론 결과를 반환합니다."""
    inferred = []
    for prop in properties:
        inferred.extend(PROPERTY_USAGE_RULES.get(prop, []))
    return _unique_terms(inferred)


def graph_search(query: str, top_k: int = 5) -> List[dict]:
    """자연어 쿼리로 Neo4j 그래프를 검색하고 추론된 원단 후보를 반환합니다."""
    terms = extract_graph_terms(query)
    inferred_usages = infer_usages_from_properties(terms["properties"])
    usage_terms = _unique_terms(terms["usages"] + inferred_usages)

    driver = get_driver()
    with driver.session() as session:
        records = session.run(
            """
            MATCH (f:Fabric)
            WITH
                f,
                [(f)-[:BELONGS_TO]->(c:Category) | c.name] AS categories,
                [(f)-[:HAS_PROPERTY]->(p:Property) | p.name] AS properties,
                [(f)-[:SUITABLE_FOR]->(u:Usage) | u.name] AS usages,
                [(f)-[:FOR_SEASON]->(s:Season) | s.name] AS seasons,
                [(f)-[:HAS_PROPERTY]->(:Property)-[:IMPLIES]->(iu:Usage) | iu.name] AS inferred_usages
            WITH
                f, categories, properties, usages, seasons, inferred_usages,
                size([x IN categories WHERE x IN $categories]) * 3 +
                size([x IN properties WHERE x IN $properties]) * 4 +
                size([x IN usages WHERE x IN $usages]) * 4 +
                size([x IN inferred_usages WHERE x IN $usages]) * 2 +
                size([x IN seasons WHERE x IN $seasons]) * 2 +
                size([
                    k IN $keywords
                    WHERE toLower(coalesce(f.name, "")) CONTAINS toLower(k)
                       OR toLower(coalesce(f.vendor, "")) CONTAINS toLower(k)
                       OR toLower(coalesce(f.item_code, "")) CONTAINS toLower(k)
                       OR toLower(coalesce(f.description, "")) CONTAINS toLower(k)
                       OR toLower(coalesce(f.texture_description, "")) CONTAINS toLower(k)
                ]) AS graph_score
            WHERE graph_score > 0
            RETURN
                f {.*} AS fabric,
                graph_score,
                categories,
                properties,
                usages,
                seasons,
                inferred_usages
            ORDER BY graph_score DESC, f.id ASC
            LIMIT $top_k
            """,
            categories=terms["categories"],
            properties=terms["properties"],
            usages=usage_terms,
            seasons=terms["seasons"],
            keywords=terms["keywords"],
            top_k=top_k,
        )

        results = []
        for record in records:
            fabric = dict(record["fabric"])
            fabric["graph_score"] = int(record["graph_score"])
            fabric["matched_graph_terms"] = {
                "categories": _intersect(record["categories"], terms["categories"]),
                "properties": _intersect(record["properties"], terms["properties"]),
                "usages": _intersect(record["usages"], usage_terms),
                "seasons": _intersect(record["seasons"], terms["seasons"]),
                "inferred_usages": _intersect(record["inferred_usages"], usage_terms),
            }
            fabric["query_terms"] = terms
            results.append(fabric)

    return results


def _intersect(values: Optional[List[str]], targets: Optional[List[str]]) -> List[str]:
    value_set = set(values or [])
    return [target for target in targets or [] if target in value_set]


def query_graph_rag(query: str, top_k: int = 5) -> dict:
    """rag_chain.py에서 호출하기 좋은 형태의 GraphRAG 검색 API입니다."""
    terms = extract_graph_terms(query)
    return {
        "query": query,
        "query_terms": terms,
        "inferred_usages": infer_usages_from_properties(terms["properties"]),
        "retrieved_fabrics": graph_search(query, top_k=top_k),
    }
