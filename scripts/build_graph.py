import argparse
import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from utils.knowledge_graph import build_knowledge_graph  # noqa: E402


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Build the Neo4j knowledge graph from data/fabric_db.json."
    )
    parser.add_argument(
        "--db-path",
        default=str(PROJECT_ROOT / "data" / "fabric_db.json"),
        help="Path to fabric_db.json.",
    )
    parser.add_argument(
        "--reset",
        action="store_true",
        help="Delete existing FRS graph nodes before importing.",
    )
    args = parser.parse_args()

    db_path = Path(args.db_path)
    if not db_path.exists():
        print(f"fabric DB not found: {db_path}", file=sys.stderr)
        return 1

    try:
        count = build_knowledge_graph(str(db_path), reset=args.reset)
    except Exception as e:
        print(f"Failed to build Neo4j knowledge graph: {e}", file=sys.stderr)
        return 1

    print(f"Neo4j knowledge graph build complete: {count} fabrics imported.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
