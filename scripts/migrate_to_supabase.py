import json
import shutil
import sys
from pathlib import Path
from typing import Dict, List, Tuple


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from utils.supabase_client import save_fabric, upload_image  # noqa: E402


DB_PATH = PROJECT_ROOT / "data" / "fabric_db.json"
DEMO_PATH = PROJECT_ROOT / "data" / "demo_fabrics.json"
IMAGE_DIR = PROJECT_ROOT / "data" / "images"
DB_BACKUP_PATH = PROJECT_ROOT / "data" / "fabric_db.json.backup"
VECTORSTORE_PATH = PROJECT_ROOT / "vectorstore"
VECTORSTORE_BACKUP_PATH = PROJECT_ROOT / "vectorstore_backup"
MIGRATION_LIMIT = 90


def load_fabrics() -> List[Dict]:
    if not DB_PATH.exists():
        raise FileNotFoundError(f"fabric DB not found: {DB_PATH}")
    with open(DB_PATH, "r", encoding="utf-8") as f:
        data = json.load(f)
    if not isinstance(data, list):
        raise ValueError("fabric_db.json must contain a JSON list.")
    return data


def write_json(path: Path, data) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)


def resolve_image_path(source_image: str) -> Path:
    if not source_image:
        return IMAGE_DIR / ""
    if source_image.startswith(("http://", "https://")):
        return IMAGE_DIR / Path(source_image).name
    return IMAGE_DIR / Path(source_image).name


def migrate_image(fabric: Dict) -> Tuple[Dict, str]:
    migrated = fabric.copy()
    source_image = str(migrated.get("source_image", "")).strip()
    image_path = resolve_image_path(source_image)

    if not source_image:
        return migrated, "source_image 없음"
    if not image_path.exists():
        return migrated, f"이미지 파일 없음: {image_path}"

    filename = f"{migrated.get('id', 'FAB')}_{image_path.name}"
    file_bytes = image_path.read_bytes()
    public_url = upload_image(file_bytes, filename)
    migrated["source_image"] = public_url
    return migrated, ""


def backup_local_files() -> None:
    shutil.copy2(DB_PATH, DB_BACKUP_PATH)
    print(f"DB 백업 완료: {DB_BACKUP_PATH}")

    if VECTORSTORE_PATH.exists():
        if VECTORSTORE_BACKUP_PATH.exists():
            shutil.rmtree(VECTORSTORE_BACKUP_PATH)
        shutil.copytree(VECTORSTORE_PATH, VECTORSTORE_BACKUP_PATH)
        print(f"vectorstore 백업 완료: {VECTORSTORE_BACKUP_PATH}")
    else:
        print("vectorstore 폴더가 없어 백업을 건너뜁니다.")


def main() -> int:
    fabrics = load_fabrics()
    total = len(fabrics)
    migrate_targets = fabrics[:MIGRATION_LIMIT]
    demo_fabrics = fabrics[MIGRATION_LIMIT:]

    print(f"로컬 원단 데이터 로드 완료: 총 {total}개")
    print(f"Supabase 이전 대상: {len(migrate_targets)}개")
    print(f"시연용 분리 대상: {len(demo_fabrics)}개")

    success_count = 0
    failure_count = 0

    for index, fabric in enumerate(migrate_targets, 1):
        fabric_id = fabric.get("id", f"NO-ID-{index}")
        try:
            migrated_fabric, image_warning = migrate_image(fabric)
            if image_warning:
                print(f"[경고] {fabric_id}: {image_warning}")

            save_fabric(migrated_fabric)
            success_count += 1
            print(f"[{index}/{len(migrate_targets)}] {fabric_id} 업로드 완료")
        except Exception as e:
            failure_count += 1
            print(f"[실패] {fabric_id}: {e}", file=sys.stderr)
            continue

    write_json(DEMO_PATH, demo_fabrics)
    print(f"시연용 데이터 저장 완료: {DEMO_PATH}")

    backup_local_files()
    write_json(DB_PATH, [])
    print("원본 fabric_db.json 초기화 완료: []")

    print(
        f"이전 완료: {success_count}개 Supabase, "
        f"{len(demo_fabrics)}개 시연용, {failure_count}개 실패"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
