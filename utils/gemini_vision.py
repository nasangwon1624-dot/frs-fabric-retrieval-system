from google import genai
import json
import os
import time
from PIL import Image
from dotenv import load_dotenv

load_dotenv()
client = genai.Client(api_key=os.getenv("GEMINI_API_KEY"))

EXTRACT_PROMPT = """
이 이미지는 패션 원단 스와치 시트(swatch sheet) 사진입니다.
이미지에서 텍스트와 원단 정보를 모두 읽어서 아래 JSON 형식으로 추출해주세요.
확인이 불가능한 항목은 "판독불가"로 표기하세요.

반드시 JSON만 출력하고, 다른 텍스트는 절대 포함하지 마세요.

{
  "vendor": "브랜드/업체명 (예: YEJIN F&G, HS Textile, Interfil)",
  "item_code": "품번/아이템코드 (예: YJCSP-23495, ML-QD240K, GT-0279)",
  "name": "원단명 또는 조직명 (예: Jersey, Tweed, Canvas, 판독불가)",
  "category": "카테고리 (아웃도어/기능성, 캐주얼, 포멀, 스포츠, 데님, 니트 등)",
  "composition": "혼용률 (예: Nylon 76% PU 24%, T/R/SP 61/33/6)",
  "weight": "평량 (예: 172 G/SQM, 330 G/SM, 180G/Y)",
  "width": "폭 (예: 56\", 57\", 145cm, 52-54\")",
  "finish": "가공방법 (예: PD, Brush, C.C.F Brush, DWR, 판독불가)",
  "characteristics": ["원단 특성1", "원단 특성2", "원단 특성3"],
  "suitable_for": ["적합 복종1", "적합 복종2"],
  "color_options": ["확인된 색상1", "확인된 색상2"],
  "season": ["봄", "여름", "가을", "겨울" 중 해당하는 것],
  "texture_description": "질감에 대한 상세 설명 (광택, 두께감, 촉감, 표면 특성 등)",
  "description": "전체적인 원단 특성 및 활용 방안 설명 (3-4문장)"
}
"""

def analyze_fabric_image(image_path: str, fabric_id: str) -> dict:
    try:
        img = Image.open(image_path)
        response = client.models.generate_content(
            model="gemini-3.5-flash",
            contents=[EXTRACT_PROMPT, img]
        )
        raw = response.text.strip()

        if raw.startswith("```"):
            raw = raw.split("```")[1]
            if raw.startswith("json"):
                raw = raw[4:]
        raw = raw.strip()

        data = json.loads(raw)
        data["id"] = fabric_id
        data["source_image"] = os.path.basename(image_path)
        data["source_type"] = "image"
        return data

    except json.JSONDecodeError:
        return {
            "id": fabric_id,
            "name": "분석 실패",
            "source_image": os.path.basename(image_path),
            "source_type": "image",
            "error": "JSON 파싱 실패",
            "raw_response": response.text if 'response' in locals() else ""
        }
    except Exception as e:
        return {
            "id": fabric_id,
            "source_image": os.path.basename(image_path),
            "source_type": "image",
            "error": str(e)
        }

def analyze_batch(image_paths: list, start_id: int = 1, delay: float = 1.5) -> list:
    results = []
    total = len(image_paths)
    for i, path in enumerate(image_paths):
        fabric_id = f"FAB-{str(start_id + i).zfill(3)}"
        print(f"[{i+1}/{total}] 분석 중: {os.path.basename(path)} → {fabric_id}")
        result = analyze_fabric_image(path, fabric_id)
        results.append(result)
        if i < total - 1:
            time.sleep(delay)
    return results
