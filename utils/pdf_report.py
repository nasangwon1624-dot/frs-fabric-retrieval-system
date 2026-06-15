from datetime import datetime
from html import escape
from io import BytesIO
from pathlib import Path
import re
from typing import Any, Dict, List

from reportlab.lib import colors
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
from reportlab.lib.units import mm
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.ttfonts import TTFont
from reportlab.platypus import Paragraph, SimpleDocTemplate, Spacer, Table, TableStyle


FONT_NAME = "FRSKorean"
FALLBACK_FONT = "Helvetica"
PROJECT_ROOT = Path(__file__).resolve().parents[1]


def register_korean_font() -> str:
    """번들/시스템 나눔고딕을 찾아 reportlab에 등록하고 폰트명을 반환합니다."""
    font_candidates = [
        PROJECT_ROOT / "fonts" / "NanumGothic.ttf",
        PROJECT_ROOT / "fonts" / "NanumGothic-Regular.ttf",
        Path("/usr/share/fonts/truetype/nanum/NanumGothic.ttf"),
        Path("/usr/share/fonts/truetype/nanum/NanumGothicBold.ttf"),
        Path("/usr/share/fonts/truetype/nanum/NanumBarunGothic.ttf"),
        Path("/usr/share/fonts/truetype/nanum/NanumGothicCoding.ttf"),
        Path("/usr/share/fonts/truetype/nanumfont/NanumGothic.ttf"),
        Path("/usr/share/fonts/truetype/nanumfont/NanumBarunGothic.ttf"),
        Path("C:/Windows/Fonts/NanumGothic.ttf"),
        Path("C:/Windows/Fonts/malgun.ttf"),
        Path("/Library/Fonts/NanumGothic.ttf"),
        Path.home() / "Library/Fonts/NanumGothic.ttf",
    ]

    for font_path in font_candidates:
        if font_path.exists():
            try:
                pdfmetrics.registerFont(TTFont(FONT_NAME, str(font_path)))
                return FONT_NAME
            except Exception:
                continue

    return FALLBACK_FONT


def _safe_text(value: Any, default: str = "-") -> str:
    if value is None:
        return default
    if isinstance(value, list):
        text = ", ".join(str(item) for item in value if str(item).strip())
    else:
        text = str(value)
    text = text.strip()
    return text if text else default


def _paragraph(text: Any, style: ParagraphStyle) -> Paragraph:
    return Paragraph(escape(_safe_text(text)).replace("\n", "<br/>"), style)


def _clean_markdown_line(line: str) -> str:
    line = line.strip()
    line = re.sub(r"^#{1,6}\s*", "", line)
    line = re.sub(r"^[-*_]{3,}$", "", line)
    line = re.sub(r"^\s*[-*+]\s+", "• ", line)
    line = re.sub(r"^\s*(\d+)\.\s+", r"\1. ", line)
    line = re.sub(r"\*\*(.*?)\*\*", r"\1", line)
    line = re.sub(r"__(.*?)__", r"\1", line)
    line = re.sub(r"`([^`]*)`", r"\1", line)
    line = re.sub(r"\s{2,}", " ", line)
    return line.strip()


def _markdown_blocks(text: Any) -> List[str]:
    raw = _safe_text(text)
    blocks = []
    current = []

    for line in raw.splitlines():
        cleaned = _clean_markdown_line(line)
        if not cleaned:
            if current:
                blocks.append(" ".join(current))
                current = []
            continue

        starts_new_block = cleaned.startswith("• ") or re.match(r"^\d+\.\s+", cleaned)
        if starts_new_block and current:
            blocks.append(" ".join(current))
            current = [cleaned]
        else:
            current.append(cleaned)

    if current:
        blocks.append(" ".join(current))

    return blocks or ["-"]


def _markdown_flowables(text: Any, style: ParagraphStyle) -> List[Any]:
    flowables = []
    for block in _markdown_blocks(text):
        flowables.append(Paragraph(escape(block), style))
        flowables.append(Spacer(1, 4))
    return flowables


def _score_percent(fabric: Dict[str, Any]) -> str:
    score = fabric.get("similarity_score")
    if score is None:
        return "-"
    try:
        score_pct = max(0, min(100, float(score) * 100))
        return f"{score_pct:.0f}%"
    except (TypeError, ValueError):
        return "-"


def _build_styles(font_name: str) -> Dict[str, ParagraphStyle]:
    base = getSampleStyleSheet()
    return {
        "title": ParagraphStyle(
            "FRSTitle",
            parent=base["Title"],
            fontName=font_name,
            fontSize=20,
            leading=26,
            textColor=colors.HexColor("#111111"),
            spaceAfter=8,
        ),
        "subtitle": ParagraphStyle(
            "FRSSubtitle",
            parent=base["Normal"],
            fontName=font_name,
            fontSize=9,
            leading=13,
            textColor=colors.HexColor("#666666"),
            spaceAfter=16,
        ),
        "heading": ParagraphStyle(
            "FRSHeading",
            parent=base["Heading2"],
            fontName=font_name,
            fontSize=13,
            leading=18,
            textColor=colors.HexColor("#222222"),
            spaceBefore=10,
            spaceAfter=8,
        ),
        "body": ParagraphStyle(
            "FRSBody",
            parent=base["BodyText"],
            fontName=font_name,
            fontSize=9,
            leading=14,
            textColor=colors.HexColor("#333333"),
        ),
        "table_header": ParagraphStyle(
            "FRSTableHeader",
            parent=base["BodyText"],
            fontName=font_name,
            fontSize=8,
            leading=11,
            textColor=colors.white,
        ),
        "footer": ParagraphStyle(
            "FRSFooter",
            parent=base["Normal"],
            fontName=font_name,
            fontSize=8,
            leading=10,
            textColor=colors.HexColor("#777777"),
            alignment=1,
        ),
    }


def _fabric_table(fabrics: List[Dict[str, Any]], styles: Dict[str, ParagraphStyle]) -> Table:
    headers = ["ID", "품번", "업체", "혼용률", "평량", "폭", "가공", "유사도"]
    rows = [[_paragraph(header, styles["table_header"]) for header in headers]]

    for fabric in fabrics:
        rows.append([
            _paragraph(fabric.get("id"), styles["body"]),
            _paragraph(fabric.get("item_code"), styles["body"]),
            _paragraph(fabric.get("vendor"), styles["body"]),
            _paragraph(fabric.get("composition"), styles["body"]),
            _paragraph(fabric.get("weight"), styles["body"]),
            _paragraph(fabric.get("width"), styles["body"]),
            _paragraph(fabric.get("finish"), styles["body"]),
            _paragraph(_score_percent(fabric), styles["body"]),
        ])

    table = Table(
        rows,
        colWidths=[18 * mm, 25 * mm, 24 * mm, 34 * mm, 20 * mm, 18 * mm, 22 * mm, 16 * mm],
        repeatRows=1,
    )
    table.setStyle(TableStyle([
        ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#111111")),
        ("TEXTCOLOR", (0, 0), (-1, 0), colors.white),
        ("GRID", (0, 0), (-1, -1), 0.25, colors.HexColor("#cccccc")),
        ("VALIGN", (0, 0), (-1, -1), "TOP"),
        ("BACKGROUND", (0, 1), (-1, -1), colors.HexColor("#fafafa")),
        ("LEFTPADDING", (0, 0), (-1, -1), 4),
        ("RIGHTPADDING", (0, 0), (-1, -1), 4),
        ("TOPPADDING", (0, 0), (-1, -1), 5),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 5),
    ]))
    return table


def generate_buyer_report_pdf(result: Dict[str, Any]) -> bytes:
    """바이어 검색 결과를 PDF 소재 제안 리포트로 생성해 bytes로 반환합니다."""
    font_name = register_korean_font()
    styles = _build_styles(font_name)
    buffer = BytesIO()

    doc = SimpleDocTemplate(
        buffer,
        pagesize=A4,
        rightMargin=16 * mm,
        leftMargin=16 * mm,
        topMargin=18 * mm,
        bottomMargin=16 * mm,
        title="FRS 소재 제안 리포트",
    )

    generated_at = datetime.now().strftime("%Y-%m-%d %H:%M")
    story = [
        Paragraph("FRS 소재 제안 리포트", styles["title"]),
        Paragraph(f"생성 날짜: {generated_at}", styles["subtitle"]),
        Paragraph("검색 쿼리 (디자이너 요청사항)", styles["heading"]),
        _paragraph(result.get("query", "-"), styles["body"]),
        Spacer(1, 8),
        Paragraph("AI 추천 답변", styles["heading"]),
    ]
    story.extend(_markdown_flowables(result.get("answer", "-"), styles["body"]))
    story.extend([
        Spacer(1, 8),
        Paragraph("참조 원단 목록", styles["heading"]),
    ])

    fabrics = result.get("retrieved_fabrics", [])
    if fabrics:
        story.append(_fabric_table(fabrics, styles))
    else:
        story.append(_paragraph("참조 원단 데이터가 없습니다.", styles["body"]))

    story.extend([
        Spacer(1, 14),
        Paragraph("FRS v1.0", styles["footer"]),
    ])

    doc.build(story)
    buffer.seek(0)
    return buffer.getvalue()
