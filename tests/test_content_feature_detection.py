from pathlib import Path

import pytest

PIL = pytest.importorskip("PIL")
pytest.importorskip("cv2")

from PIL import Image, ImageDraw, ImageFont

from ocr97 import local_inference
from ocr97 import dual_tool


def _font():
    try:
        return ImageFont.truetype("arial.ttf", 22)
    except Exception:
        return ImageFont.load_default()


def _save(img: Image.Image, tmp_path: Path, name: str) -> Path:
    path = tmp_path / name
    img.save(path)
    return path


def test_generic_goal_detects_chart_from_pixels(tmp_path: Path):
    img = Image.new("RGB", (720, 520), "white")
    draw = ImageDraw.Draw(img)
    draw.line((90, 430, 640, 430), fill="black", width=4)
    draw.line((90, 430, 90, 80), fill="black", width=4)
    for idx, height in enumerate([110, 220, 155, 285]):
        x0 = 150 + idx * 95
        draw.rectangle((x0, 430 - height, x0 + 48, 430), outline="black", fill=(45, 120, 210), width=2)
    draw.text((145, 455), "Q1     Q2     Q3     Q4", fill="black", font=_font())
    path = _save(img, tmp_path, "chart.png")

    features = local_inference.classify_document_features(path, goal="extract text from this document")

    assert features["ok"] is True
    assert features["has_chart"] is True
    assert features["layout_class"] == "chart_or_figure"


def test_generic_goal_detects_handwriting_like_strokes(tmp_path: Path):
    img = Image.new("RGB", (760, 520), "white")
    draw = ImageDraw.Draw(img)
    y = 80
    for line_idx in range(8):
        x = 70
        for word_idx in range(5):
            points = []
            for step in range(34):
                points.append((x + step * 2, y + int(7 * __import__("math").sin((step + word_idx) / 2.3))))
            draw.line(points, fill="black", width=3)
            draw.arc((x + 20, y - 10, x + 82, y + 24), 185, 350, fill="black", width=2)
            x += 118 + (word_idx % 3) * 9
        y += 46 + (line_idx % 2) * 5
    path = _save(img, tmp_path, "handwriting.png")

    features = local_inference.classify_document_features(path, goal="extract text from this document")

    assert features["ok"] is True
    assert features["has_handwriting"] is True
    assert features["layout_class"] == "handwritten"


def test_typed_page_does_not_trigger_specialized_lanes(tmp_path: Path):
    img = Image.new("RGB", (760, 520), "white")
    draw = ImageDraw.Draw(img)
    font = _font()
    for idx in range(12):
        draw.text((70, 70 + idx * 31), f"This is a normal typed OCR line number {idx}.", fill="black", font=font)
    path = _save(img, tmp_path, "typed.png")

    features = local_inference.classify_document_features(path, goal="extract text from this document")

    assert features["ok"] is True
    assert features["has_chart"] is False
    assert features["has_handwriting"] is False
    assert features["forms_or_checkboxes"] is False


def test_visual_control_detector_returns_checked_and_unchecked(tmp_path: Path):
    img = Image.new("RGB", (520, 260), "white")
    draw = ImageDraw.Draw(img)
    font = _font()
    draw.rectangle((60, 60, 86, 86), outline="black", width=3)
    draw.line((65, 72, 73, 82), fill="black", width=4)
    draw.line((73, 82, 84, 64), fill="black", width=4)
    draw.text((105, 57), "Enable morning pull", fill="black", font=font)
    draw.rectangle((60, 130, 86, 156), outline="black", width=3)
    draw.text((105, 127), "Send extra alert", fill="black", font=font)
    path = _save(img, tmp_path, "form.png")

    result = local_inference.detect_visual_controls(path)
    states = {item["state"] for item in result["controls"]}

    assert result["ok"] is True
    assert "checked" in states
    assert "unchecked" in states


def test_table_grid_is_not_misclassified_as_checkboxes(tmp_path: Path):
    img = Image.new("RGB", (520, 360), "white")
    draw = ImageDraw.Draw(img)
    for x in range(60, 461, 100):
        draw.line((x, 60, x, 260), fill="black", width=2)
    for y in range(60, 261, 50):
        draw.line((60, y, 460, y), fill="black", width=2)
    path = _save(img, tmp_path, "table.png")

    result = local_inference.detect_visual_controls(path)

    assert result["ok"] is True
    assert result["controls"] == []


def test_content_features_drive_dual_tool_doc_class():
    doc_class = dual_tool._classify_doc_type(
        Path("example.png"),
        "extract text from this document",
        document_features={"ok": True, "has_chart": True, "layout_class": "chart_or_figure"},
    )

    assert doc_class == "chart_or_figure"
