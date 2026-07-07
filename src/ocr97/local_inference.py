from __future__ import annotations

import hashlib
import importlib
import importlib.util
import json
import os
import re
import threading
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, Dict, Optional

from .legacy_env import apply_legacy_env_aliases
from .paths import ensure_paths

apply_legacy_env_aliases()
_PATHS = ensure_paths()

try:
    import fitz  # type: ignore

    PYMUPDF_AVAILABLE = True
except Exception:
    PYMUPDF_AVAILABLE = False

try:
    import pytesseract

    PYTESSERACT_AVAILABLE = True
except Exception:
    PYTESSERACT_AVAILABLE = False

try:
    from PIL import Image

    PIL_AVAILABLE = True
except Exception:
    PIL_AVAILABLE = False

try:
    import cv2  # type: ignore
    import numpy as np  # type: ignore

    CV2_AVAILABLE = True
except Exception:
    CV2_AVAILABLE = False

torch = None
AutoImageProcessor = None
AutoModelForSequenceClassification = None
AutoTokenizer = None
GotOcr2ForConditionalGeneration = None
GotOcr2Processor = None
TableTransformerForObjectDetection = None
pipeline = None

TRANSFORMERS_AVAILABLE = bool(importlib.util.find_spec("torch")) and bool(importlib.util.find_spec("transformers"))


_GOT_RUNTIME_LOCK = threading.Lock()
_GOT_RUNTIME: Dict[str, Any] = {
    "processor": None,
    "model": None,
    "loaded_model_id": "",
    "last_error": "",
    "backend": "uninitialized",
}

_FINBERT_RUNTIME_LOCK = threading.Lock()
_FINBERT_RUNTIME: Dict[str, Any] = {
    "pipeline": None,
    "loaded_model_id": "",
    "last_error": "",
    "backend": "uninitialized",
}

_TABLE_RUNTIME_LOCK = threading.Lock()
_TABLE_RUNTIME: Dict[str, Any] = {
    "processor": None,
    "model": None,
    "loaded_model_id": "",
    "last_error": "",
    "backend": "uninitialized",
}


def _got_model_id() -> str:
    return str(os.getenv("OCR97_GOT_OCR2_MODEL_ID", "stepfun-ai/GOT-OCR-2.0-hf")).strip()


def _got_backend_enabled() -> bool:
    return str(os.getenv("OCR97_GOT_OCR2_ENABLE_TRANSFORMERS", "1")).strip().lower() in {"1", "true", "yes"}


def _got_device_pref() -> str:
    raw = str(os.getenv("OCR97_GOT_OCR2_DEVICE", "auto")).strip().lower() or "auto"
    if raw in {"cpu", "cuda", "auto"}:
        return raw
    return "auto"


def _got_dtype() -> Any:
    if not TRANSFORMERS_AVAILABLE or not _ensure_transformers_loaded().get("ok"):
        return None
    raw = str(os.getenv("OCR97_GOT_OCR2_DTYPE", "auto")).strip().lower() or "auto"
    if raw in {"float32", "fp32"}:
        return torch.float32
    if raw in {"float16", "fp16"}:
        return torch.float16
    if raw in {"bfloat16", "bf16"}:
        return torch.bfloat16
    if torch.cuda.is_available():
        return torch.float16
    return torch.float32


def _torch_device_choice(raw: str = "auto") -> str:
    runtime = _ensure_transformers_loaded()
    if not runtime.get("ok"):
        return "cpu"
    pref = str(raw or "auto").strip().lower() or "auto"
    if pref == "cuda":
        return "cuda" if torch.cuda.is_available() else "cpu"
    if pref == "cpu":
        return "cpu"
    return "cuda" if torch.cuda.is_available() else "cpu"


def _transformers_device_index(device_name: str) -> int:
    runtime = _ensure_transformers_loaded()
    if not runtime.get("ok"):
        return -1
    return 0 if str(device_name or "").strip().lower() == "cuda" and torch.cuda.is_available() else -1


def _finbert_model_id() -> str:
    return str(os.getenv("OCR97_FINBERT_MODEL_ID", "ProsusAI/finbert")).strip()


def _finbert_device() -> str:
    return _torch_device_choice(os.getenv("OCR97_FINBERT_DEVICE", "cpu"))


def _tableformer_model_id() -> str:
    return str(os.getenv("OCR97_TABLEFORMER_MODEL_ID", "microsoft/table-transformer-structure-recognition-v1.1-all")).strip()


def _tableformer_device() -> str:
    return _torch_device_choice(os.getenv("OCR97_TABLEFORMER_DEVICE", "auto"))


def _call_text_normalizer(fn: Optional[Callable[..., str]], text: str, max_chars: int) -> str:
    if fn is None:
        raw = " ".join(str(text or "").split())
        return raw[: max_chars - 3] + "..." if len(raw) > max_chars and max_chars > 3 else raw[:max_chars]
    return str(fn(text, max_chars=max_chars))


def _call_markdown_normalizer(fn: Optional[Callable[..., str]], text: str, max_chars: int) -> str:
    if fn is None:
        raw = str(text or "").replace("\r\n", "\n").replace("\r", "\n").strip()
        return raw[: max_chars - 3] + "..." if len(raw) > max_chars and max_chars > 3 else raw[:max_chars]
    return str(fn(text, max_chars=max_chars))


def _open_rendered_images(path: Path, max_pages: int) -> list["Image.Image"]:
    images: list["Image.Image"] = []
    if not PIL_AVAILABLE:
        return images
    if str(path.suffix or "").lower() != ".pdf":
        try:
            with Image.open(path) as img:
                images.append(img.convert("RGB"))
        except Exception:
            return []
        return images
    if not PYMUPDF_AVAILABLE:
        return []
    try:
        doc = fitz.open(str(path))
    except Exception:
        return []
    try:
        total = max(1, min(int(max_pages), len(doc)))
        for idx in range(total):
            page = doc.load_page(idx)
            pix = page.get_pixmap(matrix=fitz.Matrix(2.0, 2.0), alpha=False)
            images.append(Image.frombytes("RGB", [pix.width, pix.height], pix.samples))
    finally:
        doc.close()
    return images


def _image_gray_array(image: "Image.Image") -> Optional[Any]:
    if not CV2_AVAILABLE or not PIL_AVAILABLE:
        return None
    try:
        rgb = image.convert("RGB")
        return cv2.cvtColor(np.array(rgb), cv2.COLOR_RGB2GRAY)
    except Exception:
        return None


def _ink_mask(gray: Any) -> Optional[Any]:
    if not CV2_AVAILABLE or gray is None:
        return None
    try:
        blurred = cv2.GaussianBlur(gray, (3, 3), 0)
        _threshold, binary = cv2.threshold(blurred, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
        return binary
    except Exception:
        return None


def _bbox_iou(a: list[int], b: list[int]) -> float:
    ax, ay, aw, ah = a
    bx, by, bw, bh = b
    ix1 = max(ax, bx)
    iy1 = max(ay, by)
    ix2 = min(ax + aw, bx + bw)
    iy2 = min(ay + ah, by + bh)
    inter = max(0, ix2 - ix1) * max(0, iy2 - iy1)
    if inter <= 0:
        return 0.0
    union = (aw * ah) + (bw * bh) - inter
    return float(inter) / float(max(1, union))


def _ocr_word_regions(image: "Image.Image") -> list[Dict[str, Any]]:
    if not PYTESSERACT_AVAILABLE:
        return []
    try:
        data = pytesseract.image_to_data(image, output_type=pytesseract.Output.DICT, config="--psm 6")
    except Exception:
        return []
    words: list[Dict[str, Any]] = []
    for idx, text in enumerate(data.get("text") or []):
        clean = str(text or "").strip()
        if not clean:
            continue
        try:
            conf = float(data.get("conf", [])[idx])
        except Exception:
            conf = -1.0
        if conf < 0:
            continue
        try:
            x = int(data.get("left", [])[idx])
            y = int(data.get("top", [])[idx])
            w = int(data.get("width", [])[idx])
            h = int(data.get("height", [])[idx])
        except Exception:
            continue
        words.append({"text": clean, "bbox": [x, y, w, h], "confidence": round(max(0.0, min(1.0, conf / 100.0)), 3)})
    return words


def _control_label(control_bbox: list[int], words: list[Dict[str, Any]]) -> tuple[str, list[int], str]:
    if not words:
        return "", [], "ocr_labels_unavailable"
    x, y, w, h = control_bbox
    cy = y + (h / 2.0)
    row_words: list[Dict[str, Any]] = []
    for word in words:
        bx, by, bw, bh = list(word.get("bbox") or [0, 0, 0, 0])
        bcy = by + (bh / 2.0)
        if bx >= x + w and bx <= x + w + 520 and abs(bcy - cy) <= max(h * 0.9, 14):
            row_words.append(word)
    if not row_words:
        for word in words:
            bx, by, bw, bh = list(word.get("bbox") or [0, 0, 0, 0])
            if by >= y and by <= y + max(h * 3, 70) and bx >= x - max(w, 30) and bx <= x + 520:
                row_words.append(word)
    row_words = sorted(row_words, key=lambda item: (int((item.get("bbox") or [0, 0, 0, 0])[1]), int((item.get("bbox") or [0, 0, 0, 0])[0])))[:8]
    if not row_words:
        return "", [], "no_nearby_label"
    label = " ".join(str(item.get("text") or "") for item in row_words).strip()
    xs = [int((item.get("bbox") or [0, 0, 0, 0])[0]) for item in row_words]
    ys = [int((item.get("bbox") or [0, 0, 0, 0])[1]) for item in row_words]
    x2s = [int((item.get("bbox") or [0, 0, 0, 0])[0]) + int((item.get("bbox") or [0, 0, 0, 0])[2]) for item in row_words]
    y2s = [int((item.get("bbox") or [0, 0, 0, 0])[1]) + int((item.get("bbox") or [0, 0, 0, 0])[3]) for item in row_words]
    label_bbox = [min(xs), min(ys), max(x2s) - min(xs), max(y2s) - min(ys)] if xs and ys else []
    return label, label_bbox, "nearest_right_label"


def detect_visual_controls(path: Path | str, ocr_regions: Optional[list[Dict[str, Any]]] = None, max_pages: int = 4) -> Dict[str, Any]:
    source = Path(path)
    warnings: list[str] = []
    if not PIL_AVAILABLE:
        return {"ok": False, "error": "pil_unavailable", "controls": [], "warnings": ["visual_control_detection_requires_pil"]}
    if not CV2_AVAILABLE:
        return {"ok": False, "error": "opencv_unavailable", "controls": [], "warnings": ["visual_control_detection_requires_opencv"]}
    images = _open_rendered_images(source, max_pages=max_pages)
    if not images:
        return {"ok": False, "error": "source_render_failed", "controls": [], "warnings": ["no_rendered_pages"]}

    controls: list[Dict[str, Any]] = []
    for page_idx, image in enumerate(images[: max(1, int(max_pages))], start=1):
        gray = _image_gray_array(image)
        binary = _ink_mask(gray)
        if gray is None or binary is None:
            warnings.append(f"page_{page_idx}_threshold_failed")
            continue
        height, width = gray.shape[:2]
        try:
            contours, _hier = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        except Exception:
            contours = []
        words = list(ocr_regions or []) if ocr_regions else _ocr_word_regions(image)
        page_candidates: list[Dict[str, Any]] = []
        for contour in contours:
            x, y, w, h = [int(v) for v in cv2.boundingRect(contour)]
            if w < 8 or h < 8:
                continue
            side = (w + h) / 2.0
            if side < 20:
                continue
            aspect = float(w) / float(max(1, h))
            if aspect < 0.72 or aspect > 1.38:
                continue
            if side > min(90, max(width, height) * 0.08):
                continue
            if (w * h) > (width * height * 0.012):
                continue
            perimeter = float(cv2.arcLength(contour, True))
            if perimeter <= 0:
                continue
            rectangularity = float(cv2.contourArea(contour)) / float(max(1, w * h))
            if rectangularity < 0.20:
                continue
            inset = max(2, int(round(side * 0.18)))
            if w <= inset * 2 or h <= inset * 2:
                continue
            band = max(2, int(round(side * 0.14)))
            top = binary[y : y + band, x : x + w]
            bottom = binary[y + h - band : y + h, x : x + w]
            left = binary[y : y + h, x : x + band]
            right = binary[y : y + h, x + w - band : x + w]
            edge_ratios = [
                float(cv2.countNonZero(top)) / float(max(1, top.size)),
                float(cv2.countNonZero(bottom)) / float(max(1, bottom.size)),
                float(cv2.countNonZero(left)) / float(max(1, left.size)),
                float(cv2.countNonZero(right)) / float(max(1, right.size)),
            ]
            if min(edge_ratios) < 0.16 or sum(1 for ratio in edge_ratios if ratio >= 0.24) < 3:
                continue
            interior = binary[y + inset : y + h - inset, x + inset : x + w - inset]
            interior_ratio = float(cv2.countNonZero(interior)) / float(max(1, interior.size))
            if interior_ratio >= 0.09:
                state = "checked"
                confidence = min(0.95, 0.62 + interior_ratio * 2.0)
                reason = "interior_ink_above_checked_threshold"
            elif interior_ratio <= 0.028:
                state = "unchecked"
                confidence = max(0.58, min(0.93, 0.90 - interior_ratio * 4.0))
                reason = "clean_interior_below_unchecked_threshold"
            else:
                state = "indeterminate"
                confidence = 0.45
                reason = "interior_ink_ambiguous"
            bbox = [x, y, w, h]
            if any(_bbox_iou(bbox, list(existing.get("bbox") or [])) > 0.45 for existing in page_candidates):
                continue
            label, label_bbox, label_reason = _control_label(bbox, words)
            page_candidates.append(
                {
                    "type": "checkbox",
                    "state": state,
                    "confidence": round(float(confidence), 3),
                    "bbox": bbox,
                    "page": page_idx,
                    "label": label,
                    "label_bbox": label_bbox,
                    "reason": f"{reason};{label_reason}",
                }
            )
        controls.extend(sorted(page_candidates, key=lambda item: (int((item.get("bbox") or [0, 0, 0, 0])[1]), int((item.get("bbox") or [0, 0, 0, 0])[0]))))
    return {"ok": True, "mode": "opencv_contour_heuristic", "controls": controls, "warnings": warnings}


def _feature_page_scores(image: "Image.Image") -> Dict[str, float]:
    gray = _image_gray_array(image)
    binary = _ink_mask(gray)
    if gray is None or binary is None:
        return {"handwriting": 0.0, "chart": 0.0, "figure": 0.0, "ink_density": 0.0}
    height, width = gray.shape[:2]
    page_area = max(1, height * width)
    ink_density = float(cv2.countNonZero(binary)) / float(page_area)
    try:
        contours, _hier = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    except Exception:
        contours = []
    component_boxes: list[tuple[int, int, int, int]] = []
    graphic_area = 0
    long_h = 0
    long_v = 0
    filled_blocks = 0
    elongated_medium = 0
    try:
        edges = cv2.Canny(gray, 50, 150)
        lines = cv2.HoughLinesP(
            edges,
            1,
            np.pi / 180.0,
            threshold=60,
            minLineLength=int(min(width, height) * 0.16),
            maxLineGap=10,
        )
    except Exception:
        lines = None
    if lines is not None:
        for line in lines[:80]:
            coords = np.asarray(line).reshape(-1).tolist()
            if len(coords) < 4:
                continue
            x1, y1, x2, y2 = [int(v) for v in coords[:4]]
            dx = abs(x2 - x1)
            dy = abs(y2 - y1)
            if dx > width * 0.16 and dy <= max(6, height * 0.015):
                long_h += 1
            if dy > height * 0.14 and dx <= max(6, width * 0.015):
                long_v += 1
    for contour in contours:
        x, y, w, h = [int(v) for v in cv2.boundingRect(contour)]
        if w < 2 or h < 2:
            continue
        area = w * h
        component_boxes.append((x, y, w, h))
        if w > width * 0.18 and h <= max(8, height * 0.018):
            long_h += 1
        if h > height * 0.15 and w <= max(8, width * 0.018):
            long_v += 1
        if area > page_area * 0.015:
            graphic_area += area
        if 24 <= w <= width * 0.18 and 8 <= h <= height * 0.08 and (float(w) / float(max(1, h))) >= 2.2:
            elongated_medium += 1
        if h > height * 0.04 and w > width * 0.025 and float(cv2.countNonZero(binary[y : y + h, x : x + w])) / float(max(1, area)) > 0.35:
            filled_blocks += 1
    widths = [w for _x, _y, w, _h in component_boxes if 3 <= w <= width * 0.25]
    heights = [h for _x, _y, _w, h in component_boxes if 3 <= h <= height * 0.15]
    if widths and heights and CV2_AVAILABLE:
        width_var = float(np.std(widths)) / float(max(1.0, np.mean(widths)))
        height_var = float(np.std(heights)) / float(max(1.0, np.mean(heights)))
    else:
        width_var = 0.0
        height_var = 0.0
    medium_components = sum(1 for _x, _y, w, h in component_boxes if 8 <= w <= width * 0.20 and 5 <= h <= height * 0.08)
    handwriting_score = min(
        1.0,
        (ink_density * 4.0)
        + (min(1.0, width_var / 1.4) * 0.25)
        + (min(1.0, height_var / 1.1) * 0.20)
        + (min(1.0, elongated_medium / 20.0) * 0.45),
    )
    if medium_components < 8:
        handwriting_score *= 0.45
    chart_score = min(1.0, (min(1.0, (long_h + long_v) / 4.0) * 0.55) + (min(1.0, filled_blocks / 4.0) * 0.30) + (min(1.0, graphic_area / float(page_area) * 8.0) * 0.15))
    figure_score = min(1.0, (graphic_area / float(page_area)) * 5.0)
    if chart_score > 0.45:
        figure_score = max(figure_score, 0.35)
    return {
        "handwriting": round(handwriting_score, 3),
        "chart": round(chart_score, 3),
        "figure": round(figure_score, 3),
        "ink_density": round(ink_density, 4),
    }


def classify_document_features(
    path: Path | str,
    goal: str = "",
    draft_text: Optional[str] = None,
    draft_quality: Optional[Dict[str, Any]] = None,
    max_pages: int = 4,
) -> Dict[str, Any]:
    source = Path(path)
    goal_norm = " ".join(str(goal or "").lower().split())
    warnings: list[str] = []
    if not PIL_AVAILABLE:
        return {
            "ok": False,
            "mode": "lightweight_heuristic",
            "error": "pil_unavailable",
            "has_handwriting": any(term in goal_norm for term in ("handwriting", "handwritten", "cursive")),
            "has_chart": any(term in goal_norm for term in ("chart", "graph", "plot")),
            "has_figure": any(term in goal_norm for term in ("figure", "diagram", "image")),
            "forms_or_checkboxes": any(term in goal_norm for term in ("checkbox", "form", "radio")),
            "layout_class": "unknown",
            "confidence_reason": "pil_unavailable_goal_hints_only",
            "warnings": ["feature_detection_requires_pil"],
        }
    images = _open_rendered_images(source, max_pages=max_pages)
    if not images:
        return {
            "ok": False,
            "mode": "lightweight_heuristic",
            "error": "source_render_failed",
            "has_handwriting": False,
            "has_chart": False,
            "has_figure": False,
            "forms_or_checkboxes": False,
            "layout_class": "unknown",
            "confidence_reason": "source_render_failed",
            "warnings": ["no_rendered_pages"],
        }
    page_scores = [_feature_page_scores(image) for image in images[: max(1, int(max_pages))]]
    handwriting_score = max([float(row.get("handwriting") or 0.0) for row in page_scores] or [0.0])
    chart_score = max([float(row.get("chart") or 0.0) for row in page_scores] or [0.0])
    figure_score = max([float(row.get("figure") or 0.0) for row in page_scores] or [0.0])
    text = str(draft_text or "")
    quality = dict(draft_quality or {})
    try:
        draft_conf = float(quality.get("confidence")) if quality.get("confidence") is not None else None
    except Exception:
        draft_conf = None
    if draft_conf is not None and draft_conf < 0.45 and len(text.strip()) < 250:
        handwriting_score = max(handwriting_score, 0.58)
    controls_result = detect_visual_controls(source, max_pages=max_pages)
    controls = list(controls_result.get("controls") or [])
    if not controls_result.get("ok"):
        warnings.extend(list(controls_result.get("warnings") or []))
    goal_handwriting = any(term in goal_norm for term in ("handwriting", "handwritten", "cursive", "scribble"))
    goal_chart = any(term in goal_norm for term in ("chart", "graph", "plot", "bar chart", "line chart"))
    goal_figure = any(term in goal_norm for term in ("figure", "diagram", "illustration", "infographic"))
    goal_form = any(term in goal_norm for term in ("checkbox", "check box", "radio button", "form control"))
    has_handwriting = bool(goal_handwriting or handwriting_score >= 0.56)
    has_chart = bool(goal_chart or (chart_score >= 0.50 and figure_score >= 0.45))
    has_figure = bool(goal_figure or (figure_score >= 0.45 and not has_chart))
    forms_or_checkboxes = bool(goal_form or len(controls) > 0)
    table_ratio = 0.0
    if text:
        rows = [line for line in text.splitlines() if line.strip()]
        table_rows = sum(1 for line in rows if line.count("|") >= 2 or line.count("\t") >= 2)
        table_ratio = round(float(table_rows) / float(max(1, len(rows))), 3)
    text_density = round(min(1.0, len(text.strip()) / 2500.0), 3)
    digits = sum(1 for ch in text if ch.isdigit())
    number_density = round(float(digits) / float(max(1, len(text))), 3)
    if forms_or_checkboxes:
        layout_class = "forms_or_checkboxes"
    elif has_chart or has_figure:
        layout_class = "chart_or_figure"
    elif has_handwriting:
        layout_class = "handwritten"
    elif table_ratio >= 0.35:
        layout_class = "table_dense"
    elif source.suffix.lower() == ".pdf":
        layout_class = "digital_pdf"
    else:
        layout_class = "photo"
    confidence_reason = (
        f"heuristic_scores handwriting={handwriting_score:.2f} chart={chart_score:.2f} "
        f"figure={figure_score:.2f} controls={len(controls)}"
    )
    return {
        "ok": True,
        "mode": "lightweight_heuristic",
        "has_handwriting": has_handwriting,
        "has_chart": has_chart,
        "has_figure": has_figure,
        "forms_or_checkboxes": forms_or_checkboxes,
        "layout_class": layout_class,
        "table_ratio": table_ratio,
        "text_density": text_density,
        "number_density": number_density,
        "confidence_reason": confidence_reason,
        "scores": {
            "handwriting": round(handwriting_score, 3),
            "chart": round(chart_score, 3),
            "figure": round(figure_score, 3),
            "ink_density": max([float(row.get("ink_density") or 0.0) for row in page_scores] or [0.0]),
        },
        "warnings": warnings,
        "visual_control_count": len(controls),
        "visual_controls": controls,
        "layout_regions": [],
    }


def _ensure_transformers_loaded() -> Dict[str, Any]:
    global torch
    global AutoImageProcessor
    global AutoModelForSequenceClassification
    global AutoTokenizer
    global GotOcr2ForConditionalGeneration
    global GotOcr2Processor
    global TableTransformerForObjectDetection
    global pipeline
    if not TRANSFORMERS_AVAILABLE:
        return {"ok": False, "error": "transformers_unavailable"}
    if torch is not None and GotOcr2Processor is not None and AutoTokenizer is not None and TableTransformerForObjectDetection is not None:
        return {"ok": True}
    try:
        torch = importlib.import_module("torch")
        transformers = importlib.import_module("transformers")
        AutoImageProcessor = getattr(transformers, "AutoImageProcessor")
        AutoModelForSequenceClassification = getattr(transformers, "AutoModelForSequenceClassification")
        AutoTokenizer = getattr(transformers, "AutoTokenizer")
        GotOcr2ForConditionalGeneration = getattr(transformers, "GotOcr2ForConditionalGeneration")
        GotOcr2Processor = getattr(transformers, "GotOcr2Processor")
        TableTransformerForObjectDetection = getattr(transformers, "TableTransformerForObjectDetection")
        pipeline = getattr(transformers, "pipeline")
        return {"ok": True}
    except Exception as exc:
        return {"ok": False, "error": f"transformers_import_failed:{type(exc).__name__}:{exc}"}


def load_got_runtime() -> Dict[str, Any]:
    model_id = _got_model_id()
    if not _got_backend_enabled():
        return {"ok": False, "error": "got_backend_disabled"}
    if not TRANSFORMERS_AVAILABLE:
        return {"ok": False, "error": "got_transformers_unavailable"}
    runtime_import = _ensure_transformers_loaded()
    if not runtime_import.get("ok"):
        return {"ok": False, "error": str(runtime_import.get("error") or "got_transformers_unavailable")}
    if not PIL_AVAILABLE:
        return {"ok": False, "error": "pil_unavailable"}
    with _GOT_RUNTIME_LOCK:
        if (
            _GOT_RUNTIME.get("model") is not None
            and _GOT_RUNTIME.get("processor") is not None
            and str(_GOT_RUNTIME.get("loaded_model_id") or "") == model_id
        ):
            return {
                "ok": True,
                "model": _GOT_RUNTIME.get("model"),
                "processor": _GOT_RUNTIME.get("processor"),
                "model_id": model_id,
                "backend": "transformers_got_ocr2",
            }
        try:
            kwargs: Dict[str, Any] = {"torch_dtype": _got_dtype()}
            device_pref = _got_device_pref()
            if device_pref == "auto":
                kwargs["device_map"] = "auto"
            elif device_pref == "cpu":
                kwargs["device_map"] = {"": "cpu"}
            processor = GotOcr2Processor.from_pretrained(model_id)
            model = GotOcr2ForConditionalGeneration.from_pretrained(model_id, **kwargs)
            if device_pref in {"cpu", "cuda"} and "device_map" not in kwargs:
                model = model.to(device_pref)
            model.eval()
            _GOT_RUNTIME["processor"] = processor
            _GOT_RUNTIME["model"] = model
            _GOT_RUNTIME["loaded_model_id"] = model_id
            _GOT_RUNTIME["last_error"] = ""
            _GOT_RUNTIME["backend"] = "transformers_got_ocr2"
            return {"ok": True, "model": model, "processor": processor, "model_id": model_id, "backend": "transformers_got_ocr2"}
        except Exception as exc:
            _GOT_RUNTIME["processor"] = None
            _GOT_RUNTIME["model"] = None
            _GOT_RUNTIME["loaded_model_id"] = model_id
            _GOT_RUNTIME["last_error"] = f"got_model_load_failed:{type(exc).__name__}:{exc}"
            _GOT_RUNTIME["backend"] = "load_failed"
            return {"ok": False, "error": _GOT_RUNTIME["last_error"]}


def got_extract_transformers(
    path: Path,
    goal: str,
    max_chars: int,
    max_pages: int,
    *,
    normalize_markdown: Optional[Callable[..., str]] = None,
    normalize_text: Optional[Callable[..., str]] = None,
) -> Dict[str, Any]:
    runtime = load_got_runtime()
    if not runtime.get("ok"):
        return {"ok": False, "error": str(runtime.get("error") or "got_runtime_unavailable")}
    images = _open_rendered_images(path, max_pages=max_pages)
    if not images:
        return {"ok": False, "error": "got_ocr2_no_renderable_pages"}
    model = runtime.get("model")
    processor = runtime.get("processor")
    model_id = str(runtime.get("model_id") or _got_model_id())
    outputs: list[str] = []
    confidences: list[float] = []
    try:
        for image in images:
            model_inputs = processor(image, return_tensors="pt")
            runtime_device = "cpu"
            try:
                runtime_device = str(next(model.parameters()).device)
            except Exception:
                runtime_device = "cuda" if torch.cuda.is_available() else "cpu"
            for key, value in list(model_inputs.items()):
                if hasattr(value, "to"):
                    model_inputs[key] = value.to(runtime_device)
            with torch.inference_mode():
                generated = model.generate(
                    **model_inputs,
                    do_sample=False,
                    tokenizer=processor.tokenizer,
                    stop_strings="<|im_end|>",
                    max_new_tokens=max(512, min(max_chars, 4096)),
                    output_scores=True,
                    return_dict_in_generate=True,
                )
            sequences = generated.sequences
            prompt_len = int(model_inputs["input_ids"].shape[-1]) if "input_ids" in model_inputs else 0
            if prompt_len > 0 and hasattr(sequences, "__getitem__"):
                sequences = sequences[:, prompt_len:]
            decoded = processor.batch_decode(sequences, skip_special_tokens=True)
            text = str(decoded[0] if decoded else "").strip()
            if text:
                outputs.append(text)
            for score_row in list(generated.scores or []):
                try:
                    probs = torch.softmax(score_row.float(), dim=-1)
                    max_prob = probs.max(dim=-1).values
                    confidences.append(float(max_prob.mean().item()))
                except Exception:
                    continue
        markdown = _call_markdown_normalizer(normalize_markdown, "\n\n".join(outputs), max_chars=max_chars)
        text = _call_text_normalizer(normalize_text, markdown, max_chars=max_chars)
        if not text:
            return {"ok": False, "error": "got_ocr2_empty_output"}
        return {
            "ok": True,
            "engine": "gb10_got_ocr2",
            "model": model_id,
            "text": text,
            "markdown": markdown,
            "confidence": round(sum(confidences) / len(confidences), 3) if confidences else None,
            "pages": len(images),
            "route": "local_got_transformers",
            "reason": "got_ocr2_transformers_local",
            "goal": goal,
        }
    except Exception as exc:
        return {"ok": False, "error": f"got_ocr2_infer_failed:{type(exc).__name__}:{exc}"}


def got_extract_texts(
    path: Path,
    goal: str,
    max_chars: int,
    max_pages: int,
    *,
    normalize_markdown: Optional[Callable[..., str]] = None,
    normalize_text: Optional[Callable[..., str]] = None,
) -> Dict[str, Any]:
    return got_extract_transformers(
        path,
        goal,
        max_chars,
        max_pages,
        normalize_markdown=normalize_markdown,
        normalize_text=normalize_text,
    )


def load_finbert_runtime() -> Dict[str, Any]:
    if not TRANSFORMERS_AVAILABLE:
        return {"ok": False, "error": "phase2_transformers_unavailable"}
    runtime_import = _ensure_transformers_loaded()
    if not runtime_import.get("ok"):
        return {"ok": False, "error": str(runtime_import.get("error") or "phase2_transformers_unavailable")}
    model_id = _finbert_model_id()
    with _FINBERT_RUNTIME_LOCK:
        if _FINBERT_RUNTIME.get("pipeline") is not None and str(_FINBERT_RUNTIME.get("loaded_model_id") or "") == model_id:
            return {"ok": True, "pipeline": _FINBERT_RUNTIME.get("pipeline"), "model_id": model_id}
        try:
            tokenizer = AutoTokenizer.from_pretrained(model_id)
            model = AutoModelForSequenceClassification.from_pretrained(model_id, weights_only=False)
            clf = pipeline(
                "text-classification",
                model=model,
                tokenizer=tokenizer,
                device=_transformers_device_index(_finbert_device()),
            )
            _FINBERT_RUNTIME["pipeline"] = clf
            _FINBERT_RUNTIME["loaded_model_id"] = model_id
            _FINBERT_RUNTIME["last_error"] = ""
            _FINBERT_RUNTIME["backend"] = "transformers_finbert"
            return {"ok": True, "pipeline": clf, "model_id": model_id}
        except Exception as exc:
            _FINBERT_RUNTIME["pipeline"] = None
            _FINBERT_RUNTIME["loaded_model_id"] = model_id
            _FINBERT_RUNTIME["last_error"] = f"finbert_load_failed:{type(exc).__name__}:{exc}"
            _FINBERT_RUNTIME["backend"] = "load_failed"
            return {"ok": False, "error": _FINBERT_RUNTIME["last_error"]}


def finbert_eval(text: str) -> Dict[str, Any]:
    runtime = load_finbert_runtime()
    if not runtime.get("ok"):
        return {"ok": False, "error": str(runtime.get("error") or "finbert_runtime_unavailable")}
    raw = str(text or "").strip()
    if not raw:
        return {"ok": False, "error": "text_required"}
    classifier = runtime.get("pipeline")
    try:
        result = classifier(raw[:4000], truncation=True, max_length=512)
        item = dict((result or [{}])[0] or {})
        return {
            "ok": True,
            "mode": "finbert_service",
            "model": str(runtime.get("model_id") or ""),
            "label": str(item.get("label") or ""),
            "score": float(item.get("score") or 0.0),
        }
    except Exception as exc:
        _FINBERT_RUNTIME["last_error"] = f"finbert_infer_failed:{type(exc).__name__}:{exc}"
        return {"ok": False, "error": _FINBERT_RUNTIME["last_error"]}


def _utc_iso() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat().replace("+00:00", "Z")


def _fingerprint_store_path(fingerprint_path: Optional[Path] = None) -> Path:
    if fingerprint_path is not None:
        return Path(fingerprint_path)
    raw = str(os.getenv("OCR97_OCR_FINGERPRINT_PATH", "")).strip()
    return Path(raw) if raw else (_PATHS.state_dir / "ocr_fingerprints.json")


def _load_fingerprint_store(*, fingerprint_path: Optional[Path] = None) -> Dict[str, Any]:
    path = _fingerprint_store_path(fingerprint_path)
    try:
        if path.exists():
            return dict(json.loads(path.read_text(encoding="utf-8")))
    except Exception:
        return {}
    return {}


def _save_fingerprint_store(store: Dict[str, Any], *, fingerprint_path: Optional[Path] = None) -> None:
    path = _fingerprint_store_path(fingerprint_path)
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        tmp = path.with_suffix(path.suffix + ".tmp")
        tmp.write_text(json.dumps(store, indent=2, sort_keys=True), encoding="utf-8")
        tmp.replace(path)
    except Exception:
        return


def semantic_diff_check(source_path: str, text: str, *, fingerprint_path: Optional[Path] = None) -> Dict[str, Any]:
    tokens = re.findall(r"-?\d+(?:\.\d+)?", str(text or ""))
    token_hash = hashlib.sha256("|".join(tokens).encode("utf-8", errors="ignore")).hexdigest() if tokens else ""
    content_hash = hashlib.sha256(str(text or "").encode("utf-8", errors="ignore")).hexdigest()
    key = str(source_path or "unknown").strip() or "unknown"
    store = _load_fingerprint_store(fingerprint_path=fingerprint_path)
    prior = dict((store.get("documents") or {}).get(key) or {})
    prior_count = int(prior.get("numeric_count") or 0)
    cur_count = len(tokens)
    denom = max(1, prior_count)
    change_pct = abs(cur_count - prior_count) / float(denom) if prior else 0.0
    changed = bool(prior) and (
        str(prior.get("token_hash") or "") != token_hash or str(prior.get("content_hash") or "") != content_hash
    )
    if not prior:
        risk_band = "none"
    elif change_pct >= 0.35:
        risk_band = "high"
    elif change_pct >= 0.15:
        risk_band = "medium"
    else:
        risk_band = "low"
    now = _utc_iso()
    docs = dict(store.get("documents") or {})
    docs[key] = {
        "updated_at": now,
        "numeric_count": cur_count,
        "token_hash": token_hash,
        "content_hash": content_hash,
    }
    store["documents"] = docs
    store["updated_at"] = now
    _save_fingerprint_store(store, fingerprint_path=fingerprint_path)
    return {
        "first_seen": not bool(prior),
        "changed": changed,
        "change_pct": round(change_pct, 3),
        "risk_band": risk_band,
        "prior_ts": str(prior.get("updated_at") or ""),
        "numeric_count": cur_count,
    }


def load_tableformer_runtime() -> Dict[str, Any]:
    if not TRANSFORMERS_AVAILABLE or not PIL_AVAILABLE:
        return {"ok": False, "error": "tableformer_runtime_unavailable"}
    runtime_import = _ensure_transformers_loaded()
    if not runtime_import.get("ok"):
        return {"ok": False, "error": str(runtime_import.get("error") or "tableformer_runtime_unavailable")}
    model_id = _tableformer_model_id()
    with _TABLE_RUNTIME_LOCK:
        if (
            _TABLE_RUNTIME.get("processor") is not None
            and _TABLE_RUNTIME.get("model") is not None
            and str(_TABLE_RUNTIME.get("loaded_model_id") or "") == model_id
        ):
            return {
                "ok": True,
                "processor": _TABLE_RUNTIME.get("processor"),
                "model": _TABLE_RUNTIME.get("model"),
                "model_id": model_id,
            }
        try:
            processor = AutoImageProcessor.from_pretrained(model_id)
            model = TableTransformerForObjectDetection.from_pretrained(model_id)
            device_name = _tableformer_device()
            model.to(device_name)
            model.eval()
            _TABLE_RUNTIME["processor"] = processor
            _TABLE_RUNTIME["model"] = model
            _TABLE_RUNTIME["loaded_model_id"] = model_id
            _TABLE_RUNTIME["last_error"] = ""
            _TABLE_RUNTIME["backend"] = f"transformers_tableformer_{device_name}"
            return {"ok": True, "processor": processor, "model": model, "model_id": model_id}
        except Exception as exc:
            _TABLE_RUNTIME["processor"] = None
            _TABLE_RUNTIME["model"] = None
            _TABLE_RUNTIME["loaded_model_id"] = model_id
            _TABLE_RUNTIME["last_error"] = f"tableformer_load_failed:{type(exc).__name__}:{exc}"
            _TABLE_RUNTIME["backend"] = "load_failed"
            return {"ok": False, "error": _TABLE_RUNTIME["last_error"]}


def _render_table_source_image(source_path: Path) -> Optional["Image.Image"]:
    try:
        if str(source_path.suffix or "").lower() == ".pdf":
            if not PYMUPDF_AVAILABLE:
                return None
            doc = fitz.open(str(source_path))
            try:
                page = doc.load_page(0)
                pix = page.get_pixmap(matrix=fitz.Matrix(2.0, 2.0), alpha=False)
                return Image.frombytes("RGB", [pix.width, pix.height], pix.samples)
            finally:
                doc.close()
        with Image.open(source_path) as image:
            return image.convert("RGB")
    except Exception:
        return None


def _label_name(model: Any, label_id: Any) -> str:
    try:
        return str(model.config.id2label[int(label_id)]).strip().lower()
    except Exception:
        return str(label_id).strip().lower()


def _box_center(box: Dict[str, float]) -> tuple[float, float]:
    return ((float(box["x0"]) + float(box["x1"])) / 2.0, (float(box["y0"]) + float(box["y1"])) / 2.0)


def _within_box(inner: Dict[str, float], outer: Dict[str, float]) -> bool:
    cx, cy = _box_center(inner)
    return outer["x0"] <= cx <= outer["x1"] and outer["y0"] <= cy <= outer["y1"]


def _projection_segments(image: "Image.Image", axis: str, max_segments: int = 12) -> list[Dict[str, int]]:
    gray = image.convert("L")
    width, height = gray.size
    arr = list(gray.getdata())
    values: list[float] = []
    if axis == "horizontal":
        for y in range(height):
            row = arr[y * width : (y + 1) * width]
            values.append(sum(255 - px for px in row) / max(1, width))
    else:
        for x in range(width):
            values.append(sum(255 - arr[(y * width) + x] for y in range(height)) / max(1, height))
    threshold = max(6.0, (sum(values) / max(1, len(values))) * 0.75)
    segments: list[Dict[str, int]] = []
    start = None
    for idx, value in enumerate(values):
        if value >= threshold and start is None:
            start = idx
        elif value < threshold and start is not None:
            if idx - start >= 8:
                segments.append({"start": start, "end": idx})
            start = None
    if start is not None and len(values) - start >= 8:
        segments.append({"start": start, "end": len(values)})
    return segments[:max_segments]


def _extract_cell_text(cell_image: "Image.Image", normalize_text: Optional[Callable[..., str]] = None) -> str:
    if not PYTESSERACT_AVAILABLE:
        return ""
    try:
        text = pytesseract.image_to_string(cell_image, config="--psm 7")
    except Exception:
        return ""
    return _call_text_normalizer(normalize_text, text, max_chars=200)


def _markdown_table_rows(text: str, max_rows: int = 32) -> list[list[str]]:
    rows: list[list[str]] = []
    for raw_line in str(text or "").splitlines():
        line = raw_line.strip()
        if not line or "|" not in line or line.count("|") < 2:
            continue
        parts = [part.strip() for part in line.split("|")]
        cells = [cell for cell in parts if cell]
        if not cells:
            continue
        if all(set(cell) <= {"-", ":"} for cell in cells):
            continue
        rows.append(cells)
        if len(rows) >= max_rows:
            break
    return rows


def _cell_density(cells: list[str]) -> int:
    return sum(1 for cell in list(cells or []) if str(cell or "").strip())


def _prefer_fallback_cell(primary: str, fallback: str) -> str:
    base = str(primary or "").strip()
    candidate = str(fallback or "").strip()
    if not candidate:
        return base
    if not base:
        return candidate
    if any(ch in base for ch in "|`[]") and candidate:
        return candidate
    if len(base) <= 2 and len(candidate) >= 4:
        return candidate
    if len(candidate) >= len(base) + 4 and base.lower() in candidate.lower():
        return candidate
    base_clean = "".join(ch for ch in base if ch.isalnum()).lower()
    candidate_clean = "".join(ch for ch in candidate if ch.isalnum()).lower()
    if base_clean and candidate_clean.startswith(base_clean) and len(candidate_clean) > len(base_clean):
        return candidate
    if len(candidate) > len(base) * 2 and any(ch.isdigit() for ch in candidate):
        return candidate
    if sum(1 for ch in base if ch.isdigit()) >= 1 and candidate.isdigit() and not base.isdigit():
        return candidate
    return base


def _synthesized_axis_boxes(table_box: Dict[str, float], total: int, axis: str) -> list[Dict[str, float]]:
    boxes: list[Dict[str, float]] = []
    if total <= 0:
        return boxes
    if axis == "row":
        step = max(1.0, (float(table_box["y1"]) - float(table_box["y0"])) / float(total))
        for idx in range(total):
            y0 = float(table_box["y0"]) + (step * idx)
            y1 = float(table_box["y1"]) if idx == total - 1 else float(table_box["y0"]) + (step * (idx + 1))
            boxes.append({"x0": float(table_box["x0"]), "y0": y0, "x1": float(table_box["x1"]), "y1": y1})
    else:
        step = max(1.0, (float(table_box["x1"]) - float(table_box["x0"])) / float(total))
        for idx in range(total):
            x0 = float(table_box["x0"]) + (step * idx)
            x1 = float(table_box["x1"]) if idx == total - 1 else float(table_box["x0"]) + (step * (idx + 1))
            boxes.append({"x0": x0, "y0": float(table_box["y0"]), "x1": x1, "y1": float(table_box["y1"])})
    return boxes


def _merge_table_rows(
    detected_rows: list[Dict[str, Any]],
    text_rows: list[list[str]],
    row_boxes: list[Dict[str, float]],
    col_boxes: list[Dict[str, float]],
    table_box: Dict[str, float],
) -> list[Dict[str, Any]]:
    merged: list[Dict[str, Any]] = []
    target_rows = max(len(detected_rows), len(text_rows), len(row_boxes), 1)
    target_cols = max(
        len(col_boxes),
        max((len(row.get("cells") or []) for row in detected_rows), default=0),
        max((len(row) for row in text_rows), default=0),
        1,
    )
    if len(row_boxes) < target_rows:
        row_boxes = _synthesized_axis_boxes(table_box, target_rows, "row")
    if len(col_boxes) < target_cols:
        col_boxes = _synthesized_axis_boxes(table_box, target_cols, "col")
    for idx in range(target_rows):
        detected = dict(detected_rows[idx]) if idx < len(detected_rows) else {}
        detected_cells = [str(cell or "") for cell in list(detected.get("cells") or [])]
        fallback_cells = [str(cell or "") for cell in (text_rows[idx] if idx < len(text_rows) else [])]
        cells: list[str] = []
        for col_idx in range(target_cols):
            primary = detected_cells[col_idx] if col_idx < len(detected_cells) else ""
            fallback = fallback_cells[col_idx] if col_idx < len(fallback_cells) else ""
            cells.append(_prefer_fallback_cell(primary, fallback))
        if not any(cell.strip() for cell in cells):
            continue
        bbox = row_boxes[idx] if idx < len(row_boxes) else detected.get("bbox") or table_box
        merged.append(
            {
                "bbox": {k: int(float(v)) for k, v in dict(bbox).items()},
                "cells": cells,
                "source": {
                    "ocr_non_empty_cells": _cell_density(detected_cells),
                    "text_non_empty_cells": _cell_density(fallback_cells),
                },
            }
        )
    return merged


def tableformer_reconstruct(
    source_path: str,
    text: str = "",
    *,
    normalize_text: Optional[Callable[..., str]] = None,
) -> Dict[str, Any]:
    path = Path(str(source_path or "").strip())
    if not path.exists():
        return {"ok": False, "error": "source_path_required"}
    runtime = load_tableformer_runtime()
    if not runtime.get("ok"):
        return {"ok": False, "error": str(runtime.get("error") or "tableformer_runtime_unavailable")}
    image = _render_table_source_image(path)
    if image is None:
        return {"ok": False, "error": "source_image_unavailable"}
    processor = runtime.get("processor")
    model = runtime.get("model")
    try:
        inputs = processor(images=image, return_tensors="pt", size={"shortest_edge": 800, "longest_edge": 1333})
        device_name = _tableformer_device()
        model_dtype = next(model.parameters()).dtype if hasattr(model, "parameters") else None
        for key, value in list(inputs.items()):
            if hasattr(value, "to"):
                if model_dtype is not None and hasattr(value, "dtype") and getattr(value, "dtype", None) is not None and str(value.dtype).startswith("torch.float"):
                    inputs[key] = value.to(device=device_name, dtype=model_dtype)
                else:
                    inputs[key] = value.to(device_name)
        with torch.inference_mode():
            outputs = model(**inputs)
        target_sizes = torch.tensor([[image.size[1], image.size[0]]], device="cpu")
        results = processor.post_process_object_detection(outputs, threshold=0.7, target_sizes=target_sizes)[0]
    except Exception as exc:
        _TABLE_RUNTIME["last_error"] = f"tableformer_infer_failed:{type(exc).__name__}:{exc}"
        return {"ok": False, "error": _TABLE_RUNTIME["last_error"]}

    detections: list[Dict[str, Any]] = []
    score_list = results.get("scores").tolist() if results.get("scores") is not None and hasattr(results.get("scores"), "tolist") else list(results.get("scores") or [])
    label_list = results.get("labels").tolist() if results.get("labels") is not None and hasattr(results.get("labels"), "tolist") else list(results.get("labels") or [])
    box_list_all = results.get("boxes").tolist() if results.get("boxes") is not None and hasattr(results.get("boxes"), "tolist") else list(results.get("boxes") or [])
    for score, label_id, box_values in zip(score_list, label_list, box_list_all):
        box_values = [float(value) for value in list(box_values)]
        detections.append(
            {
                "label": _label_name(model, label_id),
                "score": float(score),
                "box": {"x0": box_values[0], "y0": box_values[1], "x1": box_values[2], "y1": box_values[3]},
            }
        )

    table_candidates = [item for item in detections if item["label"] == "table"]
    table_box = max(table_candidates, key=lambda row: float(row.get("score") or 0.0))["box"] if table_candidates else {"x0": 0.0, "y0": 0.0, "x1": float(image.size[0]), "y1": float(image.size[1])}
    row_boxes = [item["box"] for item in detections if "row" in item["label"] and _within_box(item["box"], table_box)]
    col_boxes = [item["box"] for item in detections if "column" in item["label"] and _within_box(item["box"], table_box)]
    if not row_boxes or not col_boxes:
        table_crop = image.crop((int(table_box["x0"]), int(table_box["y0"]), int(table_box["x1"]), int(table_box["y1"])))
        if not row_boxes:
            row_boxes = [
                {"x0": table_box["x0"], "y0": table_box["y0"] + seg["start"], "x1": table_box["x1"], "y1": table_box["y0"] + seg["end"]}
                for seg in _projection_segments(table_crop, "horizontal")
            ]
        if not col_boxes:
            col_boxes = [
                {"x0": table_box["x0"] + seg["start"], "y0": table_box["y0"], "x1": table_box["x0"] + seg["end"], "y1": table_box["y1"]}
                for seg in _projection_segments(table_crop, "vertical")
            ]

    row_boxes = sorted(row_boxes, key=lambda row: (row["y0"], row["x0"]))
    col_boxes = sorted(col_boxes, key=lambda col: (col["x0"], col["y0"]))
    detected_rows: list[Dict[str, Any]] = []
    for row_box in row_boxes[:20]:
        cells: list[str] = []
        for col_box in col_boxes[:12]:
            x0 = max(int(row_box["x0"]), int(col_box["x0"]), int(table_box["x0"]))
            y0 = max(int(row_box["y0"]), int(col_box["y0"]), int(table_box["y0"]))
            x1 = min(int(row_box["x1"]), int(col_box["x1"]), int(table_box["x1"]))
            y1 = min(int(row_box["y1"]), int(col_box["y1"]), int(table_box["y1"]))
            if x1 - x0 < 8 or y1 - y0 < 8:
                cells.append("")
                continue
            cells.append(_extract_cell_text(image.crop((x0, y0, x1, y1)), normalize_text=normalize_text))
        if any(cell.strip() for cell in cells):
            detected_rows.append({"bbox": {"x0": int(row_box["x0"]), "y0": int(row_box["y0"]), "x1": int(row_box["x1"]), "y1": int(row_box["y1"])}, "cells": cells})

    text_rows = _markdown_table_rows(text)
    structured_rows = _merge_table_rows(detected_rows, text_rows, row_boxes[:20], col_boxes[:12], table_box)
    if not structured_rows:
        return {"ok": False, "error": "tableformer_no_structured_rows", "mode": "tableformer", "detections": detections[:50]}
    non_empty_ocr_cells = sum(_cell_density(list(row.get("cells") or [])) for row in detected_rows)
    non_empty_final_cells = sum(_cell_density(list(row.get("cells") or [])) for row in structured_rows)
    return {
        "ok": True,
        "mode": "tableformer",
        "model": str(runtime.get("model_id") or ""),
        "tables": [{"id": 0, "bbox": {k: int(v) for k, v in table_box.items()}, "rows": structured_rows}],
        "detections": detections[:50],
        "text_rows_detected": len(text_rows),
        "ocr_rows_detected": len(detected_rows),
        "ocr_non_empty_cells": int(non_empty_ocr_cells),
        "final_non_empty_cells": int(non_empty_final_cells),
        "source_path": str(path),
    }

