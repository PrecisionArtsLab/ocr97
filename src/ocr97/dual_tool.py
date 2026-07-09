from __future__ import annotations

import base64
import hashlib
import json
import os
import re
import tempfile
import threading
import time
import uuid
from collections import Counter
from datetime import datetime, timezone
from io import BytesIO
from pathlib import Path
from typing import Any, Dict, Optional, Tuple

import requests

from .engine_registry import (
    dedupe_engine_names,
    engine_is_optional,
    normalize_engine_name,
    select_engine_chain,
)
from .mcp import register
from . import local_inference as ocr_local_inference
from .legacy_env import apply_legacy_env_aliases
from .paths import ensure_paths
from .profiles import gb10_default_enabled, gb10_gateway_default_enabled, local_production_enabled
from .receipt_fields import append_receipt_fields, receipt_fields_from_candidates

apply_legacy_env_aliases()
_PATHS = ensure_paths()

try:
    import pytesseract
    from PIL import Image, ImageEnhance, ImageFilter

    TESS_AVAILABLE = True
    PIL_AVAILABLE = True
except Exception:
    TESS_AVAILABLE = False
    PIL_AVAILABLE = False

try:
    from rapidocr_onnxruntime import RapidOCR

    RAPID_AVAILABLE = True
except Exception:
    RAPID_AVAILABLE = False

try:
    import fitz  # PyMuPDF

    PDF_AVAILABLE = True
except Exception:
    PDF_AVAILABLE = False

try:
    import cv2  # type: ignore
    import numpy as np  # type: ignore

    CV2_AVAILABLE = True
except Exception:
    CV2_AVAILABLE = False

try:
    from surya.foundation import FoundationPredictor
    from surya.layout import LayoutPredictor

    SURYA_AVAILABLE = True
except Exception:
    SURYA_AVAILABLE = False

_RAPID_OCR = None
_SURYA_LAYOUT_PREDICTOR = None
_SURYA_LAYOUT_ERROR = ""
_SURYA_LOCK = threading.Lock()
_ENGINE_WARM_STATE: Dict[str, bool] = {"rapidocr": False}
_TOKEN_SPLIT_RE = re.compile(r"[^a-z0-9]+")
_ANCHOR_STOPWORDS = {
    "pdf",
    "doc",
    "document",
    "scan",
    "image",
    "final",
    "copy",
    "draft",
    "report",
    "statement",
}

DEFAULT_GB10_OCR_ENABLED = gb10_default_enabled()
DEFAULT_GB10_OCR_TIMEOUT_SEC = int(os.getenv("OCR97_GB10_OCR_TIMEOUT_SEC", "150"))
DEFAULT_GB10_PADDLEOCR_VL_URL = str(os.getenv("OCR97_GB10_PADDLEOCR_VL_URL", "")).strip()
DEFAULT_GB10_GOT_OCR_URL = str(os.getenv("OCR97_GB10_GOT_OCR_URL", "")).strip()
DEFAULT_GB10_QWEN_OCR_MODEL = str(
    os.getenv("OCR97_GB10_QWEN_OCR_MODEL", os.getenv("VISION_QWEN25VL_MODEL", os.getenv("VISION_QWEN2_5VL_MODEL", "qwen2.5vl:7b")))
).strip()
DEFAULT_GB10_QWEN_OCR_FALLBACK_MODEL = str(
    os.getenv("OCR97_GB10_QWEN_OCR_FALLBACK_MODEL", os.getenv("VISION_QWEN3VL_MODEL", "qwen3-vl:32b"))
).strip()
DEFAULT_GB10_QWEN_OLLAMA_URL = str(
    os.getenv("OCR97_GB10_QWEN_OLLAMA_URL", os.getenv("VISION_OLLAMA_URL", os.getenv("OLLAMA_URL", "http://127.0.0.1:11434")))
).strip()
DEFAULT_GB10_QWEN_CLEANUP = os.getenv("OCR97_GB10_OCR_USE_QWEN_CLEANUP", "1").lower() in {"1", "true", "yes"}
DEFAULT_GB10_OCR_GATEWAY_URL = str(os.getenv("OCR97_GB10_OCR_GATEWAY_URL", "")).strip()
DEFAULT_GB10_OCR_USE_GATEWAY = gb10_gateway_default_enabled()
DEFAULT_GB10_QWEN_PRIMARY = os.getenv("OCR97_OCR_QWEN_PRIMARY", "0").lower() in {"1", "true", "yes"}
DEFAULT_OCR_COMPAT_ENABLED = os.getenv("OCR97_OCR_COMPAT_ENABLED", "1" if local_production_enabled() else "0").lower() in {"1", "true", "yes"}
DEFAULT_OCR_COMPAT_OLLAMA_URL = str(os.getenv("OCR97_OCR_COMPAT_OLLAMA_URL", "")).strip()
DEFAULT_OCR_COMPAT_MODEL = str(os.getenv("OCR97_OCR_COMPAT_MODEL", DEFAULT_GB10_QWEN_OCR_MODEL)).strip()
DEFAULT_OCR_ROUTE_MODE = str(os.getenv("OCR97_OCR_ROUTE_MODE", "quality_first")).strip().lower() or "quality_first"
DEFAULT_OCR_PHASE2_ENABLED = os.getenv("OCR97_OCR_PHASE2_ENABLED", "1" if local_production_enabled() else "0").lower() in {"1", "true", "yes"}
DEFAULT_OCR_PHASE2_MAX_PASSES = max(1, min(int(os.getenv("OCR97_OCR_PHASE2_MAX_PASSES", "3")), 5))
DEFAULT_OCR_PHASE2_REGION_RETRY_MAX = max(0, min(int(os.getenv("OCR97_OCR_PHASE2_REGION_RETRY_MAX", "4")), 8))
DEFAULT_OCR_PHASE2_REGION_RETRY_CONF_THRESHOLD = max(
    0.0,
    min(float(os.getenv("OCR97_OCR_PHASE2_REGION_RETRY_CONF_THRESHOLD", "0.45")), 1.0),
)


def _coerce_ms(value: Any) -> float:
    try:
        return round(max(0.0, float(value)), 2)
    except Exception:
        return 0.0


def _empty_latency_breakdown() -> Dict[str, float]:
    return {
        "model_load_overhead_ms": 0.0,
        "preprocessing_overhead_ms": 0.0,
        "ocr_engine_time_ms": 0.0,
        "fallback_or_chaining_overhead_ms": 0.0,
        "residual_overhead_ms": 0.0,
    }


def _clone_latency_breakdown(raw: Optional[Dict[str, Any]]) -> Dict[str, float]:
    out = _empty_latency_breakdown()
    if not isinstance(raw, dict):
        return out
    for key in out.keys():
        out[key] = _coerce_ms(raw.get(key))
    return out


def _latency_breakdown_total(raw: Optional[Dict[str, Any]]) -> float:
    if not isinstance(raw, dict):
        return 0.0
    return round(sum(_coerce_ms(raw.get(key)) for key in _empty_latency_breakdown().keys()), 2)


def _merge_latency_breakdowns(*parts: Optional[Dict[str, Any]]) -> Dict[str, float]:
    out = _empty_latency_breakdown()
    for part in parts:
        if not isinstance(part, dict):
            continue
        for key in out.keys():
            out[key] = round(out[key] + _coerce_ms(part.get(key)), 2)
    return out


def _finalize_latency_breakdown(raw: Optional[Dict[str, Any]], *, total_ms: float) -> Dict[str, float]:
    out = _clone_latency_breakdown(raw)
    classified = round(
        out["model_load_overhead_ms"]
        + out["preprocessing_overhead_ms"]
        + out["ocr_engine_time_ms"]
        + out["fallback_or_chaining_overhead_ms"],
        2,
    )
    residual = round(max(0.0, _coerce_ms(total_ms) - classified), 2)
    out["residual_overhead_ms"] = residual
    return out


def _timing_meta_defaults() -> Dict[str, Any]:
    return {
        "import_path": __file__,
        "engine_selected": "",
        "selected_preprocess": "",
        "phase2_enabled": bool(DEFAULT_OCR_PHASE2_ENABLED),
        "self_consistency_used": False,
        "region_retry_count": 0,
        "engine_chain_length": 0,
        "fast_accept_applied": False,
        "fast_accept_reason": "",
        "fast_accept_thresholds": {},
        "second_local_engine_skipped": False,
    }


def _build_timing_meta(
    raw: Optional[Dict[str, Any]] = None,
    *,
    engine_selected: str = "",
    selected_preprocess: str = "",
    phase2_enabled: Optional[bool] = None,
    self_consistency_used: Optional[bool] = None,
    region_retry_count: Optional[int] = None,
    engine_chain_length: Optional[int] = None,
) -> Dict[str, Any]:
    meta = _timing_meta_defaults()
    if isinstance(raw, dict):
        meta.update(
            {
                "import_path": str(raw.get("import_path") or meta["import_path"]),
                "engine_selected": str(raw.get("engine_selected") or meta["engine_selected"]),
                "selected_preprocess": str(raw.get("selected_preprocess") or meta["selected_preprocess"]),
                "phase2_enabled": bool(raw.get("phase2_enabled")) if raw.get("phase2_enabled") is not None else meta["phase2_enabled"],
                "self_consistency_used": bool(raw.get("self_consistency_used")) if raw.get("self_consistency_used") is not None else meta["self_consistency_used"],
                "region_retry_count": int(raw.get("region_retry_count") or 0),
                "engine_chain_length": int(raw.get("engine_chain_length") or 0),
                "fast_accept_applied": bool(raw.get("fast_accept_applied")) if raw.get("fast_accept_applied") is not None else meta["fast_accept_applied"],
                "fast_accept_reason": str(raw.get("fast_accept_reason") or meta["fast_accept_reason"]),
                "fast_accept_thresholds": dict(raw.get("fast_accept_thresholds") or meta["fast_accept_thresholds"]),
                "second_local_engine_skipped": bool(raw.get("second_local_engine_skipped")) if raw.get("second_local_engine_skipped") is not None else meta["second_local_engine_skipped"],
            }
        )
    if engine_selected:
        meta["engine_selected"] = str(engine_selected)
    if selected_preprocess:
        meta["selected_preprocess"] = str(selected_preprocess)
    if phase2_enabled is not None:
        meta["phase2_enabled"] = bool(phase2_enabled)
    if self_consistency_used is not None:
        meta["self_consistency_used"] = bool(self_consistency_used)
    if region_retry_count is not None:
        meta["region_retry_count"] = int(region_retry_count)
    if engine_chain_length is not None:
        meta["engine_chain_length"] = int(engine_chain_length)
    return meta
DEFAULT_OCR_PHASE2_COLUMN_SPLIT_ENABLED = os.getenv("OCR97_OCR_PHASE2_COLUMN_SPLIT_ENABLED", "1").lower() in {"1", "true", "yes"}
DEFAULT_OCR_SURYA_COLUMN_SPLIT_ENABLED = os.getenv("OCR97_OCR_SURYA_COLUMN_SPLIT_ENABLED", "1").lower() in {"1", "true", "yes"}
DEFAULT_OCR_SURYA_DEVICE = str(os.getenv("OCR97_OCR_SURYA_DEVICE", "cuda" if local_production_enabled() else "cpu")).strip().lower() or ("cuda" if local_production_enabled() else "cpu")
DEFAULT_OCR_DOCUNET_URL = str(os.getenv("OCR97_OCR_DOCUNET_URL", "")).strip()
DEFAULT_OCR_REALESRGAN_URL = str(os.getenv("OCR97_OCR_REALESRGAN_URL", "")).strip()
DEFAULT_OCR_FINBERT_VERIFY = os.getenv("OCR97_OCR_FINBERT_VERIFY", "1" if local_production_enabled() else "0").lower() in {"1", "true", "yes"}
DEFAULT_OCR_TABLEFORMER_URL = str(os.getenv("OCR97_OCR_TABLEFORMER_URL", "")).strip()
DEFAULT_OCR_LGPMA_URL = str(os.getenv("OCR97_OCR_LGPMA_URL", "")).strip()
DEFAULT_OCR_FINBERT_URL = str(os.getenv("OCR97_OCR_FINBERT_URL", "")).strip()
DEFAULT_OCR_FINGERPRINT_PATH = Path(
    str(os.getenv("OCR97_OCR_FINGERPRINT_PATH", "")).strip()
    or str(os.getenv("OCR97_OCR_FINGERPRINT_PATH", "")).strip()
    or str(_PATHS.state_dir / "ocr_fingerprints.json")
)


def _truthy(value: Any, default: bool = False) -> bool:
    if value is None:
        return default
    return str(value).strip().lower() in {"1", "true", "yes", "on"}


def _looks_like_vision_model_name(model_name: str) -> bool:
    name = str(model_name or "").strip().lower()
    if not name:
        return False
    vision_tokens = (
        "vl",
        "vision",
        "llava",
        "minicpm-v",
        "got-ocr",
        "ocr",
    )
    return any(token in name for token in vision_tokens)


def _normalize_text(text: str, max_chars: int = 4000) -> str:
    text = re.sub(r"\s+", " ", (text or "")).strip()
    if len(text) > max_chars:
        return text[: max_chars - 3] + "..."
    return text


def _repair_flattened_markdown_tables(text: str) -> str:
    repaired_lines: list[str] = []
    for raw_line in str(text or "").splitlines():
        line = raw_line.strip()
        if line.count("|") >= 8 and " | | " in line:
            line = re.sub(r"\|\s+\|(?=\s*(?:---|[:\-]|[A-Za-z0-9$%#(]))", "|\n|", line)
        repaired_lines.append(line)
    return "\n".join(repaired_lines)


def _repair_flattened_markdown_blocks(text: str) -> str:
    raw = str(text or "")
    raw = re.sub(r"```\s+```", "```\n\n```", raw)
    raw = re.sub(r"(?<!\n)(#{2,6}\s)", r"\n\1", raw)
    raw = re.sub(r"\s+(#{2,6}\s)", r"\n\1", raw)
    raw = re.sub(r"\s+-\s+\*\*", r"\n- **", raw)
    raw = re.sub(r"\s+-\s+", r"\n- ", raw)
    raw = re.sub(r"```\s*(#{2,6}\s)", r"```\n\1", raw)
    return raw


def _normalize_markdown_layout(text: str, max_chars: int = 4000) -> str:
    raw = str(text or "").replace("\r\n", "\n").replace("\r", "\n")
    lower_raw = raw.lower()
    explanatory_markers = (
        "please note",
        "to perform ocr",
        "i can only provide guidance",
        "without the actual document image",
        "since you haven't attached",
    )
    fenced_blocks = re.findall(r"```(?:markdown)?\s*(.*?)```", raw, flags=re.IGNORECASE | re.DOTALL)
    if fenced_blocks and any(marker in lower_raw for marker in explanatory_markers):
        kept = [block.strip() for block in fenced_blocks if block.strip()]
        if kept:
            raw = "\n\n".join(kept)
    else:
        for marker in explanatory_markers:
            idx = lower_raw.find(marker)
            if idx > 0:
                raw = raw[:idx].rstrip()
                break
    raw = "\n".join(line.rstrip() for line in raw.splitlines())
    raw = _repair_flattened_markdown_tables(raw)
    raw = _repair_flattened_markdown_blocks(raw)
    raw = re.sub(r"\n{3,}", "\n\n", raw).strip()
    if len(raw) > max_chars:
        return raw[: max_chars - 3].rstrip() + "..."
    return raw


def _image_to_cv2_gray(image: "Image.Image") -> Optional["np.ndarray"]:
    if not CV2_AVAILABLE:
        return None
    try:
        arr = np.array(image.convert("L"))
        return arr
    except Exception:
        return None


def _cv2_to_pil(gray: "np.ndarray") -> "Image.Image":
    return Image.fromarray(gray.astype("uint8"), mode="L")


def _deskew_image(gray: "np.ndarray") -> "np.ndarray":
    try:
        blur = cv2.GaussianBlur(gray, (3, 3), 0)
        thresh = cv2.threshold(blur, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)[1]
        coords = np.column_stack(np.where(thresh < 255))
        if coords.size == 0:
            return gray
        angle = cv2.minAreaRect(coords)[-1]
        if angle < -45:
            angle = -(90 + angle)
        else:
            angle = -angle
        if abs(angle) < 0.2:
            return gray
        h, w = gray.shape[:2]
        matrix = cv2.getRotationMatrix2D((w // 2, h // 2), angle, 1.0)
        return cv2.warpAffine(gray, matrix, (w, h), flags=cv2.INTER_CUBIC, borderMode=cv2.BORDER_REPLICATE)
    except Exception:
        return gray


def _clahe_enhance(gray: "np.ndarray") -> "np.ndarray":
    try:
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        return clahe.apply(gray)
    except Exception:
        return gray


def _sauvola_binarize(gray: "np.ndarray", window_size: int = 31, k: float = 0.2, dynamic_range: float = 128.0) -> "np.ndarray":
    try:
        win = max(15, int(window_size) | 1)
        gray_f = gray.astype("float32")
        mean = cv2.boxFilter(gray_f, ddepth=-1, ksize=(win, win), normalize=True, borderType=cv2.BORDER_REPLICATE)
        sqmean = cv2.boxFilter(gray_f * gray_f, ddepth=-1, ksize=(win, win), normalize=True, borderType=cv2.BORDER_REPLICATE)
        variance = np.maximum(sqmean - (mean * mean), 0.0)
        std = np.sqrt(variance)
        threshold = mean * (1.0 + float(k) * ((std / float(dynamic_range)) - 1.0))
        binary = np.where(gray_f > threshold, 255, 0).astype("uint8")
        return binary
    except Exception:
        return gray


def _adaptive_binarize(gray: "np.ndarray") -> "np.ndarray":
    return _sauvola_binarize(_clahe_enhance(gray))


def _super_res(gray: "np.ndarray") -> "np.ndarray":
    try:
        h, w = gray.shape[:2]
        if min(h, w) >= 1700:
            return gray
        scale = 2 if min(h, w) < 1400 else 1
        if scale <= 1:
            return gray
        return cv2.resize(gray, (w * scale, h * scale), interpolation=cv2.INTER_CUBIC)
    except Exception:
        return gray


def _stack_column_parts(parts: list["np.ndarray"]) -> "np.ndarray":
    if not parts:
        raise ValueError("column_parts_required")
    if len(parts) == 1:
        return parts[0]
    max_width = max(int(part.shape[1]) for part in parts)
    divider = 18
    total_height = sum(int(part.shape[0]) for part in parts) + (divider * (len(parts) - 1))
    canvas = np.full((total_height, max_width), 255, dtype="uint8")
    y = 0
    for idx, part in enumerate(parts):
        h, w = part.shape[:2]
        canvas[y : y + h, :w] = part
        y += h
        if idx < len(parts) - 1:
            y += divider
    return canvas


def _surya_predictor():
    global _SURYA_LAYOUT_PREDICTOR, _SURYA_LAYOUT_ERROR
    if not SURYA_AVAILABLE or not DEFAULT_OCR_SURYA_COLUMN_SPLIT_ENABLED:
        return None
    if _SURYA_LAYOUT_PREDICTOR is not None:
        return _SURYA_LAYOUT_PREDICTOR
    with _SURYA_LOCK:
        if _SURYA_LAYOUT_PREDICTOR is not None:
            return _SURYA_LAYOUT_PREDICTOR
        try:
            foundation = FoundationPredictor(device=DEFAULT_OCR_SURYA_DEVICE)
            _SURYA_LAYOUT_PREDICTOR = LayoutPredictor(foundation)
            _SURYA_LAYOUT_ERROR = ""
            return _SURYA_LAYOUT_PREDICTOR
        except Exception as exc:
            _SURYA_LAYOUT_ERROR = f"surya_layout_init_failed:{exc}"
            return None


def _polygon_bounds(poly: Any) -> Optional[tuple[int, int, int, int]]:
    try:
        points = list(poly or [])
        if not points:
            return None
        xs: list[float] = []
        ys: list[float] = []
        if isinstance(points[0], (list, tuple)) and len(points[0]) >= 2:
            for p in points:
                xs.append(float(p[0]))
                ys.append(float(p[1]))
        elif len(points) >= 4:
            for i in range(0, len(points) - 1, 2):
                xs.append(float(points[i]))
                ys.append(float(points[i + 1]))
        if not xs or not ys:
            return None
        x0 = max(0, int(min(xs)))
        y0 = max(0, int(min(ys)))
        x1 = max(x0 + 1, int(max(xs)))
        y1 = max(y0 + 1, int(max(ys)))
        return x0, y0, x1, y1
    except Exception:
        return None


def _split_columns_surya(gray: "np.ndarray") -> list["np.ndarray"]:
    if not (SURYA_AVAILABLE and DEFAULT_OCR_SURYA_COLUMN_SPLIT_ENABLED):
        return [gray]
    predictor = _surya_predictor()
    if predictor is None:
        return [gray]
    try:
        image = _cv2_to_pil(gray)
        results = predictor([image], batch_size=1, top_k=3)
        if not results:
            return [gray]
        bboxes = list(getattr(results[0], "bboxes", []) or [])
        if not bboxes:
            return [gray]
        h, w = gray.shape[:2]
        mid = w // 2
        left_max_x = 0
        right_min_x = w
        left_hits = 0
        right_hits = 0
        for box in bboxes:
            label = str(getattr(box, "label", "") or "").lower()
            if label and not any(token in label for token in ("text", "title", "table", "list", "figure", "caption")):
                continue
            bounds = _polygon_bounds(getattr(box, "polygon", None))
            if not bounds:
                continue
            x0, _y0, x1, _y1 = bounds
            bw = x1 - x0
            if bw < 60:
                continue
            cx = (x0 + x1) // 2
            if cx < mid:
                left_hits += 1
                left_max_x = max(left_max_x, x1)
            else:
                right_hits += 1
                right_min_x = min(right_min_x, x0)
        if left_hits < 2 or right_hits < 2:
            return [gray]
        split_at = (left_max_x + right_min_x) // 2
        if split_at < int(w * 0.28) or split_at > int(w * 0.72):
            return [gray]
        left_img = gray[:, :split_at]
        right_img = gray[:, split_at:]
        if left_img.shape[1] < 120 or right_img.shape[1] < 120:
            return [gray]
        return [left_img, right_img]
    except Exception:
        return [gray]


def _split_columns(gray: "np.ndarray") -> list["np.ndarray"]:
    if not DEFAULT_OCR_PHASE2_COLUMN_SPLIT_ENABLED:
        return [gray]
    try:
        _h, w = gray.shape[:2]
        if w < 900:
            return [gray]
    except Exception:
        return [gray]
    surya_cols = _split_columns_surya(gray)
    if len(surya_cols) > 1:
        return surya_cols
    try:
        vertical_profile = np.mean(gray < 200, axis=0)
        center = w // 2
        window = max(60, int(w * 0.15))
        left = max(10, center - window)
        right = min(w - 10, center + window)
        band = vertical_profile[left:right]
        if band.size == 0:
            return [gray]
        split_at = int(np.argmin(band)) + left
        if split_at < int(w * 0.28) or split_at > int(w * 0.72):
            return [gray]
        left_img = gray[:, :split_at]
        right_img = gray[:, split_at:]
        if left_img.shape[1] < 120 or right_img.shape[1] < 120:
            return [gray]
        return [left_img, right_img]
    except Exception:
        return [gray]


def _preprocess_variants(image: "Image.Image") -> list[Dict[str, Any]]:
    source_image = image.convert("RGB")
    rectified = _preprocess_service_image(source_image, DEFAULT_OCR_DOCUNET_URL, timeout_sec=45) or source_image
    enhanced = _preprocess_service_image(rectified, DEFAULT_OCR_REALESRGAN_URL, extra_payload={"outscale": 2.0}, timeout_sec=60) or rectified
    base_gray = enhanced.convert("L")
    base_sharp = ImageEnhance.Contrast(base_gray.filter(ImageFilter.SHARPEN)).enhance(2.0)
    variants: list[Dict[str, Any]] = [{"name": "base", "image": base_sharp}]
    gray_cv = _image_to_cv2_gray(enhanced)
    if gray_cv is None:
        return variants
    dewarped = _deskew_image(gray_cv)
    clahe = _clahe_enhance(dewarped)
    sauvola = _sauvola_binarize(clahe)
    super_res = _super_res(sauvola)
    column_parts = _split_columns(super_res)
    variants.append({"name": "deskew_clahe_sauvola", "image": _cv2_to_pil(super_res)})
    if len(column_parts) > 1:
        reading_order = _stack_column_parts(column_parts)
        variants.append({"name": "column_detection_page_split_reading_order", "image": _cv2_to_pil(reading_order)})
    # Multi-resolution ensemble candidate for the same engine.
    variants.append({"name": "multi_res_1_5x", "image": base_sharp.resize((int(base_sharp.width * 1.5), int(base_sharp.height * 1.5)))})
    return variants


def _preprocess_image(image: "Image.Image") -> "Image.Image":
    return _preprocess_variants(image)[0]["image"]


def _norm_region_conf(value: Any) -> float:
    try:
        conf = float(value)
    except Exception:
        conf = 0.0
    if conf > 1.0:
        conf = conf / 100.0
    return round(max(0.0, min(conf, 1.0)), 4)


def _extract_tesseract_regions(
    ocr_data: Dict[str, Any],
    max_regions: int = 4,
    conf_threshold: float = DEFAULT_OCR_PHASE2_REGION_RETRY_CONF_THRESHOLD,
) -> list[Dict[str, Any]]:
    regions: list[Dict[str, Any]] = []
    try:
        total = len(ocr_data.get("text", []))
        for i in range(total):
            text = str(ocr_data["text"][i] or "").strip()
            if not text:
                continue
            conf = _norm_region_conf(ocr_data["conf"][i])
            if conf >= conf_threshold:
                continue
            x = int(ocr_data["left"][i])
            y = int(ocr_data["top"][i])
            w = int(ocr_data["width"][i])
            h = int(ocr_data["height"][i])
            if w <= 0 or h <= 0:
                continue
            regions.append(
                {
                    "x": x,
                    "y": y,
                    "w": w,
                    "h": h,
                    "conf": conf,
                    "text": text,
                    "source_engine": "tesseract",
                }
            )
    except Exception:
        return []
    regions = sorted(regions, key=lambda row: float(row.get("conf", 1.0)))
    return regions[:max_regions]


def _extract_rapidocr_regions(
    rapid_rows: Any,
    max_regions: int = 4,
    conf_threshold: float = DEFAULT_OCR_PHASE2_REGION_RETRY_CONF_THRESHOLD,
) -> list[Dict[str, Any]]:
    regions: list[Dict[str, Any]] = []
    for row in rapid_rows or []:
        if not isinstance(row, (list, tuple)) or len(row) < 3:
            continue
        points = row[0] if isinstance(row[0], (list, tuple)) else []
        text = str(row[1] or "").strip()
        conf = _norm_region_conf(row[2])
        if not text or conf >= conf_threshold:
            continue
        xs: list[float] = []
        ys: list[float] = []
        for point in points:
            if not isinstance(point, (list, tuple)) or len(point) < 2:
                continue
            try:
                xs.append(float(point[0]))
                ys.append(float(point[1]))
            except Exception:
                continue
        if not xs or not ys:
            continue
        min_x = int(max(0, min(xs)))
        min_y = int(max(0, min(ys)))
        max_x = int(max(xs))
        max_y = int(max(ys))
        width = max(1, max_x - min_x)
        height = max(1, max_y - min_y)
        regions.append(
            {
                "x": min_x,
                "y": min_y,
                "w": width,
                "h": height,
                "conf": conf,
                "text": text,
                "source_engine": "rapidocr",
            }
        )
    regions = sorted(regions, key=lambda row: float(row.get("conf", 1.0)))
    return regions[:max_regions]


def _encode_image_base64(path: Path) -> str:
    with open(path, "rb") as handle:
        return base64.b64encode(handle.read()).decode("utf-8")


def _normalize_gateway_url(url: str) -> str:
    raw = str(url or "").strip()
    if not raw:
        return ""
    if raw.endswith("/ocr/extract"):
        return raw
    return raw.rstrip("/") + "/ocr/extract"


def _utc_iso() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat().replace("+00:00", "Z")


def _load_fingerprint_store() -> Dict[str, Any]:
    path = Path(DEFAULT_OCR_FINGERPRINT_PATH)
    try:
        if path.exists():
            return dict(json.loads(path.read_text(encoding="utf-8")))
    except Exception:
        return {}
    return {}


def _save_fingerprint_store(store: Dict[str, Any]) -> None:
    path = Path(DEFAULT_OCR_FINGERPRINT_PATH)
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        tmp = path.with_suffix(path.suffix + ".tmp")
        tmp.write_text(json.dumps(store, indent=2, sort_keys=True), encoding="utf-8")
        tmp.replace(path)
    except Exception:
        return


def _semantic_diff_check(source_path: str, text: str) -> Dict[str, Any]:
    return ocr_local_inference.semantic_diff_check(
        source_path,
        text,
        fingerprint_path=Path(DEFAULT_OCR_FINGERPRINT_PATH),
    )


def _http_endpoint_reachable(url: str, timeout_sec: int = 3) -> Dict[str, Any]:
    target = str(url or "").strip()
    if not target:
        return {"ok": False, "reason": "url_unset", "status_code": 0, "url": target}
    methods = (requests.get, requests.head)
    last_error = ""
    for method in methods:
        try:
            response = method(target, timeout=timeout_sec)
            status_code = int(getattr(response, "status_code", 0) or 0)
            if status_code in {200, 201, 202, 204, 400, 401, 403, 405, 415, 422}:
                return {"ok": True, "reason": f"http_{status_code}", "status_code": status_code, "url": target}
            last_error = f"http_{status_code}"
        except Exception as exc:
            last_error = f"{type(exc).__name__}:{exc}"
    return {"ok": False, "reason": last_error or "endpoint_unreachable", "status_code": 0, "url": target}


def _http_json_post(url: str, payload: Dict[str, Any], timeout_sec: int = 5) -> Dict[str, Any]:
    target = str(url or "").strip()
    if not target:
        return {"ok": False, "error": "url_unset"}
    try:
        response = requests.post(target, json=payload, timeout=timeout_sec)
    except Exception as exc:
        return {"ok": False, "error": f"request_failed:{type(exc).__name__}:{exc}"}
    if not response.ok:
        return {"ok": False, "error": f"http_{response.status_code}"}
    try:
        data = response.json()
    except Exception as exc:
        return {"ok": False, "error": f"json_failed:{type(exc).__name__}:{exc}"}
    return {"ok": True, "data": data}


def _service_health_url(url: str) -> str:
    target = str(url or "").strip().rstrip("/")
    if not target:
        return ""
    for suffix in ("/rectify", "/upscale", "/reconstruct", "/eval"):
        if target.endswith(suffix):
            return target[: -len(suffix)] + "/health"
    return ""


def _phase2_literal_service_state(url: str, *, mode_name: str) -> Dict[str, Any]:
    target = str(url or "").strip()
    if not target:
        return {"classification": "service_hook_only", "configured": False, "healthy": False, "mode": "deferred", "url": target}
    health_url = _service_health_url(target)
    health = _http_endpoint_reachable(health_url or target)
    payload: Dict[str, Any] = {}
    if health_url:
        try:
            response = requests.get(health_url, timeout=5)
            if response.ok:
                payload = dict(response.json() or {})
        except Exception:
            payload = {}
    runtime_loaded = bool(payload.get("runtime_loaded"))
    backend = str(payload.get("backend") or "")
    classification = "implemented_literal" if health.get("ok") and runtime_loaded and mode_name in backend else "service_hook_only"
    return {
        "classification": classification,
        "configured": True,
        "healthy": bool(health.get("ok")),
        "runtime_loaded": runtime_loaded,
        "mode": "service",
        "url": target,
        "health_url": health_url,
        "backend": backend,
        "health": health,
    }


def _phase2_service_state(url: str, *, configured_status: str = "service_hook_only") -> Dict[str, Any]:
    target = str(url or "").strip()
    if not target:
        return {"classification": configured_status, "configured": False, "healthy": False, "mode": "deferred", "url": target}
    health = _http_endpoint_reachable(target)
    return {
        "classification": configured_status,
        "configured": True,
        "healthy": bool(health.get("ok")),
        "mode": "service",
        "url": target,
        "health": health,
    }


def _image_to_b64(image: "Image.Image") -> str:
    buffer = BytesIO()
    image.save(buffer, format="PNG")
    return base64.b64encode(buffer.getvalue()).decode("ascii")


def _b64_to_image(raw: str) -> Optional["Image.Image"]:
    token = str(raw or "").strip()
    if not token:
        return None
    if "," in token and token.lower().startswith("data:image/"):
        token = token.split(",", 1)[1]
    try:
        return Image.open(BytesIO(base64.b64decode(token))).convert("RGB")
    except Exception:
        return None


def _preprocess_service_image(image: "Image.Image", url: str, *, extra_payload: Optional[Dict[str, Any]] = None, timeout_sec: int = 30) -> Optional["Image.Image"]:
    if not PIL_AVAILABLE or not url:
        return None
    payload = {"image_b64": _image_to_b64(image)}
    payload.update(dict(extra_payload or {}))
    response = _http_json_post(url, payload, timeout_sec=timeout_sec)
    if not response.get("ok"):
        return None
    data = dict(response.get("data") or {})
    if not data.get("ok"):
        return None
    return _b64_to_image(str(data.get("image_b64") or ""))


def _ollama_model_available(base_url: str, model_names: list[str], timeout_sec: int = 5) -> Dict[str, Any]:
    target = str(base_url or "").strip().rstrip("/")
    if not target:
        return {"ok": False, "reason": "ollama_url_unset", "available_model": ""}
    try:
        response = requests.get(f"{target}/api/tags", timeout=timeout_sec)
    except Exception as exc:
        return {"ok": False, "reason": f"ollama_tags_failed:{type(exc).__name__}:{exc}", "available_model": ""}
    if not response.ok:
        return {"ok": False, "reason": f"ollama_tags_http_{response.status_code}", "available_model": ""}
    try:
        payload = response.json()
    except Exception as exc:
        return {"ok": False, "reason": f"ollama_tags_json_failed:{type(exc).__name__}:{exc}", "available_model": ""}
    models = []
    for item in list(payload.get("models") or []):
        name = str(item.get("name") or "").strip().lower()
        if name:
            models.append(name)
    for candidate in [str(item or "").strip().lower() for item in model_names if str(item or "").strip()]:
        for present in models:
            if present == candidate or present.startswith(f"{candidate}:") or candidate.startswith(f"{present}:"):
                return {"ok": True, "reason": "model_available", "available_model": present}
    return {"ok": False, "reason": "model_not_loaded", "available_model": ""}


def gb10_ocr_backend_readiness(payload: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    cfg = dict(payload or {})
    use_gateway = _truthy(cfg.get("use_gateway"), default=DEFAULT_GB10_OCR_USE_GATEWAY)
    gateway_url = str(cfg.get("gateway_url") or DEFAULT_GB10_OCR_GATEWAY_URL).strip()
    paddle_url = str(cfg.get("paddle_url") or DEFAULT_GB10_PADDLEOCR_VL_URL).strip()
    got_url = str(cfg.get("got_url") or DEFAULT_GB10_GOT_OCR_URL).strip()
    qwen_primary = _truthy(cfg.get("qwen_primary"), default=DEFAULT_GB10_QWEN_PRIMARY)
    qwen_model = str(cfg.get("qwen_model") or DEFAULT_GB10_QWEN_OCR_MODEL).strip()
    qwen_fallback_model = str(cfg.get("qwen_fallback_model") or DEFAULT_GB10_QWEN_OCR_FALLBACK_MODEL).strip()
    qwen_ollama_url = str(cfg.get("qwen_ollama_url") or DEFAULT_GB10_QWEN_OLLAMA_URL).strip()

    gateway_extract = _normalize_gateway_url(gateway_url)
    gateway_health = ""
    if gateway_extract:
        gateway_health = gateway_extract[: -len("/ocr/extract")] + "/ocr/health"
    gateway_check = _http_endpoint_reachable(gateway_health) if use_gateway and gateway_health else {"ok": False, "reason": "gateway_health_unchecked", "url": gateway_health}
    gateway_extract_check = _http_endpoint_reachable(gateway_extract) if use_gateway and gateway_extract else {"ok": False, "reason": "gateway_unset", "url": gateway_extract}
    direct_paddle_check = _http_endpoint_reachable(paddle_url)
    direct_got_check = _http_endpoint_reachable(got_url)
    qwen_check = _ollama_model_available(
        qwen_ollama_url,
        [qwen_model, qwen_fallback_model],
    )
    qwen_model_config_vision = _looks_like_vision_model_name(qwen_model) or _looks_like_vision_model_name(qwen_fallback_model)
    qwen_model_runtime_vision = _looks_like_vision_model_name(str(qwen_check.get("available_model") or ""))
    if qwen_primary and not qwen_model_config_vision:
        qwen_check = {
            "ok": False,
            "reason": "non_vision_model_configured",
            "available_model": str(qwen_check.get("available_model") or ""),
        }
    elif qwen_primary and qwen_check.get("ok") and not qwen_model_runtime_vision:
        qwen_check = {
            "ok": False,
            "reason": "non_vision_model_loaded",
            "available_model": str(qwen_check.get("available_model") or ""),
        }

    ready = False
    mode = "unavailable"
    fail_reason = "ocr_backend_not_ready"
    if use_gateway and gateway_extract and (gateway_check.get("ok") or gateway_extract_check.get("ok")):
        ready = True
        mode = "gateway"
        fail_reason = ""
    elif direct_paddle_check.get("ok") and direct_got_check.get("ok"):
        ready = True
        mode = "direct"
        fail_reason = ""
    elif qwen_primary and qwen_check.get("ok"):
        ready = True
        mode = "qwen_primary"
        fail_reason = ""
    elif qwen_primary and not qwen_check.get("ok"):
        fail_reason = f"qwen_primary_unready:{qwen_check.get('reason')}"
    elif not (use_gateway and gateway_extract) and not paddle_url and not got_url:
        fail_reason = "gateway_and_direct_unconfigured"
    elif use_gateway and gateway_extract and not (gateway_check.get("ok") or gateway_extract_check.get("ok")):
        fail_reason = f"gateway_unhealthy:{gateway_check.get('reason') or gateway_extract_check.get('reason')}"
    elif not (direct_paddle_check.get("ok") and direct_got_check.get("ok")):
        fail_reason = f"direct_unhealthy:paddle={direct_paddle_check.get('reason')},got={direct_got_check.get('reason')}"

    return {
        "ready": bool(ready),
        "mode": mode,
        "fail_reason": str(fail_reason or ""),
        "ts": _utc_iso(),
        "checks": {
            "gateway_enabled": bool(use_gateway),
            "gateway_extract": gateway_extract_check,
            "gateway_health": gateway_check,
            "direct_paddle": direct_paddle_check,
            "direct_got": direct_got_check,
            "qwen_primary": bool(qwen_primary),
            "qwen_check": qwen_check,
            "surya_layout": {
                "available": bool(SURYA_AVAILABLE),
                "enabled": bool(DEFAULT_OCR_SURYA_COLUMN_SPLIT_ENABLED),
                "device": DEFAULT_OCR_SURYA_DEVICE,
                "last_error": str(_SURYA_LAYOUT_ERROR or ""),
            },
        },
    }


def _tesseract_ocr(path: Path, max_chars: int = 4000) -> Dict[str, Any]:
    if not TESS_AVAILABLE:
        return {"ok": False, "error": "tesseract_unavailable"}
    try:
        overall_start = time.perf_counter()
        image = Image.open(path)
        preprocess_started = time.perf_counter()
        variants = _preprocess_variants(image) if DEFAULT_OCR_PHASE2_ENABLED else [{"name": "base", "image": _preprocess_image(image)}]
        preprocess_ms = (time.perf_counter() - preprocess_started) * 1000.0
        best_text = ""
        best_conf = -1.0
        best_regions: list[Dict[str, int]] = []
        best_variant_name = ""
        attempted = []
        primary_ocr_ms = 0.0
        extra_ocr_ms = 0.0
        for variant in variants:
            variant_name = str(variant.get("name") or "base")
            processed = variant.get("image")
            if processed is None:
                continue
            ocr_started = time.perf_counter()
            ocr_data = pytesseract.image_to_data(processed, output_type=pytesseract.Output.DICT)
            variant_ocr_ms = (time.perf_counter() - ocr_started) * 1000.0
            texts = []
            confidences = []
            for i, text in enumerate(ocr_data["text"]):
                token = str(text or "").strip()
                if not token:
                    continue
                conf = float(ocr_data["conf"][i])
                if conf > 0:
                    texts.append(token)
                    confidences.append(conf)
            full_text = _normalize_text(" ".join(texts), max_chars)
            avg_conf = sum(confidences) / len(confidences) / 100.0 if confidences else 0.0
            attempted.append({"variant": variant_name, "chars": len(full_text), "confidence": round(avg_conf, 3), "ocr_engine_time_ms": round(variant_ocr_ms, 2)})
            if not primary_ocr_ms:
                primary_ocr_ms = variant_ocr_ms
            else:
                extra_ocr_ms += variant_ocr_ms
            if (avg_conf > best_conf and len(full_text) >= max(20, int(len(best_text) * 0.85))) or len(full_text) > len(best_text):
                best_text = full_text
                best_conf = avg_conf
                best_regions = _extract_tesseract_regions(ocr_data)
                best_variant_name = variant_name
        total_ms = round((time.perf_counter() - overall_start) * 1000.0, 2)
        return {
            "ok": bool(best_text),
            "engine": "tesseract",
            "text": best_text,
            "confidence": round(max(0.0, best_conf), 3),
            "confidence_map_regions": best_regions,
            "preprocess_variants": attempted,
            "selected_preprocess": best_variant_name or "base",
            "latency_breakdown": _finalize_latency_breakdown(
                {
                    "model_load_overhead_ms": 0.0,
                    "preprocessing_overhead_ms": round(preprocess_ms, 2),
                    "ocr_engine_time_ms": round(primary_ocr_ms, 2),
                    "fallback_or_chaining_overhead_ms": round(extra_ocr_ms, 2),
                },
                total_ms=total_ms,
            ),
            "timing_meta": _build_timing_meta(
                engine_selected="tesseract",
                selected_preprocess=best_variant_name or "base",
                phase2_enabled=bool(DEFAULT_OCR_PHASE2_ENABLED),
                self_consistency_used=bool(len(variants) > 1),
                region_retry_count=0,
                engine_chain_length=1,
            ),
            "route": "local",
        }
    except Exception as exc:
        return {"ok": False, "error": f"tesseract_failed:{exc}"}


def _load_rapidocr():
    global _RAPID_OCR
    if _RAPID_OCR is None:
        _RAPID_OCR = RapidOCR()
    return _RAPID_OCR


def _load_rapidocr_with_timing() -> Tuple[Any, float]:
    global _RAPID_OCR
    if _RAPID_OCR is not None:
        return _RAPID_OCR, 0.0
    started = time.perf_counter()
    _RAPID_OCR = RapidOCR()
    _ENGINE_WARM_STATE["rapidocr"] = True
    return _RAPID_OCR, round((time.perf_counter() - started) * 1000.0, 2)


def _rapidocr_ocr(path: Path, max_chars: int = 4000) -> Dict[str, Any]:
    if not RAPID_AVAILABLE:
        return {"ok": False, "error": "rapidocr_unavailable"}
    try:
        overall_start = time.perf_counter()
        ocr, load_ms = _load_rapidocr_with_timing()
        start = time.perf_counter()
        result, _ = ocr(str(path))
        duration_ms = (time.perf_counter() - start) * 1000.0
    except Exception as exc:
        return {"ok": False, "error": f"rapidocr_failed:{exc}"}

    texts = []
    confidences = []
    for item in result or []:
        if len(item) >= 3:
            text = str(item[1])
            conf = float(item[2])
            if text.strip():
                texts.append(text.strip())
                confidences.append(conf)

    full_text = _normalize_text(" ".join(texts), max_chars)
    avg_conf = sum(confidences) / len(confidences) if confidences else 0.0
    regions = _extract_rapidocr_regions(result, max_regions=DEFAULT_OCR_PHASE2_REGION_RETRY_MAX)
    total_ms = round((time.perf_counter() - overall_start) * 1000.0, 2)
    return {
        "ok": True,
        "engine": "rapidocr",
        "text": full_text,
        "confidence": round(avg_conf, 3),
        "confidence_map_regions": regions,
        "duration_ms": round(duration_ms, 2),
        "selected_preprocess": "base",
        "latency_breakdown": _finalize_latency_breakdown(
            {
                "model_load_overhead_ms": round(load_ms, 2),
                "preprocessing_overhead_ms": 0.0,
                "ocr_engine_time_ms": round(duration_ms, 2),
                "fallback_or_chaining_overhead_ms": 0.0,
            },
            total_ms=total_ms,
        ),
        "timing_meta": _build_timing_meta(
            engine_selected="rapidocr",
            selected_preprocess="base",
            phase2_enabled=bool(DEFAULT_OCR_PHASE2_ENABLED),
            self_consistency_used=False,
            region_retry_count=0,
            engine_chain_length=1,
        ),
        "route": "local",
    }


def _render_pdf_pages(path: Path, max_pages: int, *, dpi: int = 200, tag: str = "") -> list[Path]:
    if not PDF_AVAILABLE:
        raise RuntimeError("pdf_unavailable")

    doc = fitz.open(str(path))
    pages = min(max_pages, doc.page_count)
    temp_paths: list[Path] = []
    try:
        scale = max(0.5, float(dpi) / 72.0)
    except Exception:
        scale = 200.0 / 72.0
    tag_suffix = re.sub(r"[^a-z0-9]+", "_", str(tag or "").strip().lower()).strip("_")
    try:
        for idx in range(pages):
            page = doc.load_page(idx)
            mat = fitz.Matrix(scale, scale)
            pix = page.get_pixmap(matrix=mat)
            mode = "RGBA" if pix.alpha else "RGB"
            img = Image.frombytes(mode, [pix.width, pix.height], pix.samples)
            dpi_suffix = f"dpi{int(round(dpi))}"
            name_parts = ["OCR97_ocr_pdf", path.stem, dpi_suffix]
            if tag_suffix:
                name_parts.append(tag_suffix)
            name_parts.extend([uuid.uuid4().hex, str(idx)])
            tmp = Path(tempfile.gettempdir()) / ("_".join(name_parts) + ".png")
            img.save(tmp)
            temp_paths.append(tmp)
    finally:
        doc.close()
    return temp_paths


def _goal_has_keyword(goal: str, keyword: str) -> bool:
    goal_norm = " ".join(part for part in _TOKEN_SPLIT_RE.split((goal or "").lower()) if part)
    keyword_norm = " ".join(part for part in _TOKEN_SPLIT_RE.split((keyword or "").lower()) if part)
    if not goal_norm or not keyword_norm:
        return False
    if " " in keyword_norm:
        pattern = r"(?:^|\s)" + re.escape(keyword_norm) + r"(?:\s|$)"
        return re.search(pattern, goal_norm) is not None
    tokens = set(goal_norm.split())
    return keyword_norm in tokens


def _needs_layout_engine(goal: str) -> bool:
    if not goal:
        return False
    keywords = [
        "table",
        "form",
        "layout",
        "handwriting",
        "checkbox",
        "signature",
        "multi-column",
        "invoice",
        "receipt",
        "diagram",
        "stamp",
        "statement",
        "brokerage statement",
        "earnings report",
    ]
    return any(_goal_has_keyword(goal, keyword) for keyword in keywords)


def _needs_dense_scan_engine(goal: str) -> bool:
    if not goal:
        return False
    keywords = [
        "handwriting",
        "tiny text",
        "dense scan",
        "fax",
        "stamp",
        "blurry",
        "low quality scan",
        "annotated",
        "math",
        "equation",
    ]
    return any(_goal_has_keyword(goal, keyword) for keyword in keywords)


def _doc_anchor_terms(path: Path) -> list[str]:
    tokens = [part for part in _TOKEN_SPLIT_RE.split(path.stem.lower()) if len(part) >= 3]
    return [token for token in tokens if token not in _ANCHOR_STOPWORDS]


def _anchor_hits(anchor_terms: list[str], text: str) -> int:
    if not anchor_terms:
        return 0
    lowered = str(text or "").lower()
    return sum(1 for token in anchor_terms if token in lowered)


def _needs_semantic_cleanup(goal: str) -> bool:
    if not goal:
        return False
    keywords = [
        "trading strategy",
        "stock trading",
        "day trading",
        "preserve headings",
        "preserve bullets",
        "markdown",
        "chart",
        "multi-column",
        "rewrite this ocr",
        "cleanup the ocr",
        "reformat the ocr",
    ]
    return any(_goal_has_keyword(goal, keyword) for keyword in keywords)


_LOCAL_FAST_ACCEPT_THRESHOLDS: Dict[str, float] = {
    "score": 0.34,
    "numeric_fidelity_score": 0.50,
    "chars": 100.0,
    "structure_score": 0.08,
}


def _goal_requests_layout_preservation(goal: str) -> bool:
    if not goal:
        return False
    keywords = [
        "table",
        "layout",
        "multi-column",
        "markdown",
        "preserve headings",
        "preserve bullets",
        "rewrite this ocr",
        "cleanup the ocr",
        "reformat the ocr",
        "semantic rewrite",
        "semantic cleanup",
    ]
    return any(_goal_has_keyword(goal, keyword) for keyword in keywords)


def _doc_class_is_layout_heavy(doc_class: str) -> bool:
    return str(doc_class or "").strip().lower() in {
        "digital_pdf",
        "table_dense",
        "scanned_pdf",
        "handwritten",
        "chart_or_figure",
        "forms_or_checkboxes",
    }


def _local_image_fast_accept_decision(
    *,
    goal: str,
    doc_class: str,
    score: float,
    numeric_fidelity_score: float,
    chars: int,
    structure_score: float,
) -> tuple[bool, str]:
    if _goal_requests_layout_preservation(goal):
        return False, "explicit_layout_or_cleanup_request"
    if score < float(_LOCAL_FAST_ACCEPT_THRESHOLDS["score"]):
        return False, "score_below_threshold"
    if numeric_fidelity_score < float(_LOCAL_FAST_ACCEPT_THRESHOLDS["numeric_fidelity_score"]):
        return False, "numeric_fidelity_below_threshold"
    if chars < int(_LOCAL_FAST_ACCEPT_THRESHOLDS["chars"]):
        return False, "chars_below_threshold"
    if structure_score >= float(_LOCAL_FAST_ACCEPT_THRESHOLDS["structure_score"]):
        return True, "structure_threshold_met"
    if not _doc_class_is_layout_heavy(doc_class):
        return True, "non_layout_heavy_doc_class"
    return False, "layout_heavy_structure_below_threshold"


def _should_run_semantic_cleanup(
    best: Dict[str, Any],
    *,
    goal: str,
    route_mode: str,
    gb10_enabled: bool,
    doc_class: str,
) -> tuple[bool, str]:
    timing_meta = dict(best.get("timing_meta") or {})
    if bool(timing_meta.get("fast_accept_applied")):
        return False, "fast_accept_applied"
    if not DEFAULT_OCR_PHASE2_ENABLED:
        return False, "phase2_disabled"
    if not DEFAULT_GB10_QWEN_CLEANUP:
        return False, "qwen_cleanup_disabled"
    if not gb10_enabled:
        return False, "gb10_disabled"
    if str(route_mode or "").strip().lower() == "balanced":
        return False, "balanced_route"
    if not _needs_semantic_cleanup(goal):
        return False, "goal_not_cleanup_specific"
    quality = dict(best.get("quality") or {})
    score = float(quality.get("score") or 0.0)
    structure = float(quality.get("structure_score") or 0.0)
    numeric = float(quality.get("numeric_fidelity_score") or 0.0)
    chars = int(quality.get("chars") or len(str(best.get("text") or best.get("markdown") or "")))
    if score >= 0.45 and structure >= 0.18 and chars >= 120:
        return False, "already_good_enough"
    if score >= 0.34 and structure >= 0.08 and numeric >= 0.50 and chars >= 100:
        return False, "no_measurable_gain_expected"
    if doc_class in {"photo", "scanned_pdf", "handwritten"} and not any(
        _goal_has_keyword(goal, keyword) for keyword in ("markdown", "preserve headings", "preserve bullets", "multi-column")
    ):
        return False, "visual_doc_without_explicit_cleanup_request"
    return True, "cleanup_candidate"


def _select_engine(goal: str, path: Path) -> Tuple[str, str]:
    if _needs_layout_engine(goal):
        return "rapidocr", "goal_requires_layout_or_handwriting"
    if path.suffix.lower() in {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff"}:
        return "tesseract", "image_text_default"
    return "tesseract", "default"


def _gb10_enabled(payload: Dict[str, Any]) -> bool:
    raw = payload.get("gb10_enabled")
    if raw is None:
        return DEFAULT_GB10_OCR_ENABLED
    return str(raw).strip().lower() in {"1", "true", "yes"}


def _result_is_usable(result: Dict[str, Any], *, min_conf: float = 0.45, min_chars: int = 40) -> bool:
    if not result.get("ok"):
        return False
    text = str(result.get("text") or "").strip()
    if len(text) < min_chars:
        return False
    conf = result.get("confidence")
    if conf is None:
        return True
    try:
        return float(conf) >= min_conf
    except Exception:
        return True


def _classify_doc_type(path: Path, goal: str, document_features: Optional[Dict[str, Any]] = None) -> str:
    features = dict(document_features or {})
    layout_class = str(features.get("layout_class") or "").strip().lower()
    if layout_class in {"forms_or_checkboxes", "chart_or_figure", "handwritten", "table_dense"}:
        return layout_class
    if bool(features.get("forms_or_checkboxes")):
        return "forms_or_checkboxes"
    if bool(features.get("has_chart")) or bool(features.get("has_figure")):
        return "chart_or_figure"
    if bool(features.get("has_handwriting")):
        return "handwritten"
    goal_norm = " ".join(str(goal or "").lower().split())
    suffix = str(path.suffix or "").lower()
    dense_scan_terms = ("handwriting", "blurry", "tiny text", "dense scan", "fax", "low quality")
    table_terms = ("table", "layout", "statement", "invoice", "form", "multi-column", "earnings", "brokerage")
    photo_terms = ("photo", "camera", "screenshot", "screen")
    if any(term in goal_norm for term in dense_scan_terms):
        return "handwritten"
    if suffix == ".pdf":
        if any(term in goal_norm for term in ("scan", "scanned", "photograph", "warped", "skew", "screen photo")):
            return "scanned_pdf"
        if any(term in goal_norm for term in table_terms):
            return "table_dense"
        return "digital_pdf"
    if any(term in goal_norm for term in photo_terms):
        return "photo"
    return "photo"


def _policy_engine_chain(doc_class: str, route_mode: str, forced_engine: str = "", document_features: Optional[Dict[str, Any]] = None) -> list[str]:
    return select_engine_chain(doc_class, route_mode, forced_engine=forced_engine)


def _table_rows(text: str) -> int:
    rows = 0
    for line in str(text or "").splitlines():
        clean = line.strip()
        if not clean:
            continue
        if ("|" in clean and clean.count("|") >= 2) or ("\t" in clean and len(clean.split("\t")) >= 3):
            rows += 1
    return rows


def _numeric_fidelity_score(text: str) -> float:
    raw = str(text or "")
    chars = max(1, len(raw))
    digits = sum(1 for ch in raw if ch.isdigit())
    finance_markers = 0
    for marker in ("$", "%", "margin", "risk", "position", "entry", "exit", "stop", "rule", "capital", "loss", "trading", "strategy", "support", "resistance", "volume", "market"):
        if marker in raw.lower():
            finance_markers += 1
    density = min(1.0, (digits / float(chars)) * 15.0)
    marker_bonus = min(1.0, finance_markers / 6.0)
    return round(max(0.0, min(1.0, (density * 0.4) + (marker_bonus * 0.6))), 3)


def _structure_score(text: str) -> float:
    raw = str(text or "")
    lines = [line for line in raw.splitlines() if line.strip()]
    if not lines:
        return 0.0
    heading_like = sum(
        1
        for line in lines[:80]
        if line.strip().endswith(":")
        or line.strip().startswith(("#", "-", "â€¢"))
        or line.strip().isupper()
    )
    table_like = _table_rows(raw)
    multi_line_bonus = min(1.0, len(lines) / 80.0)
    heading_score = min(1.0, heading_like / 12.0)
    table_score = min(1.0, table_like / 8.0)
    table_dense_bonus = 0.15 if table_like >= 6 else (0.08 if table_like >= 3 else 0.0)
    return round(max(0.0, min(1.0, (multi_line_bonus * 0.2) + (heading_score * 0.15) + (table_score * 0.5) + table_dense_bonus)), 3)


def _quality_bundle(
    text: str,
    confidence: Any,
    tables: Optional[list[Dict[str, Any]]] = None,
    finance_checks: Optional[Dict[str, Any]] = None,
    finbert_eval: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    chars = len(str(text or "").strip())
    conf = None
    try:
        if confidence is not None:
            conf = float(confidence)
    except Exception:
        conf = None
    structure_score = _structure_score(text)
    numeric_score = _numeric_fidelity_score(text)
    if finance_checks is None:
        finance_checks = _finance_consistency_checks(text, tables=tables)
    if finbert_eval is None:
        finbert_eval = _finbert_eval_signal(text)
    finance_checks = dict(finance_checks)
    finbert_eval = dict(finbert_eval)
    char_score = min(1.0, chars / 1600.0)
    conf_score = min(1.0, max(0.0, conf)) if conf is not None else 0.65
    consistency_score = float(finance_checks.get("score") or 0.0)
    score = round(
        max(
            0.0,
            min(
                1.0,
                (char_score * 0.30)
                + (structure_score * 0.30)
                + (numeric_score * 0.20)
                + (consistency_score * 0.10)
                + (conf_score * 0.10),
            ),
        ),
        3,
    )
    return {
        "score": score,
        "chars": chars,
        "confidence": conf,
        "structure_score": structure_score,
        "numeric_fidelity_score": numeric_score,
        "table_rows": _table_rows(text),
        "finance_consistency": finance_checks,
        "finbert_eval": finbert_eval,
    }


def _blocks_from_text(text: str) -> list[Dict[str, Any]]:
    blocks: list[Dict[str, Any]] = []
    for idx, line in enumerate([line for line in str(text or "").splitlines() if line.strip()][:300]):
        blocks.append({"id": idx, "text": line.strip(), "bbox": None})
    return blocks


def _tables_from_text(text: str) -> list[Dict[str, Any]]:
    rows = []
    for line in str(text or "").splitlines():
        clean = line.strip()
        if not clean:
            continue
        if "|" in clean and clean.count("|") >= 2:
            rows.append({"raw": clean, "cells": [part.strip() for part in clean.split("|") if part.strip()]})
        elif "\t" in clean and len(clean.split("\t")) >= 3:
            rows.append({"raw": clean, "cells": [part.strip() for part in clean.split("\t") if part.strip()]})
    if not rows:
        return []
    return [{"id": 0, "rows": rows[:120], "bbox": None}]


def _tableformer_or_lgpma_reconstruct(text: str, source_path: str = "") -> Dict[str, Any]:
    backends: Dict[str, Any] = {}
    backend_chain = [
        ("tableformer", DEFAULT_OCR_TABLEFORMER_URL),
        ("lgpma", DEFAULT_OCR_LGPMA_URL),
    ]
    source_token = str(source_path or "").strip()
    for name, url in backend_chain:
        backends[name] = {
            "configured": bool(url),
            "ok": False,
            "reason": "deferred_local_first" if url else "unconfigured",
        }
    if source_token:
        local = ocr_local_inference.tableformer_reconstruct(
            source_token,
            str(text or ""),
            normalize_text=_normalize_text,
        )
        if local.get("ok"):
            backends["tableformer_local"] = {"configured": True, "ok": True, "reason": "local_tables_detected"}
            return {
                "mode": str(local.get("mode") or "tableformer"),
                "ok": True,
                "tables": list(local.get("tables") or []),
                "backends": backends,
                "model": str(local.get("model") or ""),
                "detections": list(local.get("detections") or []),
                "source_path": str(local.get("source_path") or source_token),
            }
        backends["tableformer_local"] = {
            "configured": True,
            "ok": False,
            "reason": str(local.get("error") or "local_tableformer_failed"),
        }
    else:
        backends["tableformer_local"] = {"configured": False, "ok": False, "reason": "source_path_unset"}
    for name, url in backend_chain:
        if not url:
            continue
        payload = {"text": str(text or ""), "source_path": str(source_path or "").strip()}
        res = _http_json_post(url, payload, timeout_sec=12)
        if not res.get("ok"):
            backends[name] = {"configured": True, "ok": False, "reason": str(res.get("error") or "unknown")}
            continue
        data = dict(res.get("data") or {})
        tables = list(data.get("tables") or [])
        if tables:
            backends[name] = {"configured": True, "ok": True, "reason": "tables_detected"}
            return {"mode": name, "ok": True, "tables": tables, "backends": backends}
        backends[name] = {"configured": True, "ok": False, "reason": "no_tables"}
    # Phase 2 final fallback: deterministic heuristic reconstruction only.
    tables = _tables_from_text(text)
    return {
        "mode": "heuristic",
        "ok": bool(tables),
        "tables": tables,
        "backends": backends,
    }


def _table_numeric_values(tables: Optional[list[Dict[str, Any]]]) -> list[float]:
    values: list[float] = []
    for table in list(tables or []):
        for row in list(table.get("rows") or []):
            for cell in list(row.get("cells") or []):
                raw = str(cell or "").replace(",", "")
                matches = re.findall(r"-?\d+(?:\.\d+)?", raw)
                for token in matches:
                    try:
                        values.append(float(token))
                    except Exception:
                        continue
    return values


def _finance_consistency_checks(text: str, tables: Optional[list[Dict[str, Any]]] = None) -> Dict[str, Any]:
    raw = str(text or "")
    lines = [line.strip() for line in raw.splitlines() if line.strip()]
    issues: list[str] = []
    hints: list[str] = []
    percent_values: list[float] = []
    for match in re.findall(r"(-?\d+(?:\.\d+)?)\s*%", raw):
        try:
            percent_values.append(float(match))
        except Exception:
            continue
    if len(percent_values) >= 3:
        s = sum(percent_values[:12])
        if 95.0 <= s <= 105.0:
            hints.append("percent_group_balanced")
        elif s > 120.0:
            issues.append("percent_group_exceeds_120")
    for line in lines[:400]:
        m = re.search(
            r"(assets|liabilities|equity|total)\s*[:=]\s*(-?\d+(?:,\d{3})*(?:\.\d+)?)",
            line,
            flags=re.IGNORECASE,
        )
        if m:
            hints.append(f"value_seen:{m.group(1).lower()}")
    has_assets = any("assets" in line.lower() for line in lines)
    has_liab = any("liabil" in line.lower() for line in lines)
    has_equity = any("equity" in line.lower() for line in lines)
    if has_assets and (has_liab or has_equity):
        hints.append("balance_sheet_sections_detected")
    numeric_values = _table_numeric_values(tables)
    if len(numeric_values) >= 3:
        hints.append("table_numeric_values_detected")
    asset_match = re.search(r"assets?\s*[:=]?\s*\$?(-?\d+(?:,\d{3})*(?:\.\d+)?)", raw, flags=re.IGNORECASE)
    liab_match = re.search(r"liabil(?:ity|ities)\s*[:=]?\s*\$?(-?\d+(?:,\d{3})*(?:\.\d+)?)", raw, flags=re.IGNORECASE)
    equity_match = re.search(r"equity\s*[:=]?\s*\$?(-?\d+(?:,\d{3})*(?:\.\d+)?)", raw, flags=re.IGNORECASE)
    if asset_match and liab_match and equity_match:
        try:
            assets = float(asset_match.group(1).replace(",", ""))
            liabilities = float(liab_match.group(1).replace(",", ""))
            equity = float(equity_match.group(1).replace(",", ""))
            if abs(assets - (liabilities + equity)) <= max(1.0, abs(assets) * 0.02):
                hints.append("balance_sheet_balanced")
            else:
                issues.append("balance_sheet_not_balanced")
        except Exception:
            pass
    total_lines = [line for line in lines if "total" in line.lower()]
    if total_lines and numeric_values:
        hints.append("table_totals_detected")
    score = 1.0
    if issues:
        score = max(0.0, 1.0 - (len(issues) * 0.2))
    return {
        "score": round(score, 3),
        "issues": issues,
        "hints": hints[:16],
    }


def _finbert_eval_signal(text: str) -> Dict[str, Any]:
    if not DEFAULT_OCR_FINBERT_VERIFY:
        return {"enabled": False, "status": "disabled"}
    local = ocr_local_inference.finbert_eval(str(text or ""))
    if local.get("ok"):
        return {
            "enabled": True,
            "status": "ok",
            "mode": str(local.get("mode") or "finbert_service"),
            "label": str(local.get("label") or ""),
            "score": float(local.get("score") or 0.0),
            "model": str(local.get("model") or ""),
            "provider": "local_finbert_runtime",
        }
    service_error = ""
    if DEFAULT_OCR_FINBERT_URL:
        res = _http_json_post(DEFAULT_OCR_FINBERT_URL, {"text": str(text or "")}, timeout_sec=6)
        if res.get("ok"):
            data = dict(res.get("data") or {})
            return {
                "enabled": True,
                "status": "ok",
                "mode": "finbert_service",
                "label": str(data.get("label") or ""),
                "score": float(data.get("score") or 0.0),
                "provider": "remote_finbert_service",
            }
        service_error = str(res.get("error") or "")
    # Service unavailable: keep observability but neutralize score contribution.
    raw = str(text or "").lower()
    finance_tokens = (
        "balance sheet",
        "margin",
        "liquidity",
        "drawdown",
        "equity",
        "liabilities",
        "income statement",
        "cash flow",
        "trading",
        "risk",
    )
    hits = sum(1 for token in finance_tokens if token in raw)
    score = min(1.0, hits / 5.0)
    return {
        "enabled": True,
        "status": "degraded",
        "mode": "heuristic_proxy",
        "domain_score": round(score, 3),
        "token_hits": hits,
        "score_neutralized": True,
        "error": "|".join(token for token in [str(local.get("error") or "local_finbert_unavailable"), service_error] if token),
    }


def _line_vote_majority(candidates: list[Dict[str, Any]], max_chars: int) -> Dict[str, Any]:
    if not candidates:
        return {"ok": False, "error": "no_candidates"}
    line_weight: Counter[str] = Counter()
    line_source_count: Dict[str, int] = {}
    line_sources: Dict[str, set[str]] = {}
    for row in candidates:
        text = str(row.get("markdown") or row.get("text") or "")
        quality = dict(row.get("quality") or {})
        weight = max(0.2, float(quality.get("score") or 0.0))
        source_engine = str(row.get("engine") or row.get("model") or "unknown")
        seen_local: set[str] = set()
        for line in [ln.strip() for ln in text.splitlines() if ln.strip()]:
            key = re.sub(r"\s+", " ", line)
            line_weight[key] += weight
            line_sources.setdefault(key, set()).add(source_engine)
            if key not in seen_local:
                line_source_count[key] = int(line_source_count.get(key) or 0) + 1
                seen_local.add(key)
    min_sources = 2 if len(candidates) >= 2 else 1
    picked = [line for line, _score in line_weight.most_common() if int(line_source_count.get(line) or 0) >= min_sources]
    if not picked:
        best = max(candidates, key=lambda row: float((row.get("quality") or {}).get("score") or 0.0))
        return dict(best)
    markdown = _normalize_markdown_layout("\n".join(picked), max_chars=max_chars)
    quality = _quality_bundle(markdown, None)
    contributors = sorted({engine for line in picked for engine in line_sources.get(line, set()) if engine})
    return {
        "ok": bool(markdown),
        "engine": "phase2_ensemble_vote",
        "reason": "ensemble_majority_voting",
        "markdown": markdown,
        "text": _normalize_text(markdown, max_chars=max_chars),
        "confidence": quality.get("confidence"),
        "quality": quality,
        "route": "phase2_ensemble",
        "contributing_engines": contributors,
        "vote_mode": "line_majority_vote",
    }


def _phase2_base_engine_name(engine: Any) -> str:
    name = str(engine or "").strip().lower()
    if not name:
        return ""
    for separator in ("+", ":"):
        if separator in name:
            name = name.split(separator, 1)[0].strip()
    return normalize_engine_name(name)


def _run_engine_on_rendered_pages(
    source_path: Path,
    page_paths: list[Path],
    engine: str,
    goal: str,
    *,
    max_chars: int,
    route_mode: str,
    use_gateway: bool,
    variant_label: str = "",
) -> Dict[str, Any]:
    texts: list[str] = []
    confidences: list[float] = []
    confidence_regions: list[Dict[str, Any]] = []
    fallback_reason = ""
    used_model = ""
    used_route = ""
    for page_index, page_path in enumerate(page_paths):
        row = _run_engine_once(
            page_path,
            engine,
            goal,
            max_chars=max_chars,
            max_pages=1,
            route_mode=route_mode,
            use_gateway=use_gateway,
        )
        if not row.get("ok"):
            if not fallback_reason:
                fallback_reason = str(row.get("error") or row.get("fallback_reason") or "")
            continue
        markdown = str(row.get("markdown") or row.get("text") or "").strip()
        if not markdown:
            continue
        texts.append(markdown)
        if row.get("confidence") is not None:
            try:
                confidences.append(float(row.get("confidence")))
            except Exception:
                pass
        used_model = str(row.get("model") or used_model or "")
        used_route = str(row.get("route") or used_route or "")
        for region in list(row.get("confidence_map_regions") or []):
            region_row = dict(region or {})
            region_row["page"] = page_index
            region_row["source_engine"] = str(region_row.get("source_engine") or engine)
            region_row["conf"] = _norm_region_conf(region_row.get("conf"))
            confidence_regions.append(region_row)
    markdown = _normalize_markdown_layout("\n\n".join(part for part in texts if part), max_chars=max_chars)
    text = _normalize_text(markdown, max_chars=max_chars)
    avg_conf = sum(confidences) / len(confidences) if confidences else None
    quality = _quality_bundle(markdown, avg_conf)
    semantic_diff = _semantic_diff_check(str(source_path), markdown) if markdown else {"enabled": False, "status": "empty_text"}
    return {
        "ok": bool(text),
        "engine": variant_label or engine,
        "model": used_model,
        "text": text,
        "markdown": markdown,
        "confidence": round(avg_conf, 3) if avg_conf is not None else None,
        "pages": len(page_paths),
        "route": used_route or "phase2_pdf_dpi",
        "confidence_map_regions": sorted(confidence_regions, key=lambda row: float(row.get("conf", 1.0)))[
            : DEFAULT_OCR_PHASE2_REGION_RETRY_MAX
        ],
        "source_path": str(source_path),
        "quality": quality,
        "semantic_diff": semantic_diff,
        "fallback_reason": fallback_reason,
    }


def _render_image_scale_variants(path: Path, *, scales: list[float], tag: str = "") -> list[Dict[str, Any]]:
    if not (PIL_AVAILABLE and path.suffix.lower() != ".pdf"):
        return []
    tag_suffix = re.sub(r"[^a-z0-9]+", "_", str(tag or "").strip().lower()).strip("_")
    variants: list[Dict[str, Any]] = []
    try:
        with Image.open(path) as image:
            base = image.convert("RGB")
            width, height = base.size
            for scale in scales:
                try:
                    scale_value = float(scale)
                except Exception:
                    continue
                if scale_value <= 0:
                    continue
                target = base
                if abs(scale_value - 1.0) > 1e-6:
                    target = base.resize(
                        (
                            max(24, int(round(width * scale_value))),
                            max(24, int(round(height * scale_value))),
                        ),
                        Image.Resampling.LANCZOS if hasattr(Image, "Resampling") else Image.LANCZOS,
                    )
                scale_token = str(scale_value).replace(".", "_")
                name_parts = ["OCR97_ocr_img", path.stem, f"scale{scale_token}"]
                if tag_suffix:
                    name_parts.append(tag_suffix)
                name_parts.append(uuid.uuid4().hex)
                tmp = Path(tempfile.gettempdir()) / ("_".join(name_parts) + path.suffix.lower())
                target.save(tmp)
                variants.append({"scale": scale_value, "path": tmp})
    except Exception:
        return []
    return variants


def _phase2_pdf_dpi_ensemble(
    path: Path,
    engine: str,
    goal: str,
    *,
    max_chars: int,
    max_pages: int,
    route_mode: str,
    use_gateway: bool,
) -> Dict[str, Any]:
    base_engine = _phase2_base_engine_name(engine)
    if path.suffix.lower() != ".pdf":
        return {"ok": False, "error": "dpi_ensemble_non_pdf", "candidates": [], "attempts": []}
    if base_engine not in {"gb10_got_ocr2", "gb10_paddleocr_vl", "gb10_qwen_ocr", "rapidocr", "tesseract"}:
        return {"ok": False, "error": f"dpi_ensemble_engine_unsupported:{base_engine or 'unset'}", "candidates": [], "attempts": []}

    dpi_values = [150, 200, 300]
    variant_rows: list[Dict[str, Any]] = []
    attempts: list[Dict[str, Any]] = []
    for dpi in dpi_values:
        page_paths: list[Path] = []
        try:
            page_paths = _render_pdf_pages(path, max_pages, dpi=dpi, tag=base_engine)
            variant_label = f"{base_engine}:dpi_{dpi}"
            row = _run_engine_on_rendered_pages(
                path,
                page_paths,
                base_engine,
                goal,
                max_chars=max_chars,
                route_mode=route_mode,
                use_gateway=use_gateway,
                variant_label=variant_label,
            )
            row_phase2 = dict(row.get("phase2") or {})
            row["phase2"] = {
                **row_phase2,
                "self_consistency_variant": f"dpi_{dpi}",
                "dpi": dpi,
            }
            quality = dict(row.get("quality") or {})
            attempts.append(
                {
                    "pass": 0,
                    "engine": f"phase2_pdf_dpi:{variant_label}",
                    "ok": bool(row.get("ok")),
                    "latency_ms": 0.0,
                    "chars": int(quality.get("chars") or len(str(row.get("text") or ""))),
                    "confidence": row.get("confidence"),
                    "score": float(quality.get("score") or 0.0),
                    "structure_score": float(quality.get("structure_score") or 0.0),
                    "numeric_fidelity_score": float(quality.get("numeric_fidelity_score") or 0.0),
                    "fallback_reason": str(row.get("error") or row.get("fallback_reason") or ""),
                    "replaced_regions": 0,
                    "stop_reason": "",
                }
            )
            if row.get("ok"):
                variant_rows.append(row)
        except Exception as exc:
            attempts.append(
                {
                    "pass": 0,
                    "engine": f"phase2_pdf_dpi:{base_engine}:dpi_{dpi}",
                    "ok": False,
                    "latency_ms": 0.0,
                    "chars": 0,
                    "confidence": None,
                    "score": 0.0,
                    "structure_score": 0.0,
                    "numeric_fidelity_score": 0.0,
                    "fallback_reason": f"dpi_render_failed:{exc}",
                    "replaced_regions": 0,
                    "stop_reason": "",
                }
            )
        finally:
            for tmp in page_paths:
                try:
                    tmp.unlink()
                except Exception:
                    pass

    if len(variant_rows) < 2:
        return {
            "ok": False,
            "error": "dpi_ensemble_insufficient_candidates",
            "engine": base_engine,
            "candidates": variant_rows,
            "attempts": attempts,
            "dpi_values": dpi_values,
        }

    voted = _line_vote_majority(variant_rows, max_chars=max_chars)
    voted_quality = dict(voted.get("quality") or {})
    voted["source_path"] = str(path)
    voted["semantic_diff"] = _semantic_diff_check(str(path), str(voted.get("markdown") or voted.get("text") or ""))
    voted["phase2"] = {
        **dict(voted.get("phase2") or {}),
        "self_consistency_used": True,
        "vote_mode": "multi_dpi_line_vote",
        "contributing_engines": list(voted.get("contributing_engines") or []),
        "dpi_samples": list(dpi_values),
        "variant_count": len(variant_rows),
    }
    attempts.append(
        {
            "pass": 0,
            "engine": "phase2_multi_dpi_vote",
            "ok": bool(voted.get("ok")),
            "latency_ms": 0.0,
            "chars": int(voted_quality.get("chars") or len(str(voted.get("text") or ""))),
            "confidence": voted.get("confidence"),
            "score": float(voted_quality.get("score") or 0.0),
            "structure_score": float(voted_quality.get("structure_score") or 0.0),
            "numeric_fidelity_score": float(voted_quality.get("numeric_fidelity_score") or 0.0),
            "fallback_reason": str(voted.get("error") or ""),
            "replaced_regions": 0,
            "stop_reason": "",
        }
    )
    return {
        "ok": bool(voted.get("ok")),
        "engine": base_engine,
        "candidates": variant_rows,
        "attempts": attempts,
        "voted": voted,
        "dpi_values": dpi_values,
    }


def _phase2_image_scale_ensemble(
    path: Path,
    engine: str,
    goal: str,
    *,
    max_chars: int,
    max_pages: int,
    route_mode: str,
    use_gateway: bool,
) -> Dict[str, Any]:
    base_engine = _phase2_base_engine_name(engine)
    if path.suffix.lower() == ".pdf":
        return {"ok": False, "error": "scale_ensemble_pdf_only", "candidates": [], "attempts": []}
    if base_engine not in {"gb10_got_ocr2", "gb10_paddleocr_vl", "gb10_qwen_ocr", "rapidocr", "tesseract"}:
        return {"ok": False, "error": f"scale_ensemble_engine_unsupported:{base_engine or 'unset'}", "candidates": [], "attempts": []}

    scale_values = [1.0, 1.5, 2.0]
    variants = _render_image_scale_variants(path, scales=scale_values, tag=base_engine)
    variant_rows: list[Dict[str, Any]] = []
    attempts: list[Dict[str, Any]] = []
    for variant in variants:
        scale_value = float(variant.get("scale") or 1.0)
        variant_path = variant.get("path")
        try:
            if not isinstance(variant_path, Path):
                continue
            row = _run_engine_once(
                variant_path,
                base_engine,
                goal,
                max_chars=max_chars,
                max_pages=max_pages,
                route_mode=route_mode,
                use_gateway=use_gateway,
            )
            row_phase2 = dict(row.get("phase2") or {})
            row["phase2"] = {
                **row_phase2,
                "self_consistency_variant": f"scale_{scale_value}",
                "scale": scale_value,
            }
            row["engine"] = f"{base_engine}:scale_{str(scale_value).replace('.', '_')}"
            row["source_path"] = str(path)
            if row.get("ok"):
                row["semantic_diff"] = _semantic_diff_check(str(path), str(row.get("markdown") or row.get("text") or ""))
            quality = dict(row.get("quality") or {})
            attempts.append(
                {
                    "pass": 0,
                    "engine": f"phase2_image_scale:{row.get('engine')}",
                    "ok": bool(row.get("ok")),
                    "latency_ms": float(row.get("latency_ms") or 0.0),
                    "chars": int(quality.get("chars") or len(str(row.get("text") or ""))),
                    "confidence": row.get("confidence"),
                    "score": float(quality.get("score") or 0.0),
                    "structure_score": float(quality.get("structure_score") or 0.0),
                    "numeric_fidelity_score": float(quality.get("numeric_fidelity_score") or 0.0),
                    "fallback_reason": str(row.get("error") or row.get("fallback_reason") or ""),
                    "replaced_regions": 0,
                    "stop_reason": "",
                }
            )
            if row.get("ok"):
                variant_rows.append(row)
        finally:
            try:
                if isinstance(variant_path, Path):
                    variant_path.unlink()
            except Exception:
                pass

    if len(variant_rows) < 2:
        return {
            "ok": False,
            "error": "scale_ensemble_insufficient_candidates",
            "engine": base_engine,
            "candidates": variant_rows,
            "attempts": attempts,
            "scale_values": scale_values,
        }

    voted = _line_vote_majority(variant_rows, max_chars=max_chars)
    voted_quality = dict(voted.get("quality") or {})
    voted["source_path"] = str(path)
    voted["semantic_diff"] = _semantic_diff_check(str(path), str(voted.get("markdown") or voted.get("text") or ""))
    voted["phase2"] = {
        **dict(voted.get("phase2") or {}),
        "self_consistency_used": True,
        "vote_mode": "multi_scale_line_vote",
        "contributing_engines": list(voted.get("contributing_engines") or []),
        "scale_samples": list(scale_values),
        "variant_count": len(variant_rows),
    }
    attempts.append(
        {
            "pass": 0,
            "engine": "phase2_multi_scale_vote",
            "ok": bool(voted.get("ok")),
            "latency_ms": 0.0,
            "chars": int(voted_quality.get("chars") or len(str(voted.get("text") or ""))),
            "confidence": voted.get("confidence"),
            "score": float(voted_quality.get("score") or 0.0),
            "structure_score": float(voted_quality.get("structure_score") or 0.0),
            "numeric_fidelity_score": float(voted_quality.get("numeric_fidelity_score") or 0.0),
            "fallback_reason": str(voted.get("error") or ""),
            "replaced_regions": 0,
            "stop_reason": "",
        }
    )
    return {
        "ok": bool(voted.get("ok")),
        "engine": base_engine,
        "candidates": variant_rows,
        "attempts": attempts,
        "voted": voted,
        "scale_values": scale_values,
    }


def _merge_region_retry_fragments(base_markdown: str, retries: list[Dict[str, Any]], max_chars: int) -> Dict[str, Any]:
    merged = str(base_markdown or "")
    replaced_regions: list[Dict[str, Any]] = []
    for retry in retries:
        original = str(retry.get("source_text") or "").strip()
        replacement = str(retry.get("replacement_text") or "").strip()
        if not replacement:
            continue
        applied = "skipped"
        if original and original in merged:
            merged = merged.replace(original, replacement, 1)
            applied = "replace"
        elif replacement not in merged:
            merged = (merged.rstrip() + "\n" + replacement).strip()
            applied = "append"
        replaced_regions.append(
            {
                "bbox": dict(retry.get("bbox") or {}),
                "source_text": original,
                "replacement_text": replacement,
                "applied": applied,
            }
        )
    normalized = _normalize_markdown_layout(merged, max_chars=max_chars)
    return {
        "markdown": normalized,
        "text": _normalize_text(normalized, max_chars=max_chars),
        "replaced_regions": replaced_regions,
    }


def _phase2_preprocessing_status() -> Dict[str, Any]:
    surya_classification = "implemented_literal" if SURYA_AVAILABLE and DEFAULT_OCR_SURYA_COLUMN_SPLIT_ENABLED else "implemented_approximation"
    return {
        "deskew": {"classification": "implemented_literal", "enabled": bool(CV2_AVAILABLE), "active": bool(CV2_AVAILABLE)},
        "sauvola": {"classification": "implemented_literal", "enabled": bool(CV2_AVAILABLE), "active": bool(CV2_AVAILABLE)},
        "clahe": {"classification": "implemented_literal", "enabled": bool(CV2_AVAILABLE), "active": bool(CV2_AVAILABLE)},
        "surya_layout": {
            "classification": surya_classification,
            "enabled": bool(DEFAULT_OCR_SURYA_COLUMN_SPLIT_ENABLED),
            "active": bool(SURYA_AVAILABLE and DEFAULT_OCR_SURYA_COLUMN_SPLIT_ENABLED),
            "fallback": "heuristic_column_split",
            "last_error": str(_SURYA_LAYOUT_ERROR or ""),
        },
        "docunet": _phase2_literal_service_state(DEFAULT_OCR_DOCUNET_URL, mode_name="docunet"),
        "realesrgan": _phase2_literal_service_state(DEFAULT_OCR_REALESRGAN_URL, mode_name="realesrgan"),
    }


def _phase2_verification_status(
    finance_checks: Dict[str, Any],
    finbert_eval: Dict[str, Any],
    table_reconstruction: Dict[str, Any],
) -> Dict[str, Any]:
    finbert_mode = str(finbert_eval.get("mode") or "")
    finbert_ok = bool(finbert_mode == "finbert_service" and finbert_eval.get("status") == "ok")
    finbert_classification = "implemented_literal" if finbert_ok else "service_hook_only"
    table_mode = str(table_reconstruction.get("mode") or "")
    table_ok = bool(table_mode in {"tableformer", "lgpma"} and table_reconstruction.get("ok"))
    table_classification = "implemented_literal" if table_ok else "service_hook_only"
    table_component = 1.0 if table_ok else 0.0
    finbert_component = 1.0 if finbert_ok else 0.0
    verification_score = round(
        max(
            0.0,
            min(
                1.0,
                (float(finance_checks.get("score") or 0.0) * 0.6)
                + (table_component * 0.2)
                + (finbert_component * 0.2),
            ),
        ),
        3,
    )
    return {
        "finance_consistency": {"classification": "implemented_literal", **finance_checks},
        "finbert_mode": finbert_mode or "disabled",
        "finbert": {
            "classification": finbert_classification,
            "status": str(finbert_eval.get("status") or ""),
            "mode": finbert_mode,
        },
        "table_reconstruction_mode": table_mode or "disabled",
        "table_reconstruction": {
            "classification": table_classification,
            "ok": bool(table_reconstruction.get("ok")),
            "mode": table_mode,
        },
        "math_checks": list(finance_checks.get("hints") or []),
        "issues": list(finance_checks.get("issues") or []),
        "verification_score": verification_score,
    }


def _companion_tesseract_regions(path: Path, max_regions: int, conf_threshold: float) -> list[Dict[str, Any]]:
    if not (TESS_AVAILABLE and PIL_AVAILABLE):
        return []
    try:
        with Image.open(path) as image:
            processed = _preprocess_image(image)
            ocr_data = pytesseract.image_to_data(processed, output_type=pytesseract.Output.DICT)
        regions = _extract_tesseract_regions(ocr_data, max_regions=max_regions, conf_threshold=conf_threshold)
        for row in regions:
            row["source_engine"] = "tesseract_companion"
        return regions
    except Exception:
        return []


def _parse_region_retry_policy(policy: Optional[Dict[str, Any]]) -> Dict[str, Any]:
    raw = dict(policy or {})
    threshold = raw.get("threshold", raw.get("conf_threshold", DEFAULT_OCR_PHASE2_REGION_RETRY_CONF_THRESHOLD))
    try:
        conf_threshold = max(0.0, min(float(threshold), 1.0))
    except Exception:
        conf_threshold = DEFAULT_OCR_PHASE2_REGION_RETRY_CONF_THRESHOLD
    max_regions_raw = raw.get("max_regions", DEFAULT_OCR_PHASE2_REGION_RETRY_MAX)
    try:
        max_regions = max(0, min(int(max_regions_raw), 8))
    except Exception:
        max_regions = DEFAULT_OCR_PHASE2_REGION_RETRY_MAX
    retry_order_raw = raw.get("retry_order")
    retry_order: list[str] = []
    if isinstance(retry_order_raw, list):
        for item in retry_order_raw:
            name = str(item or "").strip().lower()
            if name in {"gb10_qwen_ocr", "rapidocr"} and name not in retry_order:
                retry_order.append(name)
    if not retry_order:
        retry_order = ["gb10_qwen_ocr", "rapidocr"]
    return {"conf_threshold": conf_threshold, "max_regions": max_regions, "retry_order": retry_order}


def _crop_regions_for_retry(path: Path, regions: list[Dict[str, int]], max_regions: int = 4) -> list[Path]:
    if not regions:
        return []
    out: list[Path] = []
    try:
        with Image.open(path) as image:
            width, height = image.size
            for idx, row in enumerate(regions[:max_regions]):
                x = max(0, int(row.get("x") or 0) - 8)
                y = max(0, int(row.get("y") or 0) - 8)
                w = max(10, int(row.get("w") or 0) + 16)
                h = max(10, int(row.get("h") or 0) + 16)
                box = (x, y, min(width, x + w), min(height, y + h))
                crop = image.crop(box)
                tmp = Path(tempfile.gettempdir()) / f"OCR97_ocr_phase2_region_{uuid.uuid4().hex}_{idx}.png"
                crop.save(tmp)
                out.append(tmp)
    except Exception:
        return []
    return out


def _looks_like_generic_finance_template(text: str) -> bool:
    raw = str(text or "").strip().lower()
    if not raw:
        return False
    markers = (
        "financial research document",
        "financial report",
        "market analysis",
        "key indicators",
        "sales analysis",
        "profit margins",
        "net profit",
        "expenses",
        "sector performance",
        "company performance",
        "abc corp",
        "xyz inc",
    )
    hits = sum(1 for marker in markers if marker in raw)
    return hits >= 4


def _native_pdf_text_extract(path: Path, *, max_pages: int, max_chars: int) -> Dict[str, Any]:
    if not PDF_AVAILABLE:
        return {"ok": False, "error": "native_pdf_unavailable:pymupdf_missing"}
    if path.suffix.lower() != ".pdf":
        return {"ok": False, "error": f"native_pdf_unsupported_suffix:{path.suffix.lower()}"}
    try:
        doc = fitz.open(str(path))
    except Exception as exc:
        return {"ok": False, "error": f"native_pdf_open_failed:{exc}"}
    parts: list[str] = []
    pages_read = 0
    try:
        for page in doc[: max(1, min(max_pages, len(doc)))]:
            pages_read += 1
            try:
                page_text = str(page.get_text("text", sort=True) or "").strip()
            except Exception:
                page_text = str(page.get_text() or "").strip()
            if page_text:
                parts.append(page_text)
        markdown = _normalize_markdown_layout("\n\n".join(parts), max_chars=max_chars)
        text = _normalize_text(markdown, max_chars=max_chars)
    finally:
        try:
            doc.close()
        except Exception:
            pass
    if not text:
        return {"ok": False, "error": "native_pdf_empty"}
    return {
        "ok": True,
        "engine": "native_pdf_text",
        "reason": "native_text:fitz",
        "pages": pages_read,
        "text": text,
        "markdown": markdown,
        "confidence": None,
        "route": "native_pdf",
    }


def _normalize_ocr_payload(
    raw: Dict[str, Any],
    *,
    route_mode: str,
    doc_class: str,
    engine_chain: list[str],
    attempts: Optional[list[Dict[str, Any]]] = None,
    fallback_reason: str = "",
    document_features: Optional[Dict[str, Any]] = None,
    visual_controls: Optional[list[Dict[str, Any]]] = None,
    feature_detection: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    payload = dict(raw or {})
    text = str(payload.get("text") or "").strip()
    markdown = str(payload.get("markdown") or text).strip()
    if bool(payload.get("ok")) and markdown and not list(payload.get("receipt_fields") or []):
        receipt_fields = receipt_fields_from_candidates(
            [
                {
                    "ok": True,
                    "engine": str(payload.get("engine") or ""),
                    "markdown": markdown,
                    "selection_score": float((payload.get("quality") or {}).get("score") or 0.0),
                }
            ]
        )
        if receipt_fields:
            markdown = append_receipt_fields(markdown, receipt_fields)
            text = _normalize_text(markdown, max_chars=max(4000, len(markdown)))
            payload["receipt_fields"] = receipt_fields
            payload["receipt_fields_used"] = True
    blocks = list(payload.get("blocks") or _blocks_from_text(markdown))
    existing_tables = list(payload.get("tables") or [])
    if existing_tables:
        table_reconstruction = {
            "ok": True,
            "mode": "upstream_tables",
            "tables": existing_tables,
            "reason": "tables_supplied_upstream",
            "backends": {},
        }
        tables = existing_tables
    else:
        table_reconstruction = _tableformer_or_lgpma_reconstruct(markdown, source_path=str(payload.get("source_path") or ""))
        tables = list(table_reconstruction.get("tables") or _tables_from_text(markdown))
    quality = dict(payload.get("quality") or {})
    finance_consistency = dict(quality.get("finance_consistency") or {})
    finbert_eval = dict(quality.get("finbert_eval") or {})
    if not finance_consistency:
        finance_consistency = _finance_consistency_checks(markdown, tables=tables)
    if not finbert_eval:
        finbert_eval = _finbert_eval_signal(markdown)
    if not quality:
        quality = _quality_bundle(
            markdown,
            payload.get("confidence"),
            tables=tables,
            finance_checks=finance_consistency,
            finbert_eval=finbert_eval,
        )
    else:
        quality["finance_consistency"] = finance_consistency
        quality["finbert_eval"] = finbert_eval
        quality["table_rows"] = int(quality.get("table_rows") or _table_rows(markdown))
    source_path = str(payload.get("source_path") or "").strip()
    semantic_diff = dict(payload.get("semantic_diff") or {})
    if not semantic_diff:
        semantic_diff = _semantic_diff_check(source_path, markdown) if source_path else {"enabled": False, "status": "source_path_unset"}
    reading_order = list(payload.get("reading_order") or [item.get("id") for item in blocks if isinstance(item, dict)])
    bbox = list(payload.get("bbox") or [])
    phase2_meta = dict(payload.get("phase2") or {})
    verification = _phase2_verification_status(finance_consistency, finbert_eval, table_reconstruction)
    attempts_out = list(attempts or payload.get("attempts") or [])
    latency_ms = _coerce_ms(payload.get("latency_ms"))
    latency_breakdown = _finalize_latency_breakdown(payload.get("latency_breakdown"), total_ms=latency_ms)
    ensemble = {
        "engines_considered": list(engine_chain),
        "winner": str(payload.get("engine") or ""),
        "vote_mode": str(phase2_meta.get("vote_mode") or "best_score"),
        "self_consistency_used": bool(phase2_meta.get("self_consistency_used")),
        "variant_count": int(phase2_meta.get("variant_count") or 0),
        "dpi_samples": list(phase2_meta.get("dpi_samples") or []),
        "scale_samples": list(phase2_meta.get("scale_samples") or []),
        "region_retry_count": int(phase2_meta.get("region_retry_count") or 0),
        "stop_reason": str(phase2_meta.get("stop_reason") or ""),
        "contributing_engines": list(phase2_meta.get("contributing_engines") or []),
    }
    phase2 = {
        "enabled": bool(DEFAULT_OCR_PHASE2_ENABLED),
        "passes": int(phase2_meta.get("passes") or 0),
        "attempt_count": len(attempts_out),
        "preprocessing": _phase2_preprocessing_status(),
        "ensemble": ensemble,
        "verification": verification,
        "pass_provenance": attempts_out,
    }
    payload.update(
        {
            "ok": bool(payload.get("ok")) and bool(markdown),
            "text": text,
            "markdown": markdown,
            "blocks": blocks,
            "tables": tables,
            "reading_order": reading_order,
            "bbox": bbox,
            "engine_chain": list(engine_chain),
            "attempts": attempts_out,
            "fallback_reason": fallback_reason or str(payload.get("fallback_reason") or ""),
            "latency_ms": latency_ms,
            "latency_breakdown": latency_breakdown,
            "selected_preprocess": str(payload.get("selected_preprocess") or ""),
            "timing_meta": _build_timing_meta(
                payload.get("timing_meta"),
                engine_selected=str(payload.get("engine") or ""),
                selected_preprocess=str(payload.get("selected_preprocess") or ""),
                phase2_enabled=bool(phase2.get("enabled")),
                self_consistency_used=bool(ensemble.get("self_consistency_used")),
                region_retry_count=int(ensemble.get("region_retry_count") or 0),
                engine_chain_length=len(engine_chain),
            ),
            "document_features": dict(document_features or payload.get("document_features") or {}),
            "layout_regions": list((document_features or payload.get("document_features") or {}).get("layout_regions") or payload.get("layout_regions") or []),
            "visual_controls": list(visual_controls or payload.get("visual_controls") or []),
            "feature_detection": dict(feature_detection or payload.get("feature_detection") or {}),
            "phase2": phase2,
            "quality": {
                "score": float(quality.get("score") or 0.0),
                "chars": int(quality.get("chars") or len(markdown)),
                "confidence": quality.get("confidence"),
                "structure_score": float(quality.get("structure_score") or 0.0),
                "numeric_fidelity_score": float(quality.get("numeric_fidelity_score") or 0.0),
                "table_rows": int(quality.get("table_rows") or 0),
                "finance_consistency": finance_consistency,
                "finbert_eval": finbert_eval,
                "table_reconstruction": table_reconstruction,
                "semantic_diff": semantic_diff,
                "route_mode": route_mode,
                "doc_class": doc_class,
            },
        }
    )
    return payload


def _run_engine_once(
    path: Path,
    engine: str,
    goal: str,
    *,
    max_chars: int,
    max_pages: int,
    route_mode: str = DEFAULT_OCR_ROUTE_MODE,
    use_gateway: bool = DEFAULT_GB10_OCR_USE_GATEWAY,
) -> Dict[str, Any]:
    start = time.perf_counter()
    if engine == "native_pdf_text":
        result = _native_pdf_text_extract(path, max_pages=max_pages, max_chars=max_chars)
    elif engine == "local_image_best":
        result = _local_image_best_extract(
            path,
            goal,
            max_chars=max_chars,
            max_pages=max_pages,
            route_mode=route_mode,
            use_gateway=use_gateway,
        )
    elif engine == "tesseract":
        result = _ocr_pdf_local(path, "tesseract", goal, max_pages=max_pages, max_chars=max_chars) if path.suffix.lower() == ".pdf" else _tesseract_ocr(path, max_chars=max_chars)
    elif engine == "rapidocr":
        result = _ocr_pdf_local(path, "rapidocr", goal, max_pages=max_pages, max_chars=max_chars) if path.suffix.lower() == ".pdf" else _rapidocr_ocr(path, max_chars=max_chars)
    elif engine == "gb10_qwen_ocr":
        result = _run_gb10_engine(path, engine, goal, max_chars=max_chars, max_pages=max_pages, route_mode=route_mode, use_gateway=use_gateway)
    elif engine in {"gb10_paddleocr_vl", "gb10_got_ocr2"}:
        result = _run_gb10_engine(path, engine, goal, max_chars=max_chars, max_pages=max_pages, route_mode=route_mode, use_gateway=use_gateway)
        if engine == "gb10_paddleocr_vl" and not result.get("ok"):
            fallback = _gb10_qwen_ocr(path, goal, max_chars=max_chars, max_pages=max_pages)
            if fallback.get("ok"):
                fallback["fallback_reason"] = str(result.get("error") or f"{engine}_failed")
                fallback["primary_engine"] = engine
                result = fallback
    else:
        result = {"ok": False, "engine": engine, "error": f"engine_unknown:{engine}"}
    elapsed_ms = round((time.perf_counter() - start) * 1000.0, 2)
    result = dict(result or {})
    result["latency_ms"] = elapsed_ms
    result["latency_breakdown"] = _finalize_latency_breakdown(result.get("latency_breakdown"), total_ms=elapsed_ms)
    result.setdefault("engine", engine)
    result.setdefault("model", str(result.get("model") or ""))
    result.setdefault("source_path", str(path))
    result.setdefault("page_confidence", result.get("confidence"))
    result["selected_preprocess"] = str(result.get("selected_preprocess") or "")
    result["timing_meta"] = _build_timing_meta(
        result.get("timing_meta"),
        engine_selected=str(result.get("engine") or engine),
        selected_preprocess=str(result.get("selected_preprocess") or ""),
        phase2_enabled=bool(DEFAULT_OCR_PHASE2_ENABLED),
        self_consistency_used=bool((result.get("phase2") or {}).get("self_consistency_used")),
        region_retry_count=int((result.get("phase2") or {}).get("region_retry_count") or 0),
        engine_chain_length=1,
    )
    if result.get("ok"):
        result["semantic_diff"] = _semantic_diff_check(
            str(result.get("source_path") or path),
            str(result.get("markdown") or result.get("text") or ""),
        )
    if not list(result.get("confidence_map_regions") or []) and str(result.get("engine") or engine).strip().lower() in {
        "gb10_got_ocr2",
        "gb10_qwen_ocr",
    }:
        result["confidence_map_regions"] = _companion_tesseract_regions(
            path,
            max_regions=DEFAULT_OCR_PHASE2_REGION_RETRY_MAX,
            conf_threshold=DEFAULT_OCR_PHASE2_REGION_RETRY_CONF_THRESHOLD,
        )
    result["quality"] = _quality_bundle(str(result.get("markdown") or result.get("text") or ""), result.get("confidence"))
    return result


def _remote_ocr_via_http(
    *,
    url: str,
    path: Path,
    model_name: str,
    goal: str,
    max_chars: int,
    max_pages: Optional[int] = None,
    route_mode: str = DEFAULT_OCR_ROUTE_MODE,
) -> Dict[str, Any]:
    if not url:
        return {"ok": False, "error": f"{model_name}_url_unset"}
    files = {"file": (path.name, path.read_bytes(), "application/pdf" if path.suffix.lower() == ".pdf" else "application/octet-stream")}
    data = {
        "goal": goal or "",
        "max_chars": str(max_chars),
        "model": model_name,
        "route_mode": route_mode,
    }
    if max_pages is not None:
        data["max_pages"] = str(max_pages)
    try:
        response = requests.post(url, data=data, files=files, timeout=DEFAULT_GB10_OCR_TIMEOUT_SEC)
    except Exception as exc:
        return {"ok": False, "error": f"{model_name}_request_failed:{exc}"}
    if not response.ok:
        return {"ok": False, "error": f"{model_name}_http_{response.status_code}:{response.text[:240]}"}
    try:
        payload = response.json()
    except Exception as exc:
        return {"ok": False, "error": f"{model_name}_json_failed:{exc}"}
    markdown = _normalize_markdown_layout(str(payload.get("markdown") or payload.get("text") or ""), max_chars=max_chars)
    text = _normalize_text(markdown, max_chars=max_chars)
    if _looks_like_generic_finance_template(markdown):
        return {"ok": False, "error": f"{model_name}_generic_template_detected"}
    return {
        "ok": bool(text),
        "engine": model_name,
        "model": str(payload.get("model") or model_name),
        "text": text,
        "markdown": markdown,
        "confidence": payload.get("confidence"),
        "pages": payload.get("pages"),
        "reason": str(payload.get("reason") or ""),
        "route": "gb10_remote",
        "blocks": list(payload.get("blocks") or []),
        "tables": list(payload.get("tables") or []),
        "reading_order": list(payload.get("reading_order") or []),
        "bbox": list(payload.get("bbox") or []),
        "attempts": list(payload.get("attempts") or []),
        "quality": dict(payload.get("quality") or {}),
        "engine_chain": list(payload.get("engine_chain") or []),
        "fallback_reason": str(payload.get("fallback_reason") or ""),
        "latency_ms": _coerce_ms(payload.get("latency_ms")),
        "latency_breakdown": _clone_latency_breakdown(payload.get("latency_breakdown")),
        "selected_preprocess": str(payload.get("selected_preprocess") or ""),
        "timing_meta": dict(payload.get("timing_meta") or {}),
    }


def _gateway_ocr_via_http(
    *,
    gateway_url: str,
    path: Path,
    model_name: str,
    goal: str,
    max_chars: int,
    max_pages: int,
    route_mode: str,
) -> Dict[str, Any]:
    gateway = _normalize_gateway_url(gateway_url)
    if not gateway:
        return {"ok": False, "error": "gb10_gateway_url_unset"}
    files = {"file": (path.name, path.read_bytes(), "application/pdf" if path.suffix.lower() == ".pdf" else "application/octet-stream")}
    data = {
        "goal": goal or "",
        "model": model_name,
        "max_chars": str(max_chars),
        "max_pages": str(max_pages),
        "route_mode": route_mode,
        "prewarm": "1",
    }
    try:
        response = requests.post(gateway, data=data, files=files, timeout=DEFAULT_GB10_OCR_TIMEOUT_SEC)
    except Exception as exc:
        return {"ok": False, "error": f"gb10_gateway_request_failed:{exc}"}
    if not response.ok:
        return {"ok": False, "error": f"gb10_gateway_http_{response.status_code}:{response.text[:240]}"}
    try:
        payload = response.json()
    except Exception as exc:
        return {"ok": False, "error": f"gb10_gateway_json_failed:{exc}"}
    markdown = _normalize_markdown_layout(str(payload.get("markdown") or payload.get("text") or ""), max_chars=max_chars)
    text = _normalize_text(markdown, max_chars=max_chars)
    if _looks_like_generic_finance_template(markdown):
        return {"ok": False, "error": "gb10_gateway_generic_template_detected"}
    return {
        "ok": bool(payload.get("ok")) and bool(text),
        "engine": str(payload.get("engine") or model_name),
        "model": str(payload.get("model") or model_name),
        "text": text,
        "markdown": markdown,
        "confidence": payload.get("confidence"),
        "pages": payload.get("pages"),
        "reason": str(payload.get("reason") or "gb10_gateway"),
        "route": "gb10_gateway",
        "blocks": list(payload.get("blocks") or []),
        "tables": list(payload.get("tables") or []),
        "reading_order": list(payload.get("reading_order") or []),
        "bbox": list(payload.get("bbox") or []),
        "attempts": list(payload.get("attempts") or []),
        "quality": dict(payload.get("quality") or {}),
        "engine_chain": list(payload.get("engine_chain") or []),
        "fallback_reason": str(payload.get("fallback_reason") or ""),
        "latency_ms": _coerce_ms(payload.get("latency_ms")),
        "latency_breakdown": _clone_latency_breakdown(payload.get("latency_breakdown")),
        "selected_preprocess": str(payload.get("selected_preprocess") or ""),
        "timing_meta": dict(payload.get("timing_meta") or {}),
    }


def _qwen_ocr_prompt(goal: str, draft_text: str = "", variant: str = "base") -> str:
    prompt = [
        "Perform OCR/transcription on this financial document image.",
        "Return only the extracted text in Markdown.",
        "Preserve headings, bullet lists, table-like structure, and key numeric values when visible.",
        "Keep real line breaks. Put each table row on its own line.",
        "For receipts, copy dates, totals, invoice numbers, and addresses exactly as visible.",
        "Do not normalize date formats; for example, if the image shows 02/01/2019, output 02/01/2019, not 2019-02-01.",
        "Do not correct visible years, decimal points, or currency amounts unless the character is plainly readable.",
        "Do not summarize, explain, or add commentary.",
        "Do not invent standard example content, fictional company names, generic sales/profit templates, or placeholder values.",
        "If part of the page is unclear, transcribe only what is actually visible and omit the rest.",
    ]
    if variant == "strict_literal":
        prompt.append("Strict mode: literal transcription only, no inferred labels.")
    elif variant == "numeric_focus":
        prompt.append("Numeric mode: preserve every visible numeric token exactly as shown.")
    if draft_text.strip():
        prompt.extend(["", "Draft OCR to correct:", draft_text.strip()[:4000]])
    if goal.strip():
        prompt.extend(["", f"Focus: {goal.strip()}"])
    return "\n".join(prompt).strip()


def _ollama_generate_with_image(image_path: Path, prompt: str, model: str, *, base_url: str = DEFAULT_GB10_QWEN_OLLAMA_URL, lane: str = "gb10_ollama") -> Dict[str, Any]:
    payload = {
        "model": model,
        "prompt": prompt,
        "images": [_encode_image_base64(image_path)],
        "stream": False,
        "keep_alive": os.getenv("VISION_KEEP_ALIVE", "10m"),
    }
    try:
        response = requests.post(
            f"{str(base_url).rstrip('/')}/api/generate",
            json=payload,
            timeout=DEFAULT_GB10_OCR_TIMEOUT_SEC,
        )
    except Exception as exc:
        return {"ok": False, "error": f"qwen_ocr_request_failed:{exc}"}
    if not response.ok:
        return {"ok": False, "error": f"qwen_ocr_http_{response.status_code}:{response.text[:240]}"}
    try:
        data = response.json()
    except Exception as exc:
        return {"ok": False, "error": f"qwen_ocr_json_failed:{exc}"}
    markdown = _normalize_markdown_layout(str(data.get("response") or ""), max_chars=16000)
    text = _normalize_text(markdown, max_chars=16000)
    if _looks_like_generic_finance_template(markdown):
        return {"ok": False, "error": "qwen_ocr_generic_template_detected"}
    return {
        "ok": bool(text),
        "engine": "gb10_qwen_ocr",
        "model": model,
        "text": text,
        "markdown": markdown,
        "confidence": None,
        "reason": "gb10_qwen_vlm_ocr",
        "route": lane,
    }


def _gb10_qwen_ocr(path: Path, goal: str, max_chars: int, max_pages: int, draft_text: str = "") -> Dict[str, Any]:
    draft_seed = str(draft_text or "").strip()
    if not draft_seed and path.suffix.lower() == ".pdf":
        native_seed = _native_pdf_text_extract(path, max_pages=max_pages, max_chars=max_chars)
        if native_seed.get("ok"):
            draft_seed = str(native_seed.get("markdown") or native_seed.get("text") or "").strip()[:4000]
    models = [DEFAULT_GB10_QWEN_OCR_MODEL, DEFAULT_GB10_QWEN_OCR_FALLBACK_MODEL]
    endpoint_plan = [(DEFAULT_GB10_QWEN_OLLAMA_URL, models, "gb10_ollama")]
    compat_url = str(DEFAULT_OCR_COMPAT_OLLAMA_URL or "").strip()
    if DEFAULT_OCR_COMPAT_ENABLED and compat_url and compat_url.rstrip("/") != str(DEFAULT_GB10_QWEN_OLLAMA_URL).rstrip("/"):
        compat_models = [DEFAULT_OCR_COMPAT_MODEL, DEFAULT_GB10_QWEN_OCR_FALLBACK_MODEL]
        endpoint_plan.append((compat_url, compat_models, "compat_3090"))
    page_paths = [path]
    cleanup_paths: list[Path] = []
    anchor_terms = _doc_anchor_terms(path)
    if path.suffix.lower() == ".pdf":
        try:
            page_paths = _render_pdf_pages(path, max_pages)
            cleanup_paths = page_paths
        except Exception as exc:
            return {"ok": False, "error": f"qwen_pdf_render_failed:{exc}"}
    outputs: list[str] = []
    used_model = ""
    used_lane = "gb10_ollama"
    for page_path in page_paths:
        last_error = ""
        best_page_result: Dict[str, Any] = {}
        best_page_score = -1.0
        accepted_page = False
        for base_url, model_list, lane_name in endpoint_plan:
            for model in model_list:
                if not model:
                    continue
                result = _ollama_generate_with_image(page_path, _qwen_ocr_prompt(goal, draft_text=draft_seed, variant="base"), model, base_url=base_url, lane=lane_name)
                if result.get("ok"):
                    candidate_markdown = str(result.get("markdown") or result.get("text") or "").strip()
                    candidate_quality = _quality_bundle(candidate_markdown, result.get("confidence"))
                    candidate_score = float(candidate_quality.get("score") or 0.0)
                    candidate_chars = int(candidate_quality.get("chars") or len(candidate_markdown))
                    candidate_structure = float(candidate_quality.get("structure_score") or 0.0)
                    candidate_table_rows = int(candidate_quality.get("table_rows") or 0)
                    anchor_count = _anchor_hits(anchor_terms, candidate_markdown)
                    anchor_ratio = (anchor_count / len(anchor_terms)) if anchor_terms else 0.0
                    selection_score = candidate_score
                    if candidate_structure >= 0.03:
                        selection_score += 0.08
                    if candidate_table_rows > 0:
                        selection_score += min(0.12, candidate_table_rows * 0.02)
                    if anchor_terms:
                        if anchor_count == 0:
                            selection_score -= 0.18
                        else:
                            selection_score += min(0.12, anchor_ratio * 0.18)
                    if selection_score > best_page_score:
                        best_page_result = dict(result)
                        best_page_score = selection_score
                    low_structure_pdf = (
                        path.suffix.lower() == ".pdf"
                        and candidate_structure < 0.03
                        and candidate_table_rows <= 0
                    )
                    if low_structure_pdf and selection_score < 0.22:
                        last_error = "qwen_ocr_low_structure_pdf"
                        continue
                last_error = str(result.get("error") or "qwen_ocr_failed")
        if not accepted_page and best_page_result.get("ok"):
            best_markdown = str(best_page_result.get("markdown") or best_page_result.get("text") or "").strip()
            best_quality = _quality_bundle(best_markdown, best_page_result.get("confidence"))
            best_chars = int(best_quality.get("chars") or len(best_markdown))
            if best_chars >= 120 and best_page_score >= 0.12:
                outputs.append(best_markdown)
                used_model = str(best_page_result.get("model") or used_model or "")
                used_lane = str(best_page_result.get("route") or used_lane or "gb10_ollama")
                accepted_page = True
        if accepted_page:
            continue
        else:
            return {"ok": False, "error": last_error or "qwen_ocr_failed"}
    markdown = _normalize_markdown_layout("\n\n".join(part for part in outputs if part), max_chars=max_chars)
    self_consistency_used = False
    self_consistency_contributors: list[str] = []
    self_consistency_vote_mode = ""
    if DEFAULT_OCR_PHASE2_ENABLED and page_paths and (used_model or DEFAULT_GB10_QWEN_OCR_MODEL):
        sample_candidates: list[Dict[str, Any]] = [{"engine": "gb10_qwen_ocr:base", "markdown": markdown, "quality": _quality_bundle(markdown, None)}]
        sample_model = used_model or DEFAULT_GB10_QWEN_OCR_MODEL
        sample_page = page_paths[0]
        for variant in ("strict_literal", "numeric_focus"):
            sample = _ollama_generate_with_image(
                sample_page,
                _qwen_ocr_prompt(goal, draft_text=draft_seed, variant=variant),
                sample_model,
                base_url=DEFAULT_GB10_QWEN_OLLAMA_URL,
                lane="gb10_ollama_self_consistency",
            )
            if sample.get("ok"):
                sample_markdown = str(sample.get("markdown") or sample.get("text") or "")
                sample_candidates.append(
                    {
                        "engine": f"gb10_qwen_ocr:{variant}",
                        "markdown": sample_markdown,
                        "quality": _quality_bundle(sample_markdown, sample.get("confidence")),
                    }
                )
        if len(sample_candidates) >= 2:
            voted = _line_vote_majority(sample_candidates, max_chars=max_chars)
            if voted.get("ok") and float((voted.get("quality") or {}).get("score") or 0.0) >= float((_quality_bundle(markdown, None).get("score") or 0.0)):
                markdown = str(voted.get("markdown") or markdown)
                self_consistency_used = True
                self_consistency_contributors = list(voted.get("contributing_engines") or [])
                self_consistency_vote_mode = str(voted.get("vote_mode") or "self_consistency_vote")
    text = _normalize_text(markdown, max_chars=max_chars)
    for temp_path in cleanup_paths:
        try:
            temp_path.unlink()
        except Exception:
            pass
    result = {
        "ok": bool(text),
        "engine": "gb10_qwen_ocr",
        "model": used_model or DEFAULT_GB10_QWEN_OCR_MODEL or DEFAULT_GB10_QWEN_OCR_FALLBACK_MODEL,
        "text": text,
        "markdown": markdown,
        "confidence": None,
        "pages": len(page_paths),
        "reason": "gb10_qwen_vlm_ocr",
        "route": used_lane,
    }
    if self_consistency_used:
        result["phase2"] = {
            "self_consistency_used": True,
            "vote_mode": self_consistency_vote_mode or "self_consistency_vote",
            "contributing_engines": self_consistency_contributors,
        }
    return result


def _gb10_plan(path: Path, goal: str) -> Dict[str, Any]:
    suffix = path.suffix.lower()
    if _needs_dense_scan_engine(goal):
        return {"primary": "gb10_got_ocr2", "cleanup": "gb10_qwen_ocr", "reason": "dense_scan_or_handwriting"}
    if suffix == ".pdf" or _needs_layout_engine(goal):
        return {"primary": "gb10_paddleocr_vl", "cleanup": "gb10_qwen_ocr", "reason": "document_layout_primary"}
    return {"primary": "gb10_qwen_ocr", "cleanup": "", "reason": "vision_ocr_primary"}


def _run_gb10_engine(
    path: Path,
    engine: str,
    goal: str,
    max_chars: int,
    max_pages: int,
    route_mode: str = DEFAULT_OCR_ROUTE_MODE,
    use_gateway: bool = DEFAULT_GB10_OCR_USE_GATEWAY,
) -> Dict[str, Any]:
    gateway_error = ""
    if use_gateway and DEFAULT_GB10_OCR_GATEWAY_URL and engine in {"gb10_paddleocr_vl", "gb10_qwen_ocr"}:
        gw_result = _gateway_ocr_via_http(
            gateway_url=DEFAULT_GB10_OCR_GATEWAY_URL,
            path=path,
            model_name=engine,
            goal=goal,
            max_chars=max_chars,
            max_pages=max_pages,
            route_mode=route_mode,
        )
        if gw_result.get("ok"):
            return gw_result
        if engine != "gb10_qwen_ocr":
            gateway_error = str(gw_result.get("error") or "gb10_gateway_failed")
        else:
            return gw_result
    if engine == "gb10_paddleocr_vl":
        return _remote_ocr_via_http(
            url=DEFAULT_GB10_PADDLEOCR_VL_URL,
            path=path,
            model_name="gb10_paddleocr_vl",
            goal=goal,
            max_chars=max_chars,
            max_pages=max_pages,
            route_mode=route_mode,
        )
    if engine == "gb10_got_ocr2":
        local = ocr_local_inference.got_extract_texts(
            path,
            goal,
            max_chars,
            max_pages,
            normalize_markdown=_normalize_markdown_layout,
            normalize_text=_normalize_text,
        )
        if local.get("ok"):
            return local
        local_error = str(local.get("error") or "gb10_got_ocr2_local_failed")
        remote_error = ""
        if DEFAULT_GB10_GOT_OCR_URL:
            remote = _remote_ocr_via_http(
                url=DEFAULT_GB10_GOT_OCR_URL,
                path=path,
                model_name="gb10_got_ocr2",
                goal=goal,
                max_chars=max_chars,
                max_pages=max_pages,
                route_mode=route_mode,
            )
            if remote.get("ok"):
                return remote
            remote_error = str(remote.get("error") or "gb10_got_ocr2_remote_failed")
        if use_gateway and DEFAULT_GB10_OCR_GATEWAY_URL:
            gw_result = _gateway_ocr_via_http(
                gateway_url=DEFAULT_GB10_OCR_GATEWAY_URL,
                path=path,
                model_name=engine,
                goal=goal,
                max_chars=max_chars,
                max_pages=max_pages,
                route_mode=route_mode,
            )
            if gw_result.get("ok"):
                return gw_result
            gateway_error = str(gw_result.get("error") or "gb10_gateway_failed")
        errors = [token for token in [local_error, remote_error, gateway_error] if token]
        return {"ok": False, "engine": "gb10_got_ocr2", "error": "|".join(errors) or "gb10_got_ocr2_failed"}
    if engine == "gb10_qwen_ocr":
        return _gb10_qwen_ocr(path, goal, max_chars=max_chars, max_pages=max_pages)
    return {"ok": False, "error": f"gb10_engine_unknown:{engine}"}


def _run_gb10_primary(
    path: Path,
    goal: str,
    max_chars: int,
    max_pages: int,
    forced_engine: str = "",
    route_mode: str = DEFAULT_OCR_ROUTE_MODE,
    use_gateway: bool = DEFAULT_GB10_OCR_USE_GATEWAY,
) -> Dict[str, Any]:
    plan = _gb10_plan(path, goal) if not forced_engine else {"primary": forced_engine, "cleanup": "", "reason": "forced"}
    primary = plan["primary"]
    primary_result = _run_gb10_engine(path, primary, goal, max_chars=max_chars, max_pages=max_pages, route_mode=route_mode, use_gateway=use_gateway)
    if primary_result.get("ok"):
        primary_result.setdefault("reason", plan["reason"])
        primary_result["route"] = "gb10_primary"
        if DEFAULT_GB10_QWEN_CLEANUP and plan.get("cleanup") == "gb10_qwen_ocr" and _needs_semantic_cleanup(goal):
            if not _result_is_usable(primary_result, min_conf=0.72, min_chars=120):
                cleanup = _gb10_qwen_ocr(
                    path,
                    goal=goal,
                    max_chars=max_chars,
                    max_pages=max_pages,
                    draft_text=str(primary_result.get("text") or ""),
                )
                if cleanup.get("ok"):
                    cleanup["engine"] = f"{primary}+gb10_qwen_cleanup"
                    cleanup["primary_engine"] = primary
                    cleanup["route"] = "gb10_combo"
                    cleanup["reason"] = f"{plan['reason']}_qwen_cleanup"
                    return cleanup
        return primary_result
    if primary != "gb10_qwen_ocr":
        fallback = _run_gb10_engine(path, "gb10_qwen_ocr", goal, max_chars=max_chars, max_pages=max_pages, route_mode=route_mode, use_gateway=use_gateway)
        if fallback.get("ok"):
            fallback["engine"] = "gb10_qwen_ocr"
            fallback["route"] = "gb10_fallback"
        fallback["reason"] = f"{plan['reason']}_primary_failed"
        fallback["primary_error"] = primary_result.get("error")
        return fallback
    return primary_result


def _local_image_best_extract(
    path: Path,
    goal: str,
    *,
    max_chars: int,
    max_pages: int,
    route_mode: str,
    use_gateway: bool,
) -> Dict[str, Any]:
    started = time.perf_counter()
    candidate_attempts: list[Dict[str, Any]] = []
    best: Dict[str, Any] = {}
    best_score = -1.0
    doc_class = _classify_doc_type(path, goal, document_features={})
    fast_accept_applied = False
    fast_accept_reason = "not_evaluated"
    second_local_engine_skipped = False
    selected_preprocess = ""
    for idx, candidate_engine in enumerate(("rapidocr", "tesseract")):
        row = _run_engine_once(
            path,
            candidate_engine,
            goal,
            max_chars=max_chars,
            max_pages=max_pages,
            route_mode=route_mode,
            use_gateway=use_gateway,
        )
        quality = dict(row.get("quality") or {})
        score = float(quality.get("score") or 0.0)
        chars = int(quality.get("chars") or len(str(row.get("text") or "")))
        numeric_fidelity_score = float(quality.get("numeric_fidelity_score") or 0.0)
        structure_score = float(quality.get("structure_score") or 0.0)
        candidate_attempts.append(
            {
                "pass": 1,
                "engine": str(row.get("engine") or candidate_engine),
                "model": str(row.get("model") or ""),
                "ok": bool(row.get("ok")),
                "latency_ms": float(row.get("latency_ms") or 0.0),
                "chars": chars,
                "confidence": row.get("confidence"),
                "score": score,
                "structure_score": structure_score,
                "numeric_fidelity_score": numeric_fidelity_score,
                "fallback_reason": str(row.get("error") or row.get("fallback_reason") or ""),
                "replaced_regions": int(((row.get("phase2") or {}) if isinstance(row.get("phase2"), dict) else {}).get("region_retry_count") or 0),
                "stop_reason": "",
                "selected_preprocess": str(row.get("selected_preprocess") or ""),
                "latency_breakdown": _clone_latency_breakdown(row.get("latency_breakdown")),
                "attempt_role": "primary" if idx == 0 else "fallback_chain",
            }
        )
        if row.get("ok") and score >= best_score:
            best = row
            best_score = score
            selected_preprocess = str(row.get("selected_preprocess") or selected_preprocess or "")
        if idx == 0:
            should_accept, reason = _local_image_fast_accept_decision(
                goal=goal,
                doc_class=doc_class,
                score=score,
                numeric_fidelity_score=numeric_fidelity_score,
                chars=chars,
                structure_score=structure_score,
            )
            fast_accept_reason = reason
            if row.get("ok") and should_accept:
                fast_accept_applied = True
                second_local_engine_skipped = True
                best = row
                best_score = score
                break
    total_ms = round((time.perf_counter() - started) * 1000.0, 2)
    if not best:
        return {
            "ok": False,
            "engine": "local_image_best",
            "error": "local_image_best_failed",
            "attempts": candidate_attempts,
            "latency_ms": total_ms,
            "latency_breakdown": _aggregate_route_latency(candidate_attempts, total_ms=total_ms),
            "route": "local_image_best",
        }
    out = dict(best)
    out["engine"] = "local_image_best"
    out["primary_engine"] = str(best.get("engine") or "")
    out["selected_engine"] = str(best.get("engine") or "")
    out["route"] = "local_image_best"
    out["attempts"] = candidate_attempts
    out["selected_preprocess"] = selected_preprocess or str(best.get("selected_preprocess") or "")
    out["fast_accept_applied"] = bool(fast_accept_applied)
    out["fast_accept_reason"] = str(fast_accept_reason or "")
    out["fast_accept_thresholds"] = dict(_LOCAL_FAST_ACCEPT_THRESHOLDS)
    out["second_local_engine_skipped"] = bool(second_local_engine_skipped)
    out["local_image_candidates"] = [
        {
            "engine": str(item.get("engine") or ""),
            "ok": bool(item.get("ok")),
            "latency_ms": float(item.get("latency_ms") or 0.0),
            "score": float(item.get("score") or 0.0),
            "selected_preprocess": str(item.get("selected_preprocess") or ""),
            "fallback_reason": str(item.get("fallback_reason") or ""),
        }
        for item in candidate_attempts
    ]
    out["latency_ms"] = total_ms
    out["latency_breakdown"] = _aggregate_route_latency(candidate_attempts, total_ms=total_ms)
    out["timing_meta"] = _build_timing_meta(
        out.get("timing_meta"),
        engine_selected="local_image_best",
        selected_preprocess=out["selected_preprocess"],
        phase2_enabled=bool(DEFAULT_OCR_PHASE2_ENABLED),
        self_consistency_used=False,
        region_retry_count=0,
        engine_chain_length=2,
    )
    out["timing_meta"]["fast_accept_applied"] = bool(fast_accept_applied)
    out["timing_meta"]["fast_accept_reason"] = str(fast_accept_reason or "")
    out["timing_meta"]["fast_accept_thresholds"] = dict(_LOCAL_FAST_ACCEPT_THRESHOLDS)
    out["timing_meta"]["second_local_engine_skipped"] = bool(second_local_engine_skipped)
    return out


def _ocr_pdf_local(
    path: Path,
    engine: str,
    goal: str,
    max_pages: int,
    max_chars: int,
) -> Dict[str, Any]:
    try:
        overall_start = time.perf_counter()
        render_started = time.perf_counter()
        page_paths = _render_pdf_pages(path, max_pages)
        render_ms = round((time.perf_counter() - render_started) * 1000.0, 2)
    except Exception as exc:
        return {"ok": False, "error": str(exc)}

    texts: list[str] = []
    confidences: list[float] = []
    confidence_regions: list[Dict[str, Any]] = []
    chosen_engine = engine
    reason = ""
    selected_preprocess = ""
    aggregate_breakdown = _empty_latency_breakdown()
    selection_probe_overhead_ms = 0.0

    if engine == "auto":
        preferred, reason = _select_engine(goal, path)
        chosen_engine = preferred
        if preferred == "tesseract" and page_paths:
            first = _tesseract_ocr(page_paths[0], max_chars=max_chars)
            selection_probe_overhead_ms += _latency_breakdown_total(first.get("latency_breakdown")) or _coerce_ms(first.get("latency_ms"))
            if not first.get("ok") or first.get("confidence", 0.0) < 0.55 or len(first.get("text", "")) < 40:
                chosen_engine = "rapidocr"
                reason = "tesseract_low_confidence"

    for page_index, page_path in enumerate(page_paths):
        if chosen_engine == "rapidocr":
            res = _rapidocr_ocr(page_path, max_chars=max_chars)
        else:
            res = _tesseract_ocr(page_path, max_chars=max_chars)

        if res.get("ok"):
            texts.append(res.get("text", ""))
            conf = res.get("confidence")
            if conf is not None:
                confidences.append(float(conf))
            selected_preprocess = selected_preprocess or str(res.get("selected_preprocess") or "")
            aggregate_breakdown = _merge_latency_breakdowns(aggregate_breakdown, res.get("latency_breakdown"))
            for region in list(res.get("confidence_map_regions") or []):
                row = dict(region or {})
                row["page"] = page_index
                row["source_engine"] = str(row.get("source_engine") or chosen_engine)
                row["conf"] = _norm_region_conf(row.get("conf"))
                confidence_regions.append(row)
        try:
            page_path.unlink()
        except Exception:
            pass

    markdown = _normalize_markdown_layout("\n\n".join(t for t in texts if t), max_chars=max_chars)
    text = _normalize_text(markdown, max_chars=max_chars)
    avg_conf = sum(confidences) / len(confidences) if confidences else None
    total_ms = round((time.perf_counter() - overall_start) * 1000.0, 2)
    return {
        "ok": bool(text),
        "engine": chosen_engine,
        "reason": reason or "pdf_ocr",
        "pages": len(page_paths),
        "text": text,
        "markdown": markdown,
        "confidence": round(avg_conf, 3) if avg_conf is not None else None,
        "confidence_map_regions": sorted(confidence_regions, key=lambda row: float(row.get("conf", 1.0)))[
            : DEFAULT_OCR_PHASE2_REGION_RETRY_MAX
        ],
        "selected_preprocess": selected_preprocess or "pdf_render",
        "latency_breakdown": _finalize_latency_breakdown(
            {
                "model_load_overhead_ms": aggregate_breakdown.get("model_load_overhead_ms"),
                "preprocessing_overhead_ms": round(render_ms + aggregate_breakdown.get("preprocessing_overhead_ms", 0.0), 2),
                "ocr_engine_time_ms": aggregate_breakdown.get("ocr_engine_time_ms"),
                "fallback_or_chaining_overhead_ms": round(
                    selection_probe_overhead_ms + aggregate_breakdown.get("fallback_or_chaining_overhead_ms", 0.0),
                    2,
                ),
            },
            total_ms=total_ms,
        ),
        "timing_meta": _build_timing_meta(
            engine_selected=chosen_engine,
            selected_preprocess=selected_preprocess or "pdf_render",
            phase2_enabled=bool(DEFAULT_OCR_PHASE2_ENABLED),
            self_consistency_used=bool(aggregate_breakdown.get("fallback_or_chaining_overhead_ms", 0.0)),
            region_retry_count=0,
            engine_chain_length=1,
        ),
        "route": "local",
    }


def _aggregate_route_latency(attempts: list[Dict[str, Any]], *, total_ms: float) -> Dict[str, float]:
    aggregated = _empty_latency_breakdown()
    primary_consumed = False
    for attempt in attempts:
        if not isinstance(attempt, dict):
            continue
        attempt_total = _coerce_ms(attempt.get("latency_ms"))
        attempt_breakdown = _clone_latency_breakdown(attempt.get("latency_breakdown"))
        role = str(attempt.get("attempt_role") or "").strip().lower()
        if role == "primary" and not primary_consumed:
            primary_consumed = True
            aggregated["model_load_overhead_ms"] = round(
                aggregated["model_load_overhead_ms"] + attempt_breakdown.get("model_load_overhead_ms", 0.0), 2
            )
            aggregated["preprocessing_overhead_ms"] = round(
                aggregated["preprocessing_overhead_ms"] + attempt_breakdown.get("preprocessing_overhead_ms", 0.0), 2
            )
            aggregated["ocr_engine_time_ms"] = round(
                aggregated["ocr_engine_time_ms"] + attempt_breakdown.get("ocr_engine_time_ms", 0.0), 2
            )
            base_classified = round(
                attempt_breakdown.get("model_load_overhead_ms", 0.0)
                + attempt_breakdown.get("preprocessing_overhead_ms", 0.0)
                + attempt_breakdown.get("ocr_engine_time_ms", 0.0)
                + attempt_breakdown.get("fallback_or_chaining_overhead_ms", 0.0),
                2,
            )
            overflow = round(max(0.0, attempt_total - base_classified), 2)
            if attempt_breakdown.get("fallback_or_chaining_overhead_ms", 0.0) or overflow:
                aggregated["fallback_or_chaining_overhead_ms"] = round(
                    aggregated["fallback_or_chaining_overhead_ms"]
                    + attempt_breakdown.get("fallback_or_chaining_overhead_ms", 0.0)
                    + overflow,
                    2,
                )
            continue
        if attempt_total <= 0.0:
            attempt_total = _latency_breakdown_total(attempt_breakdown)
        aggregated["fallback_or_chaining_overhead_ms"] = round(
            aggregated["fallback_or_chaining_overhead_ms"] + attempt_total,
            2,
        )
    return _finalize_latency_breakdown(aggregated, total_ms=total_ms)


def _run_policy_route(
    path: Path,
    *,
    goal: str,
    route_mode: str,
    max_chars: int,
    max_pages: int,
    forced_engine: str = "",
    gb10_enabled: bool = True,
    consensus: bool = False,
    use_gateway: bool = DEFAULT_GB10_OCR_USE_GATEWAY,
    region_retry_policy: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    route_started = time.perf_counter()
    document_features = ocr_local_inference.classify_document_features(path, goal=goal, max_pages=max_pages)
    visual_controls = list(document_features.get("visual_controls") or [])
    feature_detection = {
        "mode": str(document_features.get("mode") or "lightweight_heuristic"),
        "ok": bool(document_features.get("ok")),
        "confidence_reason": str(document_features.get("confidence_reason") or ""),
        "warnings": list(document_features.get("warnings") or []),
    }
    doc_class = _classify_doc_type(path, goal, document_features=document_features)
    chain = _policy_engine_chain(doc_class, route_mode, forced_engine=forced_engine, document_features=document_features)
    if not gb10_enabled:
        chain = [engine for engine in chain if not engine_is_optional(engine)]
        if not chain:
            chain = ["rapidocr", "tesseract"]
    attempts: list[Dict[str, Any]] = []
    best: Dict[str, Any] = {}
    best_score = -1.0
    candidate_rows: list[Dict[str, Any]] = []
    fallback_reason = ""
    target_score = 0.72 if route_mode != "balanced" else 0.60
    max_passes = DEFAULT_OCR_PHASE2_MAX_PASSES if DEFAULT_OCR_PHASE2_ENABLED else 1
    improvement = 1.0
    passes_run = 0
    phase2_stop_reason = ""
    region_retry_fragments: list[Dict[str, Any]] = []
    retry_policy = _parse_region_retry_policy(region_retry_policy)
    primary_attempt_emitted = False
    local_router_accepted = False
    for pass_idx in range(max_passes):
        passes_run = pass_idx + 1
        pass_best = best_score
        for engine in chain:
            row = _run_engine_once(path, engine, goal, max_chars=max_chars, max_pages=max_pages, route_mode=route_mode, use_gateway=use_gateway)
            quality = dict(row.get("quality") or {})
            row_phase2 = dict(row.get("phase2") or {})
            row_breakdown = _clone_latency_breakdown(row.get("latency_breakdown"))
            primary_candidate = (
                not primary_attempt_emitted
                and (
                    bool(row.get("ok"))
                    or (
                        row_breakdown.get("model_load_overhead_ms", 0.0)
                        + row_breakdown.get("preprocessing_overhead_ms", 0.0)
                        + row_breakdown.get("ocr_engine_time_ms", 0.0)
                    )
                    > 0.0
                    or _coerce_ms(row.get("latency_ms")) >= 5.0
                )
            )
            attempt_role = "primary" if primary_candidate else "fallback_chain"
            attempt = {
                "pass": pass_idx + 1,
                "engine": str(row.get("engine") or engine),
                "model": str(row.get("model") or ""),
                "ok": bool(row.get("ok")),
                "latency_ms": float(row.get("latency_ms") or 0.0),
                "chars": int(quality.get("chars") or len(str(row.get("text") or ""))),
                "confidence": row.get("confidence"),
                "score": float(quality.get("score") or 0.0),
                "structure_score": float(quality.get("structure_score") or 0.0),
                "numeric_fidelity_score": float(quality.get("numeric_fidelity_score") or 0.0),
                "fallback_reason": str(row.get("error") or row.get("fallback_reason") or ""),
                "replaced_regions": int(row_phase2.get("region_retry_count") or 0),
                "stop_reason": str(row_phase2.get("stop_reason") or ""),
                "selected_preprocess": str(row.get("selected_preprocess") or ""),
                "latency_breakdown": row_breakdown,
                "attempt_role": attempt_role,
            }
            if attempt_role == "primary":
                primary_attempt_emitted = True
            attempts.append(attempt)
            score = float(quality.get("score") or 0.0)
            if row.get("ok"):
                candidate_rows.append(row)
            if row.get("ok") and score > best_score:
                best = row
                best_score = score
            if str(row.get("engine") or engine) == "local_image_best" and row.get("ok"):
                best = row
                best_score = score
                local_router_accepted = True
                phase2_stop_reason = "local_image_router_accepted"
                break
            if row.get("ok") and score >= target_score and float(quality.get("numeric_fidelity_score") or 0.0) >= 0.50:
                best = row
                best_score = score
                break
            if not row.get("ok") and not fallback_reason:
                fallback_reason = str(row.get("error") or "")
        if local_router_accepted:
            break
        if best and best_score >= target_score and float((best.get("quality") or {}).get("numeric_fidelity_score") or 0.0) >= 0.50:
            break
        improvement = best_score - pass_best
        if pass_idx >= 1 and improvement < 0.015:
            phase2_stop_reason = "stopping_criteria_small_improvement"
            attempts.append(
                {
                    "pass": pass_idx + 1,
                    "engine": "phase2_stop",
                    "ok": True,
                    "score": round(best_score, 4) if best_score >= 0 else 0.0,
                    "fallback_reason": phase2_stop_reason,
                    "replaced_regions": 0,
                    "stop_reason": phase2_stop_reason,
                    "attempt_role": "phase2_stop",
                    "latency_breakdown": _empty_latency_breakdown(),
                }
            )
            break

    ensemble_variant_rows: list[Dict[str, Any]] = []
    if DEFAULT_OCR_PHASE2_ENABLED and consensus:
        ensemble_seed_engine = _phase2_base_engine_name(forced_engine)
        if ensemble_seed_engine in {"", "auto", "gb10_auto", "native_pdf_text"}:
            ensemble_seed_engine = _phase2_base_engine_name(best.get("primary_engine") or best.get("engine"))
        if not ensemble_seed_engine or ensemble_seed_engine == "native_pdf_text":
            for row in candidate_rows:
                candidate_engine = _phase2_base_engine_name(row.get("primary_engine") or row.get("engine"))
                if candidate_engine and candidate_engine != "native_pdf_text":
                    ensemble_seed_engine = candidate_engine
                    break
        if not ensemble_seed_engine or ensemble_seed_engine == "native_pdf_text":
            for engine_name in chain:
                candidate_engine = _phase2_base_engine_name(engine_name)
                if candidate_engine and candidate_engine != "native_pdf_text":
                    ensemble_seed_engine = candidate_engine
                    break
        ensemble_payload: Dict[str, Any] = {}
        if ensemble_seed_engine:
            if path.suffix.lower() == ".pdf":
                ensemble_payload = _phase2_pdf_dpi_ensemble(
                    path,
                    ensemble_seed_engine,
                    goal,
                    max_chars=max_chars,
                    max_pages=max_pages,
                    route_mode=route_mode,
                    use_gateway=use_gateway,
                )
            else:
                ensemble_payload = _phase2_image_scale_ensemble(
                    path,
                    ensemble_seed_engine,
                    goal,
                    max_chars=max_chars,
                    max_pages=max_pages,
                    route_mode=route_mode,
                    use_gateway=use_gateway,
                )
        attempts.extend(list(ensemble_payload.get("attempts") or []))
        ensemble_variant_rows = list(ensemble_payload.get("candidates") or [])
        candidate_rows.extend(ensemble_variant_rows)
        voted = dict(ensemble_payload.get("voted") or {})
        vote_score = float((voted.get("quality") or {}).get("score") or 0.0)
        if voted.get("ok") and vote_score >= best_score:
            best = voted
            best_score = vote_score

    if DEFAULT_OCR_PHASE2_ENABLED and consensus and len(candidate_rows) >= 2:
        consensus_candidates = sorted(
            candidate_rows,
            key=lambda row: float((row.get("quality") or {}).get("score") or 0.0),
            reverse=True,
        )[:5]
        voted = _line_vote_majority(consensus_candidates, max_chars=max_chars)
        vote_quality = dict(voted.get("quality") or {})
        attempts.append(
            {
                "pass": passes_run,
                "engine": "phase2_ensemble_vote",
                "ok": bool(voted.get("ok")),
                "latency_ms": 0.0,
                "chars": int(vote_quality.get("chars") or 0),
                "confidence": vote_quality.get("confidence"),
                "score": float(vote_quality.get("score") or 0.0),
                "structure_score": float(vote_quality.get("structure_score") or 0.0),
                "numeric_fidelity_score": float(vote_quality.get("numeric_fidelity_score") or 0.0),
                "fallback_reason": str(voted.get("error") or ""),
                "replaced_regions": 0,
                "stop_reason": "",
                "attempt_role": "ensemble_vote",
                "latency_breakdown": _empty_latency_breakdown(),
            }
        )
        vote_score = float(vote_quality.get("score") or 0.0)
        if voted.get("ok") and vote_score >= best_score:
            best = voted
            best_score = vote_score

    cleanup_allowed, cleanup_reason = _should_run_semantic_cleanup(
        best,
        goal=goal,
        route_mode=route_mode,
        gb10_enabled=gb10_enabled,
        doc_class=doc_class,
    ) if best else (False, "no_best_result")
    if best and cleanup_allowed:
        best_quality = dict(best.get("quality") or {})
        if float(best_quality.get("structure_score") or 0.0) < 0.55:
            cleanup = _run_engine_once(path, "gb10_qwen_ocr", goal, max_chars=max_chars, max_pages=max_pages, route_mode=route_mode, use_gateway=use_gateway)
            cleanup_quality = dict(cleanup.get("quality") or {})
            attempts.append(
                {
                    "pass": passes_run,
                    "engine": str(cleanup.get("engine") or "gb10_qwen_ocr"),
                    "model": str(cleanup.get("model") or ""),
                    "ok": bool(cleanup.get("ok")),
                    "latency_ms": float(cleanup.get("latency_ms") or 0.0),
                    "chars": int(cleanup_quality.get("chars") or len(str(cleanup.get("text") or ""))),
                    "confidence": cleanup.get("confidence"),
                    "score": float(cleanup_quality.get("score") or 0.0),
                    "structure_score": float(cleanup_quality.get("structure_score") or 0.0),
                    "numeric_fidelity_score": float(cleanup_quality.get("numeric_fidelity_score") or 0.0),
                    "fallback_reason": "consensus_cleanup",
                    "replaced_regions": 0,
                    "stop_reason": "",
                    "selected_preprocess": str(cleanup.get("selected_preprocess") or ""),
                    "latency_breakdown": _clone_latency_breakdown(cleanup.get("latency_breakdown")),
                    "attempt_role": "cleanup",
                }
            )
            if cleanup.get("ok") and float(cleanup_quality.get("score") or 0.0) >= best_score:
                cleanup["fallback_reason"] = "consensus_cleanup"
                best = cleanup
                best_score = float(cleanup_quality.get("score") or 0.0)
    elif best:
        attempts.append(
            {
                "pass": passes_run,
                "engine": "cleanup_skipped",
                "ok": True,
                "latency_ms": 0.0,
                "chars": int(((best.get("quality") or {}) if isinstance(best.get("quality"), dict) else {}).get("chars") or 0),
                "confidence": best.get("confidence"),
                "score": float(((best.get("quality") or {}) if isinstance(best.get("quality"), dict) else {}).get("score") or 0.0),
                "structure_score": float(((best.get("quality") or {}) if isinstance(best.get("quality"), dict) else {}).get("structure_score") or 0.0),
                "numeric_fidelity_score": float(((best.get("quality") or {}) if isinstance(best.get("quality"), dict) else {}).get("numeric_fidelity_score") or 0.0),
                "fallback_reason": cleanup_reason,
                "replaced_regions": 0,
                "stop_reason": cleanup_reason,
                "selected_preprocess": str(best.get("selected_preprocess") or ""),
                "latency_breakdown": _empty_latency_breakdown(),
                "attempt_role": "cleanup_skipped",
            }
        )

    if (
        DEFAULT_OCR_PHASE2_ENABLED
        and best
        and path.suffix.lower() != ".pdf"
        and float((best.get("quality") or {}).get("score") or 0.0) < target_score
    ):
        region_source: Dict[str, Any] = {}
        region_priority = ["tesseract", "rapidocr", str(best.get("engine") or "").strip().lower()]
        for engine_name in region_priority:
            region_source = next(
                (
                    row
                    for row in candidate_rows
                    if str(row.get("engine") or "").strip().lower() == engine_name and list(row.get("confidence_map_regions") or [])
                ),
                {},
            )
            if region_source:
                break

        regions = list(region_source.get("confidence_map_regions") or [])
        if not regions:
            regions = _companion_tesseract_regions(
                path,
                max_regions=max(1, int(retry_policy.get("max_regions") or DEFAULT_OCR_PHASE2_REGION_RETRY_MAX)),
                conf_threshold=float(retry_policy.get("conf_threshold") or DEFAULT_OCR_PHASE2_REGION_RETRY_CONF_THRESHOLD),
            )
        conf_threshold = float(retry_policy.get("conf_threshold") or DEFAULT_OCR_PHASE2_REGION_RETRY_CONF_THRESHOLD)
        regions = [
            row
            for row in sorted(regions, key=lambda item: float(_norm_region_conf((item or {}).get("conf"))))
            if _norm_region_conf((row or {}).get("conf")) < conf_threshold
        ]
        max_retry_regions = max(0, int(retry_policy.get("max_regions") or DEFAULT_OCR_PHASE2_REGION_RETRY_MAX))
        crop_paths = _crop_regions_for_retry(path, regions, max_regions=max_retry_regions)
        try:
            retry_order = list(retry_policy.get("retry_order") or ["gb10_qwen_ocr", "rapidocr"])
            if not gb10_enabled:
                retry_order = [name for name in retry_order if name != "gb10_qwen_ocr"]
            if not retry_order:
                retry_order = ["rapidocr"]
            for region_info, crop_path in zip(regions, crop_paths):
                selected_retry: Dict[str, Any] = {}
                selected_engine = ""
                for region_engine in retry_order:
                    retry_row = _run_engine_once(
                        crop_path,
                        region_engine,
                        goal,
                        max_chars=max_chars // 2,
                        max_pages=1,
                        route_mode=route_mode,
                        use_gateway=use_gateway,
                    )
                    retry_quality = dict(retry_row.get("quality") or {})
                    attempts.append(
                        {
                            "pass": passes_run,
                            "engine": f"phase2_region_retry:{region_engine}",
                            "ok": bool(retry_row.get("ok")),
                            "latency_ms": float(retry_row.get("latency_ms") or 0.0),
                            "chars": int(retry_quality.get("chars") or 0),
                            "confidence": retry_row.get("confidence"),
                            "score": float(retry_quality.get("score") or 0.0),
                            "structure_score": float(retry_quality.get("structure_score") or 0.0),
                            "numeric_fidelity_score": float(retry_quality.get("numeric_fidelity_score") or 0.0),
                            "fallback_reason": str(retry_row.get("error") or ""),
                            "replaced_regions": 0,
                            "stop_reason": "",
                            "selected_preprocess": str(retry_row.get("selected_preprocess") or ""),
                            "latency_breakdown": _clone_latency_breakdown(retry_row.get("latency_breakdown")),
                            "attempt_role": "region_retry",
                        }
                    )
                    if retry_row.get("ok"):
                        selected_retry = retry_row
                        selected_engine = region_engine
                        break
                if selected_retry:
                    region_retry_fragments.append(
                        {
                            "bbox": {key: int(region_info.get(key) or 0) for key in ("x", "y", "w", "h")},
                            "source_text": str(region_info.get("text") or "").strip(),
                            "replacement_text": str(selected_retry.get("markdown") or selected_retry.get("text") or "").strip(),
                            "source_engine": str(region_info.get("source_engine") or ""),
                            "retry_engine": selected_engine,
                            "source_conf": _norm_region_conf(region_info.get("conf")),
                        }
                    )
            if region_retry_fragments:
                merged_payload = _merge_region_retry_fragments(
                    str(best.get("markdown") or best.get("text") or ""),
                    region_retry_fragments,
                    max_chars=max_chars,
                )
                merged_quality = _quality_bundle(str(merged_payload.get("markdown") or ""), best.get("confidence"))
                attempts.append(
                    {
                        "pass": passes_run,
                        "engine": f"{best.get('engine')}+phase2_region_retry_merge",
                        "ok": True,
                        "latency_ms": 0.0,
                        "chars": int(merged_quality.get("chars") or 0),
                        "confidence": best.get("confidence"),
                        "score": float(merged_quality.get("score") or 0.0),
                        "structure_score": float(merged_quality.get("structure_score") or 0.0),
                        "numeric_fidelity_score": float(merged_quality.get("numeric_fidelity_score") or 0.0),
                        "fallback_reason": "confidence_map_region_retry",
                        "replaced_regions": len(list(merged_payload.get("replaced_regions") or [])),
                        "stop_reason": "",
                        "attempt_role": "region_retry_merge",
                        "latency_breakdown": _empty_latency_breakdown(),
                    }
                )
                if float(merged_quality.get("score") or 0.0) > best_score + 0.01:
                    replaced_regions = list(merged_payload.get("replaced_regions") or [])
                    best_phase2 = dict(best.get("phase2") or {})
                    best = {
                        **best,
                        "engine": f"{best.get('engine')}+phase2_region_retry",
                        "markdown": str(merged_payload.get("markdown") or ""),
                        "text": _normalize_text(str(merged_payload.get("markdown") or ""), max_chars=max_chars),
                        "quality": merged_quality,
                        "fallback_reason": "confidence_map_region_retry",
                        "phase2": {
                            **best_phase2,
                            "region_retry_count": len(replaced_regions),
                            "replaced_regions": replaced_regions,
                        },
                    }
                    best_score = float(merged_quality.get("score") or 0.0)
        finally:
            for tmp in crop_paths:
                try:
                    tmp.unlink()
                except Exception:
                    pass

    if not best:
        return _normalize_ocr_payload(
            {"ok": False, "engine": "", "error": "all_engines_failed"},
            route_mode=route_mode,
            doc_class=doc_class,
            engine_chain=chain,
            attempts=attempts,
            fallback_reason=fallback_reason or "all_engines_failed",
        )

    best_phase2 = dict(best.get("phase2") or {})
    self_consistency_variants = [row for row in candidate_rows if str((row.get("phase2") or {}).get("self_consistency_variant") or "").strip()]
    dpi_samples = sorted(
        {
            int((row.get("phase2") or {}).get("dpi"))
            for row in self_consistency_variants
            if (row.get("phase2") or {}).get("dpi") is not None
        }
    )
    scale_samples = sorted(
        {
            float((row.get("phase2") or {}).get("scale"))
            for row in self_consistency_variants
            if (row.get("phase2") or {}).get("scale") is not None
        }
    )
    best["phase2"] = {
        **best_phase2,
        "enabled": bool(DEFAULT_OCR_PHASE2_ENABLED),
        "passes": int(passes_run),
        "attempt_count": len(attempts),
        "vote_mode": str(best_phase2.get("vote_mode") or ("line_majority_vote" if str(best.get("engine") or "") == "phase2_ensemble_vote" else "best_score")),
        "self_consistency_used": bool(
            best_phase2.get("self_consistency_used")
            or any(bool((row.get("phase2") or {}).get("self_consistency_used")) for row in candidate_rows)
            or bool(self_consistency_variants)
        ),
        "variant_count": int(best_phase2.get("variant_count") or len(self_consistency_variants)),
        "dpi_samples": list(best_phase2.get("dpi_samples") or dpi_samples),
        "scale_samples": list(best_phase2.get("scale_samples") or scale_samples),
        "region_retry_count": int(best_phase2.get("region_retry_count") or 0),
        "replaced_regions": list(best_phase2.get("replaced_regions") or []),
        "region_retry_policy": {
            "conf_threshold": float(retry_policy.get("conf_threshold") or DEFAULT_OCR_PHASE2_REGION_RETRY_CONF_THRESHOLD),
            "max_regions": int(retry_policy.get("max_regions") or DEFAULT_OCR_PHASE2_REGION_RETRY_MAX),
            "retry_order": list(retry_policy.get("retry_order") or ["gb10_qwen_ocr", "rapidocr"]),
        },
        "stop_reason": str(best_phase2.get("stop_reason") or phase2_stop_reason),
        "contributing_engines": list(
            best_phase2.get("contributing_engines")
            or ([str(best.get("engine") or "")] if str(best.get("engine") or "") else [])
        ),
    }
    route_elapsed_ms = round((time.perf_counter() - route_started) * 1000.0, 2)
    best["latency_ms"] = route_elapsed_ms
    best["latency_breakdown"] = _aggregate_route_latency(attempts, total_ms=route_elapsed_ms)
    best["selected_preprocess"] = str(best.get("selected_preprocess") or "")
    best["timing_meta"] = _build_timing_meta(
        best.get("timing_meta"),
        engine_selected=str(best.get("engine") or ""),
        selected_preprocess=str(best.get("selected_preprocess") or ""),
        phase2_enabled=bool(DEFAULT_OCR_PHASE2_ENABLED),
        self_consistency_used=bool(best["phase2"].get("self_consistency_used")),
        region_retry_count=int(best["phase2"].get("region_retry_count") or 0),
        engine_chain_length=len(chain),
    )

    return _normalize_ocr_payload(
        best,
        route_mode=route_mode,
        doc_class=doc_class,
        engine_chain=chain,
        attempts=attempts,
        fallback_reason=fallback_reason or str(best.get("fallback_reason") or ""),
        document_features=document_features,
        visual_controls=visual_controls,
        feature_detection=feature_detection,
    )


def ocr_dual(payload: Dict[str, Any]) -> Dict[str, Any]:
    overall_started = time.perf_counter()
    path_str = (payload.get("path") or "").strip()
    if not path_str:
        return {"ok": False, "error": "path_required"}

    path = Path(path_str)
    if not path.exists():
        return {"ok": False, "error": f"file_not_found:{path_str}"}

    engine = normalize_engine_name(str(payload.get("engine") or "auto").strip().lower())
    goal = (payload.get("goal") or "").strip()
    max_chars = int(payload.get("max_chars", 4000))
    max_pages = int(payload.get("max_pages", 2))
    route_mode = str(payload.get("route_mode") or DEFAULT_OCR_ROUTE_MODE).strip().lower() or DEFAULT_OCR_ROUTE_MODE
    consensus = _truthy(payload.get("consensus"), default=(route_mode != "balanced"))
    use_gateway = _truthy(payload.get("use_gateway"), default=DEFAULT_GB10_OCR_USE_GATEWAY)
    region_retry_policy = dict(payload.get("region_retry_policy") or {}) if isinstance(payload.get("region_retry_policy"), dict) else {}
    document_features = ocr_local_inference.classify_document_features(path, goal=goal, max_pages=max_pages)
    visual_controls = list(document_features.get("visual_controls") or [])
    feature_detection = {
        "mode": str(document_features.get("mode") or "lightweight_heuristic"),
        "ok": bool(document_features.get("ok")),
        "confidence_reason": str(document_features.get("confidence_reason") or ""),
        "warnings": list(document_features.get("warnings") or []),
    }
    doc_class = _classify_doc_type(path, goal, document_features=document_features)
    gb10_enabled = _gb10_enabled(payload)

    gb10_aliases = {"gb10_auto", "gb10_paddleocr_vl", "gb10_got_ocr2", "gb10_qwen_ocr"}

    gb10_attempt: Dict[str, Any] = {}
    if engine in gb10_aliases or (engine == "auto" and gb10_enabled):
        forced_engine = "" if engine == "auto" or engine == "gb10_auto" else engine
        gb10_attempt = _run_gb10_primary(path, goal, max_chars=max_chars, max_pages=max_pages, forced_engine=forced_engine, route_mode=route_mode, use_gateway=use_gateway)
        if gb10_attempt.get("ok"):
            gb10_attempt["latency_ms"] = _coerce_ms(gb10_attempt.get("latency_ms")) or round((time.perf_counter() - overall_started) * 1000.0, 2)
            return _normalize_ocr_payload(
                gb10_attempt,
                route_mode=route_mode,
                doc_class=doc_class,
                engine_chain=[str(gb10_attempt.get("engine") or forced_engine or "gb10_auto")],
                attempts=gb10_attempt.get("attempts"),
                fallback_reason=str(gb10_attempt.get("fallback_reason") or ""),
                document_features=document_features,
                visual_controls=visual_controls,
                feature_detection=feature_detection,
            )
        if engine in gb10_aliases and engine not in {"gb10_auto"}:
            failed_payload = {"ok": False, "engine": engine, "error": f"gb10_ocr_failed:{gb10_attempt.get('error', 'unknown')}", "latency_ms": round((time.perf_counter() - overall_started) * 1000.0, 2)}
            return _normalize_ocr_payload(
                failed_payload,
                route_mode=route_mode,
                doc_class=doc_class,
                engine_chain=[engine],
                attempts=gb10_attempt.get("attempts"),
                fallback_reason=str(gb10_attempt.get("error") or ""),
                document_features=document_features,
                visual_controls=visual_controls,
                feature_detection=feature_detection,
            )

    if engine in {"auto", "gb10_auto"}:
        return _run_policy_route(
            path,
            goal=goal,
            route_mode=route_mode,
            max_chars=max_chars,
            max_pages=max_pages,
            gb10_enabled=gb10_enabled,
            consensus=consensus,
            use_gateway=use_gateway,
            region_retry_policy=region_retry_policy,
        )

    if path.suffix.lower() == ".pdf":
        if engine in gb10_aliases and engine != "gb10_auto":
            return {"ok": False, "error": f"gb10_ocr_failed:{_run_gb10_primary(path, goal, max_chars=max_chars, max_pages=max_pages, forced_engine=(engine if engine != 'gb10_auto' else ''), route_mode=route_mode, use_gateway=use_gateway).get('error', 'unknown')}"}
        if max_pages < 1:
            return {"ok": False, "error": "max_pages_invalid"}
        local_pdf = _native_pdf_text_extract(path, max_pages=max_pages, max_chars=max_chars) if engine == "native_pdf_text" else _ocr_pdf_local(path, engine, goal, max_pages, max_chars)
        local_pdf["latency_ms"] = _coerce_ms(local_pdf.get("latency_ms")) or round((time.perf_counter() - overall_started) * 1000.0, 2)
        return _normalize_ocr_payload(
            local_pdf,
            route_mode=route_mode,
            doc_class=doc_class,
            engine_chain=[engine or "auto"],
            document_features=document_features,
            visual_controls=visual_controls,
            feature_detection=feature_detection,
        )

    valid_forced_engines = set(
        dedupe_engine_names(
            [
                "tesseract",
                "rapidocr",
                "native_pdf_text",
                "local_image_best",
                "local_image_preprocessed_best",
                *gb10_aliases,
            ]
        )
    )
    if engine not in {"auto", "gb10_auto", *valid_forced_engines}:
        return {"ok": False, "error": "engine_invalid"}

    if engine == "tesseract":
        result = _tesseract_ocr(path, max_chars=max_chars)
        if result.get("ok"):
            result["reason"] = "forced_tesseract"
        result["latency_ms"] = _coerce_ms(result.get("latency_ms")) or round((time.perf_counter() - overall_started) * 1000.0, 2)
        return _normalize_ocr_payload(result, route_mode=route_mode, doc_class=doc_class, engine_chain=["tesseract"], document_features=document_features, visual_controls=visual_controls, feature_detection=feature_detection)

    if engine == "rapidocr":
        result = _rapidocr_ocr(path, max_chars=max_chars)
        if result.get("ok"):
            result["reason"] = "forced_rapidocr"
        result["latency_ms"] = _coerce_ms(result.get("latency_ms")) or round((time.perf_counter() - overall_started) * 1000.0, 2)
        return _normalize_ocr_payload(result, route_mode=route_mode, doc_class=doc_class, engine_chain=["rapidocr"], document_features=document_features, visual_controls=visual_controls, feature_detection=feature_detection)

    preferred, reason = _select_engine(goal, path)
    if preferred == "rapidocr":
        result = _rapidocr_ocr(path, max_chars=max_chars)
        if result.get("ok"):
            result["reason"] = reason
            result["latency_ms"] = _coerce_ms(result.get("latency_ms")) or round((time.perf_counter() - overall_started) * 1000.0, 2)
            return _normalize_ocr_payload(result, route_mode=route_mode, doc_class=doc_class, engine_chain=["rapidocr", "tesseract"], document_features=document_features, visual_controls=visual_controls, feature_detection=feature_detection)

    result = _tesseract_ocr(path, max_chars=max_chars)
    if result.get("ok"):
        if result.get("confidence", 0.0) >= 0.55 and len(result.get("text", "")) >= 40:
            result["reason"] = reason
            result["latency_ms"] = _coerce_ms(result.get("latency_ms")) or round((time.perf_counter() - overall_started) * 1000.0, 2)
            return _normalize_ocr_payload(result, route_mode=route_mode, doc_class=doc_class, engine_chain=["tesseract", "rapidocr"], document_features=document_features, visual_controls=visual_controls, feature_detection=feature_detection)

    rapid_result = _rapidocr_ocr(path, max_chars=max_chars)
    if rapid_result.get("ok"):
        rapid_result["reason"] = "tesseract_low_confidence"
        rapid_result["latency_ms"] = _coerce_ms(rapid_result.get("latency_ms")) or round((time.perf_counter() - overall_started) * 1000.0, 2)
        return _normalize_ocr_payload(rapid_result, route_mode=route_mode, doc_class=doc_class, engine_chain=["tesseract", "rapidocr"], fallback_reason="tesseract_low_confidence", document_features=document_features, visual_controls=visual_controls, feature_detection=feature_detection)

    if result.get("ok"):
        result["reason"] = "rapidocr_unavailable_fallback"
    result["latency_ms"] = _coerce_ms(result.get("latency_ms")) or round((time.perf_counter() - overall_started) * 1000.0, 2)
    return _normalize_ocr_payload(result, route_mode=route_mode, doc_class=doc_class, engine_chain=["tesseract"], fallback_reason="rapidocr_unavailable_fallback", document_features=document_features, visual_controls=visual_controls, feature_detection=feature_detection)


def ocr_backend_readiness(payload: Dict[str, Any]) -> Dict[str, Any]:
    return gb10_ocr_backend_readiness(payload)


register("ocr.dual", ocr_dual)
register("ocr.backend_readiness", ocr_backend_readiness)

