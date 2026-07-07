from __future__ import annotations

import asyncio
import base64
import concurrent.futures
import hashlib
import importlib
import json
import os
import re
import shutil
import subprocess
import sys
import tempfile
import threading
import time
import types
import uuid
from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone
from io import BytesIO
from pathlib import Path
from typing import Any, Dict, Mapping, Optional

import requests
from flask import jsonify, request

from . import diagnostics as diag
from .legacy_env import apply_legacy_env_aliases
from .paths import ensure_paths
from .profiles import gb10_default_enabled, local_production_enabled
from .field_evidence import field_candidates
from .receipt_fields import append_receipt_fields, receipt_fields_from_candidates

apply_legacy_env_aliases()
_PATHS = ensure_paths()


class _LazyModule:
    def __init__(self, module_name: str):
        self.module_name = module_name
        self._module = None

    def _load(self):
        if self._module is None:
            self._module = importlib.import_module(self.module_name)
        return self._module

    def __getattr__(self, name: str):
        return getattr(self._load(), name)


ocr_local_inference = _LazyModule("ocr97.local_inference")
ocr_dual_tool = _LazyModule("ocr97.dual_tool")

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

torch = None
GotOcr2ForConditionalGeneration = None
GotOcr2Processor = None
AutoImageProcessor = None
AutoModelForSequenceClassification = None
AutoTokenizer = None
TableTransformerForObjectDetection = None
pipeline = None
DocPreprocessor = None
PaddleOCR = None

GOT_TRANSFORMERS_AVAILABLE = bool(importlib.util.find_spec("torch")) and bool(importlib.util.find_spec("transformers"))
PHASE2_TRANSFORMERS_AVAILABLE = GOT_TRANSFORMERS_AVAILABLE
PADDLE_DOCPREPROCESSOR_AVAILABLE = bool(importlib.util.find_spec("paddleocr"))
PADDLE_OCR_AVAILABLE = PADDLE_DOCPREPROCESSOR_AVAILABLE


def _utc_iso() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat().replace("+00:00", "Z")


def _truthy(value: Any, default: bool = False) -> bool:
    if value is None:
        return default
    return str(value).strip().lower() in {"1", "true", "yes", "on"}


def _normalize_space(value: Any) -> str:
    return " ".join(str(value or "").strip().split())


def _auth_required() -> bool:
    token = os.getenv("OCR97_LOCAL_TOKEN", "").strip()
    if not token:
        return False
    return os.getenv("OCR97_ALLOW_ANON", "1") != "1"


def _auth_ok() -> bool:
    token = os.getenv("OCR97_LOCAL_TOKEN", "").strip()
    if not token:
        return True
    header_token = request.headers.get("X-OCR97-Token", "").strip()
    if header_token and header_token == token:
        return True
    auth = (request.headers.get("Authorization") or "").strip()
    if auth.lower().startswith("bearer "):
        return auth[7:].strip() == token
    return False


def _resolve_upload_dir(instance_name: str, upload_dir: Optional[Path]) -> Path:
    if upload_dir is not None:
        return upload_dir
    env_dir = os.getenv("OCR97_OCR_UPLOAD_DIR") or os.getenv("OCR_UPLOAD_DIR")
    if env_dir:
        return Path(env_dir)
    return _PATHS.state_dir / instance_name.lower() / "ocr_uploads"


def _install_metadata_path(instance_name: str, upload_root: Path) -> Path:
    env_path = str(os.getenv("OCR97_INSTALL_METADATA_PATH", "")).strip() or str(os.getenv("OCR97_OCR_INSTALL_METADATA_PATH", "")).strip()
    if env_path:
        return Path(env_path)
    return upload_root.parent / f"{instance_name.lower()}_ocr_install_metadata.json"


def _safe_json_load(path: Path) -> Dict[str, Any]:
    try:
        if path.exists():
            return dict(json.loads(path.read_text(encoding="utf-8")))
    except Exception:
        pass
    return {}


def _safe_json_write(path: Path, payload: Dict[str, Any]) -> None:
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")
    except Exception:
        return


def _hash_dir_metadata(root: Path, *, limit: int = 3000) -> str:
    if not root.exists():
        return ""
    dig = hashlib.sha256()
    seen = 0
    for item in sorted(root.rglob("*")):
        if not item.is_file():
            continue
        try:
            stat = item.stat()
        except Exception:
            continue
        dig.update(str(item.relative_to(root)).encode("utf-8", errors="ignore"))
        dig.update(str(stat.st_size).encode("utf-8"))
        dig.update(str(int(stat.st_mtime)).encode("utf-8"))
        seen += 1
        if seen >= limit:
            break
    return dig.hexdigest()[:24]


def _module_available(module_name: str) -> bool:
    if not module_name:
        return False
    cache = globals().setdefault("_MODULE_AVAIL_CACHE", {})
    if module_name in cache:
        return bool(cache[module_name])
    try:
        available = importlib.util.find_spec(module_name) is not None
        cache[module_name] = available
        return available
    except Exception:
        cache[module_name] = False
        return False


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
    table_terms = ("table", "layout", "statement", "invoice", "form", "multi-column", "earnings")
    photo_terms = ("photo", "camera", "screenshot", "screen")
    if any(term in goal_norm for term in dense_scan_terms):
        return "handwritten"
    if suffix == ".pdf":
        if any(term in goal_norm for term in ("scan", "scanned", "photograph", "warped", "skew")):
            return "scanned_pdf"
        if any(term in goal_norm for term in table_terms):
            return "table_dense"
        return "digital_pdf"
    if any(term in goal_norm for term in photo_terms):
        return "photo"
    if any(term in goal_norm for term in dense_scan_terms):
        return "handwritten"
    return "photo"


def _engine_chain(doc_class: str, route_mode: str, requested_model: str, document_features: Optional[Dict[str, Any]] = None) -> list[str]:
    requested = str(requested_model or "").strip().lower()
    if requested and requested not in {"auto", "gb10_auto"}:
        chain = [requested, "gb10_qwen_ocr", "rapidocr", "tesseract"]
    elif str(route_mode or "quality_first").strip().lower() != "balanced":
        if doc_class == "forms_or_checkboxes":
            chain = ["gb10_qwen_ocr", "gb10_got_ocr2", "rapidocr", "tesseract"]
        elif doc_class == "chart_or_figure":
            chain = ["gb10_qwen_ocr", "gb10_paddleocr_vl", "rapidocr", "tesseract"]
        elif doc_class in {"digital_pdf", "table_dense"}:
            chain = ["native_pdf_text", "gb10_paddleocr_vl", "mineru2_5", "olmocr2", "gb10_qwen_ocr", "rapidocr", "tesseract"]
        elif doc_class in {"scanned_pdf", "handwritten"}:
            chain = ["gb10_got_ocr2", "mineru2_5", "gb10_qwen_ocr", "rapidocr", "tesseract"]
        else:
            chain = ["local_image_best", "gb10_qwen_ocr", "gb10_got_ocr2", "rapidocr", "tesseract"]
    elif doc_class == "forms_or_checkboxes":
        chain = ["gb10_qwen_ocr", "rapidocr", "tesseract"]
    elif doc_class == "chart_or_figure":
        chain = ["gb10_qwen_ocr", "rapidocr"]
    elif doc_class in {"digital_pdf", "table_dense"}:
        chain = ["native_pdf_text", "gb10_paddleocr_vl", "gb10_qwen_ocr", "rapidocr"]
    elif doc_class in {"scanned_pdf", "handwritten"}:
        chain = ["gb10_got_ocr2", "gb10_qwen_ocr", "rapidocr"]
    else:
        chain = ["local_image_best", "gb10_qwen_ocr", "rapidocr"]
    if not gb10_default_enabled():
        chain = [engine for engine in chain if not str(engine).startswith("gb10_")]
        if not chain:
            chain = ["rapidocr", "tesseract"]
    return chain


def _table_rows(text: str) -> int:
    rows = 0
    for line in str(text or "").splitlines():
        if ("|" in line and line.count("|") >= 2) or ("\t" in line and len(line.split("\t")) >= 3):
            rows += 1
    return rows


def _numeric_fidelity_score(text: str) -> float:
    raw = str(text or "")
    chars = max(1, len(raw))
    digits = sum(1 for ch in raw if ch.isdigit())
    finance_markers = 0
    for marker in ("$", "%", "margin", "risk", "position", "entry", "exit", "stop", "rule"):
        if marker in raw.lower():
            finance_markers += 1
    density = min(1.0, (digits / float(chars)) * 15.0)
    marker_bonus = min(1.0, finance_markers / 6.0)
    return round(max(0.0, min(1.0, (density * 0.6) + (marker_bonus * 0.4))), 3)


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
    return round(max(0.0, min(1.0, (multi_line_bonus * 0.35) + (heading_score * 0.35) + (table_score * 0.30))), 3)


def _quality_bundle(text: str, confidence: Any) -> Dict[str, Any]:
    chars = len(str(text or "").strip())
    conf = None
    try:
        if confidence is not None:
            conf = float(confidence)
    except Exception:
        conf = None
    structure_score = _structure_score(text)
    numeric_score = _numeric_fidelity_score(text)
    char_score = min(1.0, chars / 1600.0)
    conf_score = min(1.0, max(0.0, conf)) if conf is not None else 0.65
    score = round(max(0.0, min(1.0, (char_score * 0.35) + (structure_score * 0.35) + (numeric_score * 0.20) + (conf_score * 0.10))), 3)
    return {
        "score": score,
        "chars": chars,
        "confidence": conf,
        "structure_score": structure_score,
        "numeric_fidelity_score": numeric_score,
        "table_rows": _table_rows(text),
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


def _native_pdf_text_extract(path: Path, *, max_pages: int, max_chars: int) -> Dict[str, Any]:
    if not PYMUPDF_AVAILABLE:
        return {"ok": False, "engine": "native_pdf_text", "error": "native_pdf_unavailable:pymupdf_missing"}
    if path.suffix.lower() != ".pdf":
        return {"ok": False, "engine": "native_pdf_text", "error": f"native_pdf_unsupported_suffix:{path.suffix.lower()}"}
    try:
        doc = fitz.open(str(path))
    except Exception as exc:
        return {"ok": False, "engine": "native_pdf_text", "error": f"native_pdf_open_failed:{type(exc).__name__}:{exc}"}
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
    finally:
        try:
            doc.close()
        except Exception:
            pass
    text = "\n".join(parts).strip()[:max_chars]
    if not text:
        return {"ok": False, "engine": "native_pdf_text", "error": "native_pdf_empty"}
    min_chars_per_page = int(os.getenv("OCR97_OCR_NATIVE_PDF_MIN_CHARS_PER_PAGE", "40"))
    if pages_read > 0 and len(text) < pages_read * min_chars_per_page:
        return {"ok": False, "engine": "native_pdf_text", "error": "native_pdf_sparse:likely_scanned"}
    return {
        "ok": True,
        "engine": "native_pdf_text",
        "model": "pymupdf",
        "text": text,
        "markdown": text,
        "pages": pages_read,
        "confidence": None,
        "route": "native_pdf",
        "reason": "native_text:pymupdf",
    }


def _ocr_confusable_text(value: Any) -> str:
    raw = str(value or "").lower()
    return raw.translate(str.maketrans({"1": "l", "0": "o", "5": "s"}))


def _numeric_tokens(text: str) -> list[str]:
    tokens: list[str] = []
    for match in re.finditer(r"\(?-?\$?\d[\d,\s]*(?:\.\d+)?%?\)?", str(text or "")):
        token = match.group(0)
        cleaned = re.sub(r"(?<=\d)[,\s]+(?=\d{3}(?:\D|$))", "", token)
        cleaned = cleaned.replace("$", "").replace(",", "").replace("%", "").replace("(", "").replace(")", "").strip()
        if cleaned and any(ch.isdigit() for ch in cleaned):
            tokens.append(cleaned)
    return tokens


def _numeric_consensus_score(text: str, peers: list[str]) -> float:
    tokens = _numeric_tokens(text)
    if not tokens:
        return 0.0
    peer_counts: Dict[str, int] = {}
    for peer in peers:
        for token in set(_numeric_tokens(peer)):
            peer_counts[token] = peer_counts.get(token, 0) + 1
    hits = sum(1 for token in tokens if peer_counts.get(token, 0) > 0)
    return min(1.0, hits / float(max(1, len(tokens))))


def _table_like_score(text: str) -> float:
    raw = str(text or "")
    pipe_rows = _table_rows(raw)
    if pipe_rows:
        return min(1.0, pipe_rows / 4.0)
    pseudo_rows = 0
    for row in raw.splitlines() or [raw]:
        clean = f" {row.strip()} "
        if re.search(r"\s[|1il]\s+[^|]+?\s[|1il]\s+", clean, flags=re.IGNORECASE):
            pseudo_rows += 1
    if pseudo_rows:
        return min(0.75, pseudo_rows / 4.0)
    pseudo_cells = len(re.findall(r"\s[|1il]\s", raw, flags=re.IGNORECASE))
    return min(0.65, pseudo_cells / 12.0)


def _numeric_confusion_penalty(text: str) -> float:
    raw = str(text or "").lower()
    penalty = sum(raw.count(token) for token in ("tota1", "subtota1", "@", " ee", ",ee")) * 4.0
    penalty += len(re.findall(r"\$\d{1,3},\s+\d{3}", raw)) * 3.0
    penalty += len(re.findall(r"\b(?:va1ue|lial3ilities|liabilit1es|equlty)\b", raw)) * 2.0
    penalty += len(re.findall(r"\$\d[\d,]*\.(?:20|49)\b", raw)) * 0.75
    penalty += len(re.findall(r"\$\S*[Â¢@e]\S*", raw)) * 8.0
    return penalty


def _clean_finance_token_score(text: str) -> float:
    raw = str(text or "")
    money = re.findall(r"\$\d{1,3}(?:,\d{3})*(?:\.\d+)?\b", raw)
    dates = re.findall(r"\b\d{4}-\d{2}-\d{2}\b", raw)
    percents = re.findall(r"\b\d+(?:\.\d+)?%\b", raw)
    return min(24.0, (len(money) * 8.0) + (len(dates) * 5.0) + (len(percents) * 5.0))


_COMMON_FIELD_ALIASES: tuple[tuple[str, tuple[str, ...]], ...] = (
    ("invoice_number", ("invoice number", "invoice")),
    ("subtotal", ("subtotal",)),
    ("tax", ("tax",)),
    ("total", ("total", "amount due", "payment due")),
    ("assets", ("assets",)),
    ("liabilities", ("liabilities",)),
    ("equity", ("equity",)),
    ("opening_balance", ("opening balance",)),
    ("deposits", ("deposits",)),
    ("closing_balance", ("closing balance",)),
    ("cash", ("cash",)),
    ("market_value", ("market value",)),
    ("gross_pay", ("gross pay",)),
    ("deductions", ("deductions",)),
    ("net_pay", ("net pay",)),
    ("agi", ("adjusted gross income",)),
    ("taxable_income", ("taxable income",)),
    ("tax_due", ("tax due",)),
    ("account", ("account",)),
    ("due_date", ("due date", "date")),
    ("revenue", ("revenue",)),
    ("cost", ("cost",)),
    ("margin", ("margin",)),
    ("principal", ("principal",)),
    ("interest_rate", ("interest rate",)),
)

_FIELD_DISPLAY_LABELS: Dict[str, str] = {
    "invoice_number": "Invoice",
    "opening_balance": "Opening Balance",
    "closing_balance": "Closing Balance",
    "market_value": "Market Value",
    "gross_pay": "Gross Pay",
    "net_pay": "Net Pay",
    "taxable_income": "Taxable Income",
    "tax_due": "Tax Due",
    "due_date": "Due Date",
    "interest_rate": "Interest Rate",
}

_FIELD_TYPES: Dict[str, str] = {
    "invoice_number": "text",
    "subtotal": "money",
    "tax": "money",
    "total": "money",
    "assets": "money",
    "liabilities": "money",
    "equity": "money",
    "opening_balance": "money",
    "deposits": "money",
    "closing_balance": "money",
    "cash": "money",
    "market_value": "money",
    "gross_pay": "money",
    "deductions": "money",
    "net_pay": "money",
    "agi": "money",
    "taxable_income": "money",
    "tax_due": "money",
    "account": "text",
    "due_date": "date",
    "revenue": "money",
    "cost": "money",
    "margin": "percent",
    "principal": "money",
    "interest_rate": "percent",
}


def _field_alias_regex(alias: str) -> str:
    parts: list[str] = []
    for ch in str(alias or ""):
        lower = ch.lower()
        if ch.isspace():
            parts.append(r"\s+")
        elif lower in {"l", "i"}:
            parts.append(r"[li1|]")
        elif lower == "o":
            parts.append(r"[o0]")
        elif lower == "s":
            parts.append(r"[s5]")
        else:
            parts.append(re.escape(ch))
    return "".join(parts)


def _clean_consensus_value(value: Any) -> str:
    raw = _normalize_space(value)
    raw = re.sub(r"^(?:[:=\-|{}\[\]()]+|[1ilI|])\s+", "", raw)
    raw = re.sub(r"\s+(?:[1ilI|]|[:=\-|{}\[\]()]+)(?:\s+(?:[1ilI|]|[:=\-|{}\[\]()]+))*$", "", raw)
    raw = re.sub(r"(?<=\d)[,\s]+(?=\d{3}(?:\D|$))", ",", raw)
    return raw.strip(" :|={}[]()")


def _consensus_value_key(value: Any, field_name: str = "") -> str:
    raw = _clean_consensus_value(value).replace(";", ",")
    if str(field_name or "").strip().lower() in {"invoice_number", "account"}:
        return re.sub(r"[^a-z0-9]+", " ", raw.lower()).strip()
    date = re.search(r"\b\d{4}[-/]\d{1,2}[-/]\d{1,2}\b", raw)
    if date:
        return date.group(0).replace("/", "-")
    number = re.search(r"\(?-?\$?\d(?:[\d,\s]*\d)?(?:\.\d+)?%?\)?", raw)
    if number:
        token = re.sub(r"(?<=\d)[,\s]+(?=\d{3}(?:\D|$))", "", number.group(0).replace(";", ","))
        return token.replace("$", "").replace(",", "").replace("%", "").replace("(", "").replace(")", "").replace(" ", "")
    return re.sub(r"[^a-z0-9]+", " ", raw.lower()).strip()


def _display_consensus_value(field_name: str, value: Any) -> str:
    raw = _clean_consensus_value(value).replace(";", ",")
    money = re.findall(r"\$\d{1,3}(?:[,\s]\d{3})*(?:\.\d+)?", raw)
    percent = re.findall(r"\b\d+(?:\.\d+)?%", raw)
    date = re.findall(r"\b\d{4}[-/]\d{1,2}[-/]\d{1,2}\b", raw)
    field = str(field_name or "")
    if date and field in {"due_date"}:
        return date[0].replace("/", "-")
    if percent and field in {"margin", "interest_rate"}:
        return percent[0]
    if money:
        picked = money[-1] if field in {"market_value", "total", "tax_due", "amount_due", "payment_due"} else money[0]
        return re.sub(r"(?<=\d)[,\s]+(?=\d{3}(?:\D|$))", ",", picked)
    return raw


def _extract_field_values(text: str) -> list[Dict[str, str]]:
    raw = _normalize_space(text)
    if not raw:
        return []
    all_aliases = [alias for _field, aliases in _COMMON_FIELD_ALIASES for alias in aliases]
    label_union = "|".join(_field_alias_regex(alias) for alias in sorted(all_aliases, key=len, reverse=True))
    values: list[Dict[str, str]] = []
    for field_name, aliases in _COMMON_FIELD_ALIASES:
        for alias in aliases:
            alias_re = _field_alias_regex(alias)
            pattern = rf"\b{alias_re}\b\s*(?:[:=\|]|\s+[1ilI|]\s+|\s+)?\s*(.+?)(?=\s+(?:{label_union})\b\s*(?:[:=\|]|\s+[1ilI|]\s+|\s+)?|$)"
            match = re.search(pattern, raw, flags=re.IGNORECASE)
            if not match:
                continue
            value = _clean_consensus_value(match.group(1))
            key = _consensus_value_key(value, field_name)
            if value and key:
                values.append({"field": field_name, "alias": alias, "value": value, "value_key": key})
                break
    return values


def _extract_ranked_field_values(text: str) -> list[Dict[str, Any]]:
    values: list[Dict[str, Any]] = []
    for field_name, aliases in _COMMON_FIELD_ALIASES:
        candidates = field_candidates(
            text,
            {
                "name": field_name,
                "aliases": list(aliases),
                "type": _FIELD_TYPES.get(field_name, "text"),
            },
        )
        if not candidates:
            continue
        top = dict(candidates[0])
        value = _clean_consensus_value(top.get("value") or top.get("normalized_value") or "")
        key = _consensus_value_key(value, field_name) or str(top.get("normalized_value") or "")
        if value and key:
            values.append(
                {
                    "field": field_name,
                    "alias": aliases[0],
                    "value": value,
                    "value_key": key,
                    "candidate_confidence": float(top.get("confidence") or 0.0),
                    "candidate_reason": str(top.get("reason") or ""),
                }
            )
    return values


def _field_consensus_from_candidates(candidates: list[Dict[str, Any]]) -> list[Dict[str, Any]]:
    buckets: Dict[tuple[str, str], Dict[str, Any]] = {}
    ok_count = max(1, sum(1 for row in candidates if row.get("ok")))
    for row in candidates:
        if not row.get("ok"):
            continue
        source = {
            "engine": str(row.get("engine") or ""),
            "preprocess": str(row.get("preprocess") or ""),
            "selection_score": float(row.get("_selection_score") or 0.0),
        }
        text = str(row.get("markdown") or row.get("text") or "")
        merged_items: Dict[tuple[str, str], Dict[str, Any]] = {}
        for item in _extract_field_values(text):
            merged_items[(str(item["field"]), str(item["value_key"]))] = {
                **item,
                "candidate_confidence": 0.45,
                "candidate_reason": "legacy consensus parser",
            }
        for item in _extract_ranked_field_values(text):
            merged_items[(str(item["field"]), str(item["value_key"]))] = item
        for item in merged_items.values():
            key = (item["field"], item["value_key"])
            bucket = buckets.setdefault(
                key,
                {
                    "field": item["field"],
                    "value": item["value"],
                    "normalized_value": item["value_key"],
                    "aliases": set(),
                    "sources": [],
                    "best_source_score": 0.0,
                    "best_candidate_confidence": 0.0,
                    "reasons": set(),
                },
            )
            bucket["aliases"].add(item["alias"])
            source_with_evidence = dict(source)
            source_with_evidence["candidate_confidence"] = round(float(item.get("candidate_confidence") or 0.0), 3)
            source_with_evidence["candidate_reason"] = str(item.get("candidate_reason") or "")
            bucket["sources"].append(source_with_evidence)
            bucket["best_source_score"] = max(float(bucket.get("best_source_score") or 0.0), source["selection_score"])
            bucket["best_candidate_confidence"] = max(
                float(bucket.get("best_candidate_confidence") or 0.0),
                float(item.get("candidate_confidence") or 0.0),
            )
            if item.get("candidate_reason"):
                bucket["reasons"].add(str(item.get("candidate_reason")))
    rows: list[Dict[str, Any]] = []
    for bucket in buckets.values():
        support = len(bucket["sources"])
        evidence_bonus = min(0.35, float(bucket.get("best_candidate_confidence") or 0.0) * 0.35)
        confidence = min(1.0, (support / float(ok_count)) + min(0.2, float(bucket["best_source_score"]) / 600.0) + evidence_bonus)
        rows.append(
            {
                "field": str(bucket["field"]),
                "value": str(bucket["value"]),
                "normalized_value": str(bucket["normalized_value"]),
                "confidence": round(confidence, 3),
                "support": support,
                "aliases": sorted(bucket["aliases"]),
                "evidence_reasons": sorted(bucket["reasons"]),
                "sources": bucket["sources"][:6],
            }
        )
    best_by_field: Dict[str, Dict[str, Any]] = {}
    for row in sorted(rows, key=lambda item: (-float(item["confidence"]), -int(item["support"]), str(item["field"]))):
        best_by_field.setdefault(str(row["field"]), row)
    return sorted(best_by_field.values(), key=lambda item: str(item["field"]))


def _append_field_consensus(markdown: str, field_consensus: list[Dict[str, Any]]) -> str:
    base = str(markdown or "").strip()
    rows = []
    for item in field_consensus:
        confidence = float(item.get("confidence") or 0.0)
        support = int(item.get("support") or 0)
        if confidence < 0.5 and support < 2:
            continue
        field_name = str(item.get("field") or "")
        label = _FIELD_DISPLAY_LABELS.get(field_name, field_name.replace("_", " ").title())
        value = _display_consensus_value(field_name, item.get("value"))
        if label and value:
            rows.append(f"{label}: {value}")
    if not rows:
        return base
    appendix = "Field consensus:\n" + "\n".join(rows)
    if appendix in base:
        return base
    return (base + "\n\n" + appendix).strip() if base else appendix


def _local_image_score_components(result: Mapping[str, Any], *, peers: Optional[list[str]] = None) -> Dict[str, float]:
    if not result.get("ok"):
        return {"total": -1.0}
    text = str(result.get("markdown") or result.get("text") or "")
    quality = dict(result.get("quality") or _quality_bundle(text, result.get("confidence")))
    raw = text.lower()
    confusable = _ocr_confusable_text(raw)
    marker_hits = sum(
        1
        for marker in (
            "total",
            "subtotal",
            "balance",
            "assets",
            "liabilities",
            "equity",
            "revenue",
            "margin",
            "principal",
            "interest",
            "payment",
            "invoice",
            "account",
        )
        if marker in raw or marker in confusable
    )
    digit_count = sum(1 for ch in text if ch.isdigit())
    peer_texts = list(peers or [])
    components = {
        "quality": float(quality.get("score") or 0.0) * 100.0,
        "numeric_fidelity": float(quality.get("numeric_fidelity_score") or 0.0) * 30.0,
        "structure": float(quality.get("structure_score") or 0.0) * 15.0,
        "digit_density": min(20.0, digit_count / 3.0),
        "marker_hits": min(20.0, marker_hits * 2.5),
        "table_like": _table_like_score(text) * 12.0,
        "clean_finance_tokens": _clean_finance_token_score(text),
        "numeric_consensus": _numeric_consensus_score(text, peer_texts) * 8.0 if peer_texts else 0.0,
        "engine_prior": 0.0,
        "penalty": _numeric_confusion_penalty(text),
    }
    if str(result.get("engine") or "").strip().lower() == "rapidocr" and components["clean_finance_tokens"] >= 16.0:
        components["engine_prior"] = 8.0 if components["table_like"] > 0 else 3.0
    components["total"] = (
        components["quality"]
        + components["numeric_fidelity"]
        + components["structure"]
        + components["digit_density"]
        + components["marker_hits"]
        + components["table_like"]
        + components["clean_finance_tokens"]
        + components["numeric_consensus"]
        + components["engine_prior"]
        - components["penalty"]
    )
    return {key: round(value, 3) for key, value in components.items()}


def _local_image_candidate_score(result: Mapping[str, Any]) -> float:
    return float(_local_image_score_components(result).get("total") or -1.0)


def _rescore_local_image_candidates(candidates: list[Dict[str, Any]]) -> None:
    all_texts = [str(row.get("markdown") or row.get("text") or "") for row in candidates if row.get("ok")]
    for row in candidates:
        own_text = str(row.get("markdown") or row.get("text") or "")
        peers = [text for text in all_texts if text != own_text]
        components = _local_image_score_components(row, peers=peers)
        row["score_components"] = components
        row["_selection_score"] = round(float(components.get("total") or -1.0), 3)


_STRUCTURED_DOC_KEYWORDS: tuple[tuple[str, tuple[str, ...]], ...] = (
    ("banking", ("opening balance", "closing balance", "deposits", "withdrawals", "account number")),
    ("brokerage", ("shares", "market value", "symbol", "portfolio", "unrealized", "cost basis")),
    ("invoice", ("invoice number", "amount due", "purchase order", "line item", "subtotal")),
    ("payroll", ("gross pay", "net pay", "deductions", "pay period", "federal tax", "fica")),
    ("tax", ("taxable income", "adjusted gross", "filing status", "tax due", "refund amount")),
    ("statement", ("previous balance", "new charges", "minimum payment", "payment due date", "statement date")),
    ("loan", ("principal balance", "interest rate", "monthly payment", "remaining term", "origination")),
    ("table_dense", ("| ---", "|---|", "---+---", ":---:", "-----")),
)


def _classify_content_doc_type(text: str) -> str:
    raw = str(text or "").lower()
    if len(raw) < 20:
        return "photo"
    pipe_rows = sum(1 for line in raw.splitlines() if line.count("|") >= 3)
    if pipe_rows >= 4:
        return "table_dense"
    best_class = "photo"
    best_hits = 0
    for class_name, keywords in _STRUCTURED_DOC_KEYWORDS:
        hits = sum(1 for kw in keywords if kw in raw)
        if hits > best_hits:
            best_hits = hits
            best_class = class_name
    if best_hits == 0:
        finance_markers = sum(1 for m in ("$", "balance", "total", "invoice", "account", "payment") if m in raw)
        if finance_markers >= 3:
            return "table_dense"
    return best_class


def _preprocess_fast_accept_enabled() -> bool:
    return _truthy(os.getenv("OCR97_OCR_PREPROCESS_FAST_ACCEPT"), default=not local_production_enabled())


def _preprocess_fast_accept_threshold() -> float:
    try:
        return max(0.0, min(250.0, float(os.getenv("OCR97_OCR_PREPROCESS_FAST_ACCEPT_SCORE", "92"))))
    except ValueError:
        return 92.0


def _fast_accept_local_image_candidate(candidates: list[Dict[str, Any]]) -> Optional[Dict[str, Any]]:
    if not _preprocess_fast_accept_enabled():
        return None
    ok_candidates = [row for row in candidates if row.get("ok")]
    if not ok_candidates:
        return None
    best = max(ok_candidates, key=lambda item: float(item.get("_selection_score") or -1.0))
    text = str(best.get("markdown") or best.get("text") or "")
    components = dict(best.get("score_components") or {})
    if _looks_like_receipt_candidates(candidates):
        return None
    if len(text.strip()) < 80:
        return None
    if sum(1 for ch in text if ch.isdigit()) < 4:
        return None
    if float(best.get("_selection_score") or -1.0) < _preprocess_fast_accept_threshold():
        return None
    if float(components.get("penalty") or 0.0) > 8.0:
        return None
    content_class = _classify_content_doc_type(text)
    if content_class in {"banking", "brokerage", "table_dense"}:
        if _table_like_score(text) < 0.15:
            return None
    return best


def _local_image_best_extract(
    path: Path,
    *,
    goal: str,
    max_pages: int,
    max_chars: int,
    route_mode: str,
    region_retry_policy: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    candidates: list[Dict[str, Any]] = []
    for engine in ("tesseract", "rapidocr"):
        started = time.perf_counter()
        try:
            row = ocr_dual_tool.ocr_dual(
                {
                    "path": str(path),
                    "goal": goal,
                    "engine": engine,
                    "max_pages": max_pages,
                    "max_chars": max_chars,
                    "route_mode": route_mode,
                    "gb10_enabled": True,
                    "use_gateway": False,
                    "region_retry_policy": dict(region_retry_policy or {}),
                }
            )
        except Exception as exc:
            row = {"ok": False, "engine": engine, "error": f"{engine}_failed:{type(exc).__name__}:{exc}"}
        row = dict(row or {})
        row.setdefault("engine", engine)
        row["latency_ms"] = round((time.perf_counter() - started) * 1000.0, 2)
        if row.get("ok") and not row.get("quality"):
            row["quality"] = _quality_bundle(str(row.get("markdown") or row.get("text") or ""), row.get("confidence"))
        candidates.append(row)

    _rescore_local_image_candidates(candidates)
    best = max(candidates, key=lambda item: float(item.get("_selection_score") or -1.0))
    out = dict(best)
    out["router"] = "local_image_best"
    out["selected_engine"] = str(best.get("engine") or "")
    out["route"] = str(best.get("route") or "local_image_best")
    out["field_consensus"] = _field_consensus_from_candidates(candidates)
    if out["field_consensus"]:
        merged = _append_field_consensus(str(out.get("markdown") or out.get("text") or ""), list(out["field_consensus"]))
        out["markdown"] = merged
        out["text"] = merged
        out["field_consensus_used"] = True
    out["receipt_fields"] = receipt_fields_from_candidates(candidates)
    if out["receipt_fields"]:
        merged = append_receipt_fields(str(out.get("markdown") or out.get("text") or ""), list(out["receipt_fields"]))
        out["markdown"] = merged
        out["text"] = merged
        out["receipt_fields_used"] = True
    out["local_image_candidates"] = [
        {
            "engine": str(row.get("engine") or ""),
            "ok": bool(row.get("ok")),
            "latency_ms": float(row.get("latency_ms") or 0.0),
            "selection_score": float(row.get("_selection_score") or 0.0),
            "score_components": dict(row.get("score_components") or {}),
            "chars": len(str(row.get("markdown") or row.get("text") or "")),
            "error": str(row.get("error") or ""),
        }
        for row in candidates
    ]
    return out


def _estimate_cv2_skew_angle(gray: Any) -> Optional[float]:
    if not CV2_AVAILABLE:
        return None
    try:
        blur = cv2.GaussianBlur(gray, (3, 3), 0)
        thresh = cv2.threshold(blur, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)[1]
        coords = np.column_stack(np.where(thresh < 255))
        if coords.size == 0:
            return None
        angle = float(cv2.minAreaRect(coords)[-1])
        if angle < -45.0:
            angle = -(90.0 + angle)
        else:
            angle = -angle
        if abs(angle) < 0.15 or abs(angle) > 12.0:
            return None
        return round(angle, 2)
    except Exception:
        return None


def _rotate_gray_cv2(gray: Any, angle: float) -> Any:
    h, w = gray.shape[:2]
    center = (w / 2.0, h / 2.0)
    matrix = cv2.getRotationMatrix2D(center, float(angle), 1.0)
    cos = abs(matrix[0, 0])
    sin = abs(matrix[0, 1])
    new_w = int((h * sin) + (w * cos))
    new_h = int((h * cos) + (w * sin))
    matrix[0, 2] += (new_w / 2.0) - center[0]
    matrix[1, 2] += (new_h / 2.0) - center[1]
    return cv2.warpAffine(gray, matrix, (new_w, new_h), flags=cv2.INTER_CUBIC, borderMode=cv2.BORDER_CONSTANT, borderValue=255)


def _cv2_to_pillow_gray(gray: Any) -> Any:
    return Image.fromarray(gray.astype("uint8"), mode="L")


def _preprocessed_image_variants(path: Path, temp_dir: Path) -> list[Dict[str, Any]]:
    variants: list[Dict[str, Any]] = [{"label": "original", "path": path, "detected_angle": None}]
    if not PIL_AVAILABLE:
        return variants
    try:
        from PIL import ImageEnhance, ImageFilter, ImageOps

        image = Image.open(path)
        image.load()
        base = image.convert("L")
        allowed = {
            item.strip().lower()
            for item in str(
                os.getenv(
                    "OCR97_OCR_PREPROCESS_VARIANTS",
                    "autocontrast,contrast_sharp,threshold,upscale_sharp,denoise_median,denoise_blur_threshold,deskew_neg3,deskew_pos3,deskew_cv2,deskew_cv2_threshold,angle_sweep",
                )
            ).split(",")
            if item.strip()
        }
        resampling = getattr(getattr(Image, "Resampling", Image), "LANCZOS", Image.BICUBIC)
        max_candidates = max(2, min(24, int(os.getenv("OCR97_OCR_PREPROCESS_MAX_CANDIDATES", "20"))))

        def _save(label: str, candidate: Any, *, detected_angle: Optional[float] = None) -> None:
            if len(variants) >= max_candidates:
                return
            family = label.split("_", 2)[0] if "_" in label else label
            allowed_by_prefix = any(label.startswith(f"{item}_") for item in allowed)
            if label not in allowed and family not in allowed and not allowed_by_prefix:
                return
            out_path = temp_dir / f"{path.stem}.{label}.png"
            candidate.convert("L").save(out_path)
            variants.append({"label": label, "path": out_path, "detected_angle": detected_angle})

        autocontrast = ImageOps.autocontrast(base)
        _save("autocontrast", autocontrast)
        _save("contrast_sharp", ImageEnhance.Contrast(autocontrast).enhance(1.8).filter(ImageFilter.SHARPEN))
        _save("threshold", autocontrast.point(lambda px: 255 if px > 176 else 0))
        upscale = autocontrast.resize((max(1, base.width * 2), max(1, base.height * 2)), resampling).filter(ImageFilter.SHARPEN)
        _save("upscale_sharp", upscale)
        denoised_base = base.filter(ImageFilter.MedianFilter(size=3))
        denoised_ac = ImageOps.autocontrast(denoised_base, cutoff=2)
        _save("denoise_median", denoised_ac)
        blurred = base.filter(ImageFilter.GaussianBlur(radius=0.5))
        blurred_ac = ImageOps.autocontrast(blurred, cutoff=2)
        _dbt_pixels = sorted(blurred_ac.getdata())
        _dbt_thresh = _dbt_pixels[len(_dbt_pixels) // 2]
        _save("denoise_blur_threshold", blurred_ac.point(lambda px: 255 if px > _dbt_thresh else 0))
        _deskew_neg = autocontrast.rotate(-3.2, expand=True, fillcolor=255)
        _save("deskew_neg3", _deskew_neg)
        _save("deskew_neg3_sharp", ImageEnhance.Contrast(_deskew_neg).enhance(1.6).filter(ImageFilter.SHARPEN))
        _save("deskew_neg3_threshold", _deskew_neg.point(lambda px: 255 if px > 176 else 0))
        _save("deskew_pos3", autocontrast.rotate(3.2, expand=True, fillcolor=255))
        if CV2_AVAILABLE:
            gray = np.array(autocontrast)
            angle = _estimate_cv2_skew_angle(gray)
            if angle is not None:
                deskewed = _rotate_gray_cv2(gray, angle)
                _save(f"deskew_cv2_{angle:+.2f}".replace("+", "pos").replace("-", "neg"), _cv2_to_pillow_gray(deskewed), detected_angle=angle)
                thresholded = cv2.threshold(deskewed, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)[1]
                _save("deskew_cv2_threshold", _cv2_to_pillow_gray(thresholded), detected_angle=angle)
            for sweep_angle in (-4.0, -3.5, -3.0, -2.5, -2.0, -1.5, -1.0, 1.0, 1.5, 2.0, 2.5, 3.0, 3.5, 4.0):
                _save(f"angle_sweep_{sweep_angle:+.1f}".replace("+", "pos").replace("-", "neg"), autocontrast.rotate(sweep_angle, expand=True, fillcolor=255), detected_angle=sweep_angle)
    except Exception:
        return variants
    return variants


def _looks_like_receipt_candidates(candidates: list[Dict[str, Any]]) -> bool:
    text = "\n".join(str(row.get("markdown") or row.get("text") or "") for row in candidates if row.get("ok")).lower()
    if not text:
        return False
    markers = ("tax invoice", "cash customer", "receipt", "gst", "sst", "cashier", "change due", "sales inclusive", "invoice no")
    return sum(1 for marker in markers if marker in text) >= 2


def _receipt_region_retry_candidates(path: Path, *, max_chars: int) -> list[Dict[str, Any]]:
    if not (PIL_AVAILABLE and PYTESSERACT_AVAILABLE):
        return []
    if not _truthy(os.getenv("OCR97_OCR_RECEIPT_REGION_RETRY"), default=True):
        return []
    started_all = time.perf_counter()
    rows: list[Dict[str, Any]] = []
    try:
        from PIL import ImageEnhance, ImageFilter, ImageOps

        image = Image.open(path)
        image.load()
        base = image.convert("L")
        width, height = base.size
        resampling = getattr(getattr(Image, "Resampling", Image), "LANCZOS", Image.BICUBIC)
        specs = [
            ("receipt_header_top45_psm6", (0.0, 0.0, 1.0, 0.45), "6", "autocontrast"),
            ("receipt_header_top45_threshold_psm6", (0.0, 0.0, 1.0, 0.45), "6", "threshold"),
            ("receipt_body_date_band_psm6", (0.0, 0.25, 1.0, 0.75), "6", "autocontrast"),
            ("receipt_date_mid42_75_psm6", (0.0, 0.42, 1.0, 0.75), "6", "autocontrast"),
            ("receipt_body_date_band_upscale_psm6", (0.0, 0.35, 1.0, 0.75), "6", "upscale"),
            ("receipt_header_left45_psm4", (0.0, 0.0, 0.72, 0.45), "4", "autocontrast"),
        ]
        max_region_candidates = max(1, min(8, int(os.getenv("OCR97_OCR_RECEIPT_REGION_MAX_CANDIDATES", "6"))))
        for label, frac_box, psm, mode in specs[:max_region_candidates]:
            started = time.perf_counter()
            left, top, right, bottom = frac_box
            crop = base.crop((int(width * left), int(height * top), int(width * right), int(height * bottom)))
            crop = ImageOps.autocontrast(crop)
            if mode == "threshold":
                crop = crop.point(lambda px: 255 if px > 170 else 0)
            elif mode == "upscale":
                crop = ImageEnhance.Contrast(crop).enhance(1.25)
                crop = crop.resize((max(1, crop.width * 2), max(1, crop.height * 2)), resampling).filter(ImageFilter.SHARPEN)
            text = pytesseract.image_to_string(crop, config=f"--psm {psm}")
            text = str(text or "").strip()[: max(1000, min(max_chars, 6000))]
            if not text:
                continue
            rows.append(
                {
                    "ok": True,
                    "engine": "tesseract_receipt_region",
                    "preprocess": label,
                    "receipt_region": True,
                    "text": text,
                    "markdown": text,
                    "confidence": None,
                    "latency_ms": round((time.perf_counter() - started) * 1000.0, 2),
                    "quality": _quality_bundle(text, None),
                }
            )
    except Exception as exc:
        rows.append(
            {
                "ok": False,
                "engine": "tesseract_receipt_region",
                "preprocess": "receipt_region_retry",
                "receipt_region": True,
                "latency_ms": round((time.perf_counter() - started_all) * 1000.0, 2),
                "error": f"receipt_region_retry_failed:{type(exc).__name__}:{exc}",
            }
        )
    return rows


def _local_image_preprocessed_best_extract(
    path: Path,
    *,
    goal: str,
    max_pages: int,
    max_chars: int,
    route_mode: str,
    region_retry_policy: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    candidates: list[Dict[str, Any]] = []
    rapid_preprocess = _truthy(os.getenv("OCR97_OCR_PREPROCESS_RAPIDOCR"), default=False)
    include_candidate_text = _truthy(os.getenv("OCR97_OCR_PREPROCESS_INCLUDE_TEXT"), default=False)
    fast_accept_candidate: Optional[Dict[str, Any]] = None
    parallel_workers = max(1, min(8, int(os.getenv("OCR97_OCR_PREPROCESS_WORKERS", "4"))))

    def _run_variant(variant: Dict[str, Any]) -> list[Dict[str, Any]]:
        label = str(variant.get("label") or "")
        vpath = Path(variant.get("path") or path)
        detected_angle = variant.get("detected_angle")
        engines = ("tesseract", "rapidocr") if label == "original" or rapid_preprocess else ("tesseract",)
        rows: list[Dict[str, Any]] = []
        for engine in engines:
            started = time.perf_counter()
            try:
                row = ocr_dual_tool.ocr_dual(
                    {
                        "path": str(vpath),
                        "goal": goal,
                        "engine": engine,
                        "max_pages": max_pages,
                        "max_chars": max_chars,
                        "route_mode": route_mode,
                        "gb10_enabled": True,
                        "use_gateway": False,
                        "region_retry_policy": dict(region_retry_policy or {}),
                    }
                )
            except Exception as exc:
                row = {"ok": False, "engine": engine, "error": f"{engine}_failed:{type(exc).__name__}:{exc}"}
            row = dict(row or {})
            row.setdefault("engine", engine)
            row["preprocess"] = label
            row["detected_angle"] = detected_angle
            row["latency_ms"] = round((time.perf_counter() - started) * 1000.0, 2)
            if row.get("ok") and not row.get("quality"):
                row["quality"] = _quality_bundle(str(row.get("markdown") or row.get("text") or ""), row.get("confidence"))
            rows.append(row)
        return rows

    with tempfile.TemporaryDirectory(prefix="ocr97_preprocess_") as raw_temp_dir:
        temp_dir = Path(raw_temp_dir)
        all_variants = _preprocessed_image_variants(path, temp_dir)
        original_variant = all_variants[0]
        remaining_variants = all_variants[1:]

        for row in _run_variant(original_variant):
            candidates.append(row)
        _rescore_local_image_candidates(candidates)
        fast_accept_candidate = _fast_accept_local_image_candidate(candidates)

        if not fast_accept_candidate and remaining_variants:
            workers = min(parallel_workers, len(remaining_variants))
            with concurrent.futures.ThreadPoolExecutor(max_workers=workers) as pool:
                for rows in pool.map(_run_variant, remaining_variants):
                    candidates.extend(rows)

    _rescore_local_image_candidates(candidates)
    if not fast_accept_candidate and _looks_like_receipt_candidates(candidates):
        candidates.extend(_receipt_region_retry_candidates(path, max_chars=max_chars))
        _rescore_local_image_candidates(candidates)
    full_doc_candidates = [row for row in candidates if not row.get("receipt_region")]
    best_pool = full_doc_candidates or candidates
    best = max(best_pool, key=lambda item: float(item.get("_selection_score") or -1.0)) if best_pool else {}
    out = dict(best)
    out["router"] = "local_image_preprocessed_best"
    out["selected_engine"] = str(best.get("engine") or "")
    out["selected_preprocess"] = str(best.get("preprocess") or "")
    out["route"] = str(best.get("route") or "local_image_preprocessed_best")
    out["fast_accept"] = bool(fast_accept_candidate)
    if fast_accept_candidate:
        out["fast_accept_reason"] = "original_candidate_above_threshold"
    out["field_consensus"] = _field_consensus_from_candidates(full_doc_candidates or candidates)
    if out["field_consensus"]:
        merged = _append_field_consensus(str(out.get("markdown") or out.get("text") or ""), list(out["field_consensus"]))
        out["markdown"] = merged
        out["text"] = merged
        out["field_consensus_used"] = True
    out["receipt_fields"] = receipt_fields_from_candidates(candidates)
    if out["receipt_fields"]:
        merged = append_receipt_fields(str(out.get("markdown") or out.get("text") or ""), list(out["receipt_fields"]))
        out["markdown"] = merged
        out["text"] = merged
        out["receipt_fields_used"] = True
    out["local_image_candidates"] = [
        {
            "engine": str(row.get("engine") or ""),
            "preprocess": str(row.get("preprocess") or ""),
            "detected_angle": row.get("detected_angle"),
            "ok": bool(row.get("ok")),
            "latency_ms": float(row.get("latency_ms") or 0.0),
            "selection_score": float(row.get("_selection_score") or 0.0),
            "score_components": dict(row.get("score_components") or {}),
            "chars": len(str(row.get("markdown") or row.get("text") or "")),
            "receipt_region": bool(row.get("receipt_region")),
            **(
                {"text_snippet": str(row.get("markdown") or row.get("text") or "")[:600]}
                if include_candidate_text
                else {}
            ),
            "error": str(row.get("error") or ""),
        }
        for row in candidates
    ]
    return out


def _infer_ollama_model(base_url: str, preferred: str = "") -> str:
    preferred_model = str(preferred or "").strip()
    if preferred_model:
        return preferred_model
    try:
        response = requests.get(f"{base_url}/api/tags", timeout=4)
        data = response.json() if response.ok else {}
    except Exception:
        return ""
    names = [str((item or {}).get("name") or "") for item in list(data.get("models") or [])]
    for candidate in ("qwen3-vl:32b", "qwen2.5vl:7b", "qwen2.5-vl:7b", "qwen2.5-vl:32b"):
        if candidate in names:
            return candidate
    return names[0] if names else ""


def _prewarm_model(base_url: str, model: str, timeout_sec: int) -> Dict[str, Any]:
    target_model = str(model or "").strip() or _infer_ollama_model(base_url)
    if not target_model:
        return {"ok": False, "error": "no_ollama_model_available"}
    payload = {
        "model": target_model,
        "prompt": "Reply exactly OCR_GATEWAY_WARM.",
        "stream": False,
        "keep_alive": os.getenv("OCR97_OCR_GATEWAY_KEEP_ALIVE", "30m"),
        "options": {"num_predict": 8},
    }
    try:
        response = requests.post(f"{base_url}/api/generate", json=payload, timeout=max(15, timeout_sec))
        if not response.ok:
            return {"ok": False, "error": f"warmup_http_{response.status_code}"}
        return {"ok": True, "model": target_model}
    except Exception as exc:
        return {"ok": False, "error": f"warmup_failed:{type(exc).__name__}:{exc}"}


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

_TABLEFORMER_RUNTIME_LOCK = threading.Lock()
_TABLEFORMER_RUNTIME: Dict[str, Any] = {
    "processor": None,
    "model": None,
    "loaded_model_id": "",
    "last_error": "",
    "backend": "uninitialized",
}

_DOCUNET_RUNTIME_LOCK = threading.Lock()
_DOCUNET_RUNTIME: Dict[str, Any] = {
    "pipeline": None,
    "loaded_model_id": "",
    "last_error": "",
    "backend": "uninitialized",
}

_REALESRGAN_RUNTIME_LOCK = threading.Lock()
_REALESRGAN_RUNTIME: Dict[str, Any] = {
    "upsampler": None,
    "loaded_model_id": "",
    "last_error": "",
    "backend": "uninitialized",
}

_PADDLE_VL_RUNTIME_LOCK = threading.Lock()
_PADDLE_VL_RUNTIME: Dict[str, Any] = {
    "pipeline": None,
    "loaded_model_id": "",
    "last_error": "",
    "backend": "uninitialized",
}


def _paddle_model_id() -> str:
    return str(os.getenv("OCR97_PADDLEOCR_VL_MODEL_ID", "PaddleOCR-VL")).strip() or "PaddleOCR-VL"


def _mineru_model_id() -> str:
    return str(os.getenv("OCR97_MINERU2_5_MODEL_ID", "opendatalab/MinerU")).strip() or "opendatalab/MinerU"


def _olmocr_model_id() -> str:
    return str(os.getenv("OCR97_OLMOCR2_MODEL_ID", "allenai/olmOCR-2-7B")).strip() or "allenai/olmOCR-2-7B"


def _paddle_model_dir() -> Path:
    raw = str(os.getenv("OCR97_PADDLEOCR_VL_MODEL_DIR", "")).strip()
    return Path(raw) if raw else Path.home() / ".cache" / "paddleocr_vl"


def _mineru_model_dir() -> Path:
    raw = str(os.getenv("OCR97_MINERU2_5_MODEL_DIR", "")).strip()
    return Path(raw) if raw else Path.home() / ".cache" / "mineru2_5"


def _olmocr_model_dir() -> Path:
    raw = str(os.getenv("OCR97_OLMOCR2_MODEL_DIR", "")).strip()
    return Path(raw) if raw else Path.home() / ".cache" / "olmocr2"


def _got_backend_enabled() -> bool:
    return _truthy(os.getenv("OCR97_GOT_OCR2_ENABLE_TRANSFORMERS", "1"), default=True)


def _got_force_qwen_fallback() -> bool:
    return _truthy(os.getenv("OCR97_GOT_OCR2_FORCE_QWEN_FALLBACK", "0"), default=False)


def _got_unload_after_request() -> bool:
    return _truthy(os.getenv("OCR97_GOT_OCR2_UNLOAD_AFTER_REQUEST", "1"), default=True)


def _got_model_id() -> str:
    return str(os.getenv("OCR97_GOT_OCR2_MODEL_ID", "stepfun-ai/GOT-OCR-2.0-hf")).strip()


def _got_device() -> str:
    raw = str(os.getenv("OCR97_GOT_OCR2_DEVICE", "auto")).strip().lower() or "auto"
    if raw in {"cpu", "cuda", "auto"}:
        return raw
    return "auto"


def _got_dtype() -> Any:
    if not GOT_TRANSFORMERS_AVAILABLE:
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
    pref = str(raw or "auto").strip().lower() or "auto"
    if torch is None:
        return "cpu"
    if pref == "cuda":
        return "cuda" if torch.cuda.is_available() else "cpu"
    if pref == "cpu":
        return "cpu"
    return "cuda" if torch.cuda.is_available() else "cpu"


def _transformers_device_index(device_name: str) -> int:
    return 0 if torch is not None and str(device_name or "").strip().lower() == "cuda" and torch.cuda.is_available() else -1


def _finbert_model_id() -> str:
    return str(os.getenv("OCR97_FINBERT_MODEL_ID", "ProsusAI/finbert")).strip()


def _finbert_device() -> str:
    return _torch_device_choice(os.getenv("OCR97_FINBERT_DEVICE", "cpu"))


def _tableformer_model_id() -> str:
    return str(os.getenv("OCR97_TABLEFORMER_MODEL_ID", "microsoft/table-transformer-structure-recognition-v1.1-all")).strip()


def _tableformer_device() -> str:
    return _torch_device_choice(os.getenv("OCR97_TABLEFORMER_DEVICE", "auto"))


def _tableformer_dtype() -> Any:
    try:
        return torch.float16 if _tableformer_device() == "cuda" else torch.float32
    except Exception:
        return None


def _docunet_model_id() -> str:
    return str(os.getenv("OCR97_DOCUNET_MODEL_ID", "UVDoc")).strip() or "UVDoc"


def _docunet_device() -> str:
    raw = str(os.getenv("OCR97_DOCUNET_DEVICE", "cpu")).strip().lower() or "cpu"
    if raw.startswith("gpu"):
        return raw
    return "cpu"


def _realesrgan_model_name() -> str:
    return str(os.getenv("OCR97_REALESRGAN_MODEL_NAME", "RealESRGAN_x4plus")).strip() or "RealESRGAN_x4plus"


def _realesrgan_device() -> str:
    return _torch_device_choice(os.getenv("OCR97_REALESRGAN_DEVICE", "auto"))


def _realesrgan_outscale() -> float:
    try:
        return max(1.0, min(float(os.getenv("OCR97_REALESRGAN_OUTSCALE", "2.0")), 4.0))
    except Exception:
        return 2.0


def _realesrgan_tile() -> int:
    try:
        return max(0, int(os.getenv("OCR97_REALESRGAN_TILE", "0")))
    except Exception:
        return 0


def _has_model_assets(model_dir: Path) -> bool:
    try:
        if not model_dir.exists():
            return False
        return any(True for _ in model_dir.iterdir())
    except Exception:
        return False


def _paddle_backend_status() -> Dict[str, Any]:
    override = os.getenv("OCR97_OCR_ENGINE_PADDLEOCR_VL_READY")
    if str(override or "").strip().lower() in {"0", "1", "true", "false", "yes", "no"}:
        ready = _truthy(override, default=False)
        return {
            "ready": ready,
            "reason": "override_flag",
            "remote_url": str(os.getenv("OCR97_GB10_PADDLEOCR_VL_URL") or "").strip(),
            "remote_configured": bool(str(os.getenv("OCR97_GB10_PADDLEOCR_VL_URL") or "").strip()),
            "worker_callable": ready,
            "model_dir": str(_paddle_model_dir()),
            "model_assets_present": ready,
            "runtime_loaded": ready,
            "backend": "override_flag",
            "last_error": "",
        }
    remote_url = str(os.getenv("OCR97_GB10_PADDLEOCR_VL_URL") or "").strip()
    allow_autodownload = _truthy(os.getenv("OCR97_PADDLEOCR_VL_ALLOW_AUTODOWNLOAD", "1"), default=True)
    model_dir = _paddle_model_dir()
    model_assets = _has_model_assets(model_dir)
    local_ready = bool(PADDLE_OCR_AVAILABLE and (model_assets or allow_autodownload))
    remote_ready = bool(remote_url)
    ready = local_ready or remote_ready
    reason = ""
    if ready:
        reason = "remote_configured" if remote_ready and not local_ready else "local_worker_ready"
    elif not PADDLE_OCR_AVAILABLE:
        reason = "engine_not_installed"
    elif not model_assets and not allow_autodownload:
        reason = "model_assets_missing"
    else:
        reason = "worker_unhealthy"
    return {
        "ready": ready,
        "reason": reason,
        "remote_url": remote_url,
        "remote_configured": remote_ready,
        "worker_callable": bool(PADDLE_OCR_AVAILABLE),
        "model_dir": str(model_dir),
        "model_assets_present": model_assets,
        "runtime_loaded": bool(_PADDLE_VL_RUNTIME.get("pipeline") is not None),
        "backend": str(_PADDLE_VL_RUNTIME.get("backend") or ""),
        "last_error": str(_PADDLE_VL_RUNTIME.get("last_error") or ""),
    }


def _mineru_backend_status() -> Dict[str, Any]:
    override = os.getenv("OCR97_OCR_ENGINE_MINERU2_5_READY")
    if str(override or "").strip().lower() in {"0", "1", "true", "false", "yes", "no"}:
        ready = _truthy(override, default=False)
        return {
            "ready": ready,
            "reason": "override_flag",
            "worker_callable": ready,
            "module_available": ready,
            "command_available": ready,
            "command": str(os.getenv("OCR97_MINERU2_5_CMD", "mineru")).strip() or "mineru",
            "model_dir": str(_mineru_model_dir()),
            "model_assets_present": ready,
            "runtime_managed_assets": ready,
        }
    module_ready = _module_available("mineru")
    cmd_name = str(os.getenv("OCR97_MINERU2_5_CMD", "mineru")).strip()
    cmd_ready = bool(shutil.which(cmd_name))
    model_dir = _mineru_model_dir()
    model_assets = _has_model_assets(model_dir)
    runtime_managed_assets = bool(module_ready)
    assets_available = bool(model_assets or runtime_managed_assets)
    ready = bool((module_ready or cmd_ready) and assets_available)
    if ready:
        reason = "worker_ready"
    elif not (module_ready or cmd_ready):
        reason = "engine_not_installed"
    elif not assets_available:
        reason = "model_assets_missing"
    else:
        reason = "worker_unhealthy"
    return {
        "ready": ready,
        "reason": reason,
        "worker_callable": bool(module_ready or cmd_ready),
        "module_available": module_ready,
        "command_available": cmd_ready,
        "command": cmd_name,
        "model_dir": str(model_dir),
        "model_assets_present": model_assets,
        "runtime_managed_assets": runtime_managed_assets,
    }


def _olmocr_backend_status() -> Dict[str, Any]:
    override = os.getenv("OCR97_OCR_ENGINE_OLMOCR2_READY")
    if str(override or "").strip().lower() in {"0", "1", "true", "false", "yes", "no"}:
        ready = _truthy(override, default=False)
        return {
            "ready": ready,
            "reason": "override_flag",
            "worker_callable": ready,
            "module_available": ready,
            "command_available": ready,
            "command": str(os.getenv("OCR97_OLMOCR2_CMD", "olmocr")).strip() or "olmocr",
            "model_dir": str(_olmocr_model_dir()),
            "model_assets_present": ready,
            "runtime_managed_assets": ready,
        }
    module_ready = _module_available("olmocr")
    cmd_name = str(os.getenv("OCR97_OLMOCR2_CMD", "olmocr")).strip()
    cmd_ready = bool(shutil.which(cmd_name))
    model_dir = _olmocr_model_dir()
    model_assets = _has_model_assets(model_dir)
    runtime_managed_assets = bool(module_ready)
    assets_available = bool(model_assets or runtime_managed_assets)
    ready = bool((module_ready or cmd_ready) and assets_available)
    if ready:
        reason = "worker_ready"
    elif not (module_ready or cmd_ready):
        reason = "engine_not_installed"
    elif not assets_available:
        reason = "model_assets_missing"
    else:
        reason = "worker_unhealthy"
    return {
        "ready": ready,
        "reason": reason,
        "worker_callable": bool(module_ready or cmd_ready),
        "module_available": module_ready,
        "command_available": cmd_ready,
        "command": cmd_name,
        "model_dir": str(model_dir),
        "model_assets_present": model_assets,
        "runtime_managed_assets": runtime_managed_assets,
    }


def _engine_health(engine: str) -> Dict[str, Any]:
    name = str(engine or "").strip().lower()
    if name == "gb10_paddleocr_vl":
        return _paddle_backend_status()
    if name == "mineru2_5":
        return _mineru_backend_status()
    if name == "olmocr2":
        return _olmocr_backend_status()
    if name == "gb10_got_ocr2":
        runtime_loaded = bool(_GOT_RUNTIME.get("model") is not None and _GOT_RUNTIME.get("processor") is not None)
        if _got_force_qwen_fallback():
            return {"ready": True, "reason": "forced_qwen_fallback", "runtime_loaded": runtime_loaded}
        if _got_backend_enabled() and GOT_TRANSFORMERS_AVAILABLE and PIL_AVAILABLE:
            return {
                "ready": True,
                "reason": "transformers_available",
                "runtime_loaded": runtime_loaded,
                "last_error": str(_GOT_RUNTIME.get("last_error") or ""),
            }
        return {"ready": False, "reason": "worker_unhealthy", "runtime_loaded": runtime_loaded}
    if name == "native_pdf_text":
        return {"ready": bool(PYMUPDF_AVAILABLE), "reason": "pymupdf_available" if PYMUPDF_AVAILABLE else "pymupdf_missing", "runtime_loaded": False}
    if name in {"local_image_best", "local_image_preprocessed_best"}:
        rapid_available = _module_available("rapidocr_onnxruntime")
        preprocess_ok = True if name == "local_image_best" else bool(PIL_AVAILABLE)
        ready = bool((PYTESSERACT_AVAILABLE or rapid_available) and preprocess_ok)
        return {
            "ready": ready,
            "reason": "local_image_engines_available"
            if ready
            else ("pillow_missing" if not preprocess_ok else "local_image_engines_missing"),
            "runtime_loaded": False,
            "tesseract_available": bool(PYTESSERACT_AVAILABLE),
            "rapidocr_available": bool(rapid_available),
            "pillow_available": bool(PIL_AVAILABLE),
        }
    if name in {"rapidocr", "tesseract", "gb10_qwen_ocr"}:
        return {"ready": True, "reason": "builtin_ready"}
    return {"ready": False, "reason": "unknown_engine"}


def _runtime_loaded_flags() -> Dict[str, bool]:
    mineru = _mineru_backend_status()
    olmocr = _olmocr_backend_status()
    return {
        "gb10_got_ocr2": bool(_GOT_RUNTIME.get("model") is not None and _GOT_RUNTIME.get("processor") is not None),
        "gb10_paddleocr_vl": bool(_PADDLE_VL_RUNTIME.get("pipeline") is not None),
        "mineru2_5": bool(mineru.get("ready")),
        "olmocr2": bool(olmocr.get("ready")),
        "finbert": bool(_FINBERT_RUNTIME.get("pipeline") is not None),
        "tableformer": bool(_TABLEFORMER_RUNTIME.get("processor") is not None and _TABLEFORMER_RUNTIME.get("model") is not None),
        "docunet": bool(_DOCUNET_RUNTIME.get("pipeline") is not None),
        "realesrgan": bool(_REALESRGAN_RUNTIME.get("upsampler") is not None),
    }


def _got_prepare_image_paths(path: Path, max_pages: int) -> list[Path]:
    if str(path.suffix or "").lower() != ".pdf":
        return [path]
    try:
        return list(ocr_dual_tool._render_pdf_pages(path, max_pages=max(1, min(int(max_pages), 12))))
    except Exception:
        return [path]


def _cleanup_temp_paths(paths: list[Path]) -> None:
    for item in paths:
        try:
            if item.exists() and item.name.startswith("OCR97_ocr_pdf_"):
                item.unlink()
        except Exception:
            continue


def _canonicalize_ocr_payload(
    *,
    engine: str,
    model: str,
    markdown: str,
    confidence: Optional[float] = None,
    pages: int = 0,
    route: str = "",
    reason: str = "",
    lane_signature: str = "",
    fallback_reason: str = "",
    raw_tables: Optional[list[Dict[str, Any]]] = None,
    bbox: Optional[list[Dict[str, Any]]] = None,
) -> Dict[str, Any]:
    norm_markdown = ocr_dual_tool._normalize_markdown_layout(markdown, max_chars=80000)
    text = ocr_dual_tool._normalize_text(norm_markdown, max_chars=80000)
    blocks = ocr_dual_tool._blocks_from_text(norm_markdown)
    tables = list(raw_tables or ocr_dual_tool._tables_from_text(norm_markdown))
    reading_order = [item.get("id") for item in blocks if isinstance(item, dict)]
    quality = _quality_bundle(norm_markdown, confidence)
    payload: Dict[str, Any] = {
        "ok": bool(text),
        "engine": engine,
        "model": model,
        "text": text,
        "markdown": norm_markdown,
        "confidence": confidence,
        "pages": int(max(0, pages)),
        "route": route,
        "reason": reason,
        "lane_signature": lane_signature,
        "blocks": blocks,
        "tables": tables,
        "reading_order": reading_order,
        "bbox": list(bbox or []),
        "quality": quality,
    }
    if fallback_reason:
        payload["fallback_reason"] = fallback_reason
    return payload


def _collect_text_artifacts(workspace: Path) -> Dict[str, Any]:
    markdown_candidates: list[tuple[int, Path]] = []
    text_candidates: list[tuple[int, Path]] = []
    tables: list[Dict[str, Any]] = []
    bbox: list[Dict[str, Any]] = []
    pages_detected = 0
    for item in workspace.rglob("*"):
        if not item.is_file():
            continue
        suffix = item.suffix.lower()
        try:
            size = int(item.stat().st_size)
        except Exception:
            size = 0
        if suffix in {".md", ".markdown", ".mmd"}:
            markdown_candidates.append((size, item))
        elif suffix in {".txt", ".text"}:
            text_candidates.append((size, item))
        elif suffix == ".json":
            try:
                payload = json.loads(item.read_text(encoding="utf-8", errors="ignore"))
            except Exception:
                continue
            if isinstance(payload, dict):
                page_like = payload.get("pages")
                if isinstance(page_like, list):
                    pages_detected = max(pages_detected, len(page_like))
                doc_rows = payload.get("documents")
                if isinstance(doc_rows, list):
                    pages_detected = max(pages_detected, len(doc_rows))
                table_rows = payload.get("tables")
                if isinstance(table_rows, list):
                    for row in table_rows:
                        if isinstance(row, dict):
                            tables.append(row)
                bbox_rows = payload.get("bbox")
                if isinstance(bbox_rows, list):
                    for row in bbox_rows:
                        if isinstance(row, dict):
                            bbox.append(row)
                markdown_val = payload.get("markdown")
                text_val = payload.get("text")
                if isinstance(markdown_val, str) and markdown_val.strip():
                    markdown_candidates.append((len(markdown_val), item))
                elif isinstance(text_val, str) and text_val.strip():
                    text_candidates.append((len(text_val), item))
    markdown = ""
    text = ""
    if markdown_candidates:
        _sz, picked = sorted(markdown_candidates, key=lambda row: row[0], reverse=True)[0]
        markdown = picked.read_text(encoding="utf-8", errors="ignore")
    if text_candidates:
        _sz, picked = sorted(text_candidates, key=lambda row: row[0], reverse=True)[0]
        text = picked.read_text(encoding="utf-8", errors="ignore")
    if not markdown:
        markdown = text
    return {
        "markdown": markdown,
        "text": text or markdown,
        "tables": tables,
        "bbox": bbox,
        "pages_detected": pages_detected,
    }


def _tail(value: str, max_chars: int = 1200) -> str:
    raw = str(value or "")
    if len(raw) <= max_chars:
        return raw
    return raw[-max_chars:]


def _mineru_cli_command(path: Path, output_dir: Path, max_pages: int) -> list[str]:
    cmd_name = str(os.getenv("OCR97_MINERU2_5_CMD", "mineru")).strip() or "mineru"
    method = str(os.getenv("OCR97_MINERU2_5_METHOD", "auto")).strip() or "auto"
    backend = str(os.getenv("OCR97_MINERU2_5_BACKEND", "pipeline")).strip() or "pipeline"
    lang = str(os.getenv("OCR97_MINERU2_5_LANG", "en")).strip() or "en"
    command = [cmd_name, "-p", str(path), "-o", str(output_dir), "-m", method, "-b", backend, "-l", lang, "-s", "0"]
    if max_pages > 0:
        command.extend(["-e", str(max(0, max_pages - 1))])
    command.extend(["-f", "true", "-t", "true"])
    server_url = str(os.getenv("OCR97_MINERU2_5_SERVER_URL", "")).strip()
    if server_url:
        command.extend(["-u", server_url])
    return command


def _olmocr_cli_command(path: Path, workspace: Path) -> list[str]:
    cmd_name = str(os.getenv("OCR97_OLMOCR2_CMD", "olmocr")).strip() or "olmocr"
    command = [cmd_name, str(workspace), "--pdfs", str(path), "--markdown", "--workers", "1", "--max_page_retries", "1"]
    model = str(os.getenv("OCR97_OLMOCR2_MODEL_ID", _olmocr_model_id())).strip()
    if model:
        command.extend(["--model", model])
    server = str(os.getenv("OCR97_OLMOCR2_SERVER_URL", "")).strip()
    if server:
        command.extend(["--server", server])
    api_key = str(os.getenv("OCR97_OLMOCR2_API_KEY", "")).strip()
    if api_key:
        command.extend(["--api_key", api_key])
    return command


def _safe_latency_tail(data: Dict[str, Any]) -> Dict[str, Any]:
    return {
        "stderr_tail": _tail(str(data.get("stderr") or "")),
        "stdout_tail": _tail(str(data.get("stdout") or "")),
    }


def _load_paddle_runtime() -> Dict[str, Any]:
    model_id = _paddle_model_id()
    status = _paddle_backend_status()
    if not status.get("worker_callable"):
        return {"ok": False, "error": "engine_not_installed"}
    if not status.get("model_assets_present") and not _truthy(os.getenv("OCR97_PADDLEOCR_VL_ALLOW_AUTODOWNLOAD", "1"), default=True):
        return {"ok": False, "error": "model_assets_missing"}
    with _PADDLE_VL_RUNTIME_LOCK:
        if _PADDLE_VL_RUNTIME.get("pipeline") is not None and str(_PADDLE_VL_RUNTIME.get("loaded_model_id") or "") == model_id:
            return {"ok": True, "pipeline": _PADDLE_VL_RUNTIME.get("pipeline"), "model_id": model_id}
        try:
            det_limit = max(640, min(int(os.getenv("OCR97_PADDLEOCR_VL_DET_LIMIT", "1216")), 2048))
            lang = str(os.getenv("OCR97_PADDLEOCR_VL_LANG", "en")).strip() or "en"
            use_gpu = _truthy(os.getenv("OCR97_PADDLEOCR_VL_USE_GPU", "0"), default=False)
            pipeline_obj = PaddleOCR(use_angle_cls=True, lang=lang, use_gpu=use_gpu, det_limit_side_len=det_limit, show_log=False)
            _PADDLE_VL_RUNTIME["pipeline"] = pipeline_obj
            _PADDLE_VL_RUNTIME["loaded_model_id"] = model_id
            _PADDLE_VL_RUNTIME["backend"] = "paddleocr_local_worker"
            _PADDLE_VL_RUNTIME["last_error"] = ""
            return {"ok": True, "pipeline": pipeline_obj, "model_id": model_id}
        except Exception as exc:
            _PADDLE_VL_RUNTIME["pipeline"] = None
            _PADDLE_VL_RUNTIME["loaded_model_id"] = model_id
            _PADDLE_VL_RUNTIME["backend"] = "load_failed"
            _PADDLE_VL_RUNTIME["last_error"] = f"paddle_load_failed:{type(exc).__name__}:{exc}"
            return {"ok": False, "error": str(_PADDLE_VL_RUNTIME.get("last_error") or "worker_unhealthy")}


def _extract_paddleocr_vl(path: Path, goal: str, max_chars: int, max_pages: int) -> Dict[str, Any]:
    status = _paddle_backend_status()
    remote_url = str(status.get("remote_url") or "").strip()
    if remote_url:
        remote = ocr_dual_tool.ocr_dual(
            {
                "path": str(path),
                "goal": goal,
                "engine": "gb10_paddleocr_vl",
                "max_pages": max_pages,
                "max_chars": max_chars,
                "route_mode": "quality_first",
                "gb10_enabled": True,
                "use_gateway": False,
            }
        )
        if remote.get("ok"):
            remote = dict(remote)
            remote.setdefault("engine", "gb10_paddleocr_vl")
            remote.setdefault("route", "OCR97_paddle_remote")
            remote.setdefault("reason", "paddleocr_vl_remote")
            return remote
    runtime = _load_paddle_runtime()
    if not runtime.get("ok"):
        return {"ok": False, "engine": "gb10_paddleocr_vl", "error": str(runtime.get("error") or "worker_unhealthy")}
    pipeline_obj = runtime.get("pipeline")
    model_id = str(runtime.get("model_id") or _paddle_model_id())
    paths = _got_prepare_image_paths(path, max_pages=max_pages)
    outputs: list[str] = []
    confs: list[float] = []
    cleaned_temp: list[Path] = []
    try:
        for item in paths:
            if item != path:
                cleaned_temp.append(item)
            try:
                img = Image.open(item).convert("RGB")
            except Exception:
                continue
            try:
                import numpy as _np  # type: ignore

                image_np = cv2.cvtColor(_np.array(img), cv2.COLOR_RGB2BGR) if CV2_AVAILABLE else _np.array(img)
                result = pipeline_obj.ocr(image_np, cls=True) if pipeline_obj is not None else []
            except Exception:
                continue
            rows = result[0] if isinstance(result, list) and result else []
            for row in rows or []:
                if not isinstance(row, (list, tuple)) or len(row) < 2:
                    continue
                cell = row[1]
                if not isinstance(cell, (list, tuple)) or len(cell) < 2:
                    continue
                text = str(cell[0] or "").strip()
                if not text:
                    continue
                outputs.append(text)
                try:
                    confs.append(float(cell[1]))
                except Exception:
                    continue
        markdown = ocr_dual_tool._normalize_markdown_layout("\n".join(outputs), max_chars=max_chars)
        text = ocr_dual_tool._normalize_text(markdown, max_chars=max_chars)
        if not text:
            return {"ok": False, "engine": "gb10_paddleocr_vl", "error": "worker_unhealthy:paddle_empty_output"}
        return {
            "ok": True,
            "engine": "gb10_paddleocr_vl",
            "model": model_id,
            "text": text,
            "markdown": markdown,
            "confidence": round(sum(confs) / len(confs), 3) if confs else None,
            "pages": len(paths),
            "route": "OCR97_paddle_local",
            "reason": "paddleocr_vl_local",
        }
    except Exception as exc:
        return {"ok": False, "engine": "gb10_paddleocr_vl", "error": f"worker_unhealthy:{type(exc).__name__}:{exc}"}
    finally:
        _cleanup_temp_paths(cleaned_temp)


def _extract_mineru2_5(path: Path, goal: str, max_chars: int, max_pages: int) -> Dict[str, Any]:
    status = _mineru_backend_status()
    if not status.get("worker_callable"):
        return {"ok": False, "engine": "mineru2_5", "error": "engine_not_installed"}
    if not status.get("model_assets_present"):
        return {"ok": False, "engine": "mineru2_5", "error": "model_assets_missing"}
    timeout_sec = max(60, min(int(os.getenv("OCR97_MINERU2_5_TIMEOUT_SEC", "900")), 3600))
    api_error = ""
    try:
        from mineru.cli.client import run_orchestrated_cli  # type: ignore

        workspace = Path(tempfile.mkdtemp(prefix="OCR97_mineru_api_"))
        try:
            run_orchestrated_cli(
                input_path=path,
                output_dir=workspace,
                method=str(os.getenv("OCR97_MINERU2_5_METHOD", "auto")).strip() or "auto",
                backend=str(os.getenv("OCR97_MINERU2_5_BACKEND", "pipeline")).strip() or "pipeline",
                lang=str(os.getenv("OCR97_MINERU2_5_LANG", "en")).strip() or "en",
                server_url=str(os.getenv("OCR97_MINERU2_5_SERVER_URL", "")).strip() or None,
                api_url=str(os.getenv("OCR97_MINERU2_5_API_URL", "")).strip() or None,
                start_page_id=0,
                end_page_id=max(0, int(max_pages) - 1) if int(max_pages) > 0 else None,
                formula_enable=_truthy(os.getenv("OCR97_MINERU2_5_FORMULA_ENABLE", "1"), default=True),
                table_enable=_truthy(os.getenv("OCR97_MINERU2_5_TABLE_ENABLE", "1"), default=True),
                extra_cli_args=tuple(str(item).strip() for item in str(os.getenv("OCR97_MINERU2_5_EXTRA_ARGS", "")).split() if str(item).strip()),
            )
            artifacts = _collect_text_artifacts(workspace)
            payload = _canonicalize_ocr_payload(
                engine="mineru2_5",
                model=_mineru_model_id(),
                markdown=str(artifacts.get("markdown") or ""),
                pages=int(artifacts.get("pages_detected") or max_pages or 0),
                route="OCR97_mineru_native",
                reason="mineru_native_api",
                lane_signature="mineru_native_api",
                raw_tables=[row for row in list(artifacts.get("tables") or []) if isinstance(row, dict)],
                bbox=[row for row in list(artifacts.get("bbox") or []) if isinstance(row, dict)],
            )
            if payload.get("ok"):
                return payload
            api_error = "worker_unhealthy:mineru_api_empty_output"
        finally:
            shutil.rmtree(workspace, ignore_errors=True)
    except Exception as exc:
        api_error = f"worker_unhealthy:mineru_api_failed:{type(exc).__name__}:{exc}"

    workspace = Path(tempfile.mkdtemp(prefix="OCR97_mineru_cli_"))
    try:
        command = _mineru_cli_command(path, workspace, max_pages=max_pages)
        proc = subprocess.run(command, capture_output=True, text=True, timeout=timeout_sec)
        if proc.returncode != 0:
            return {
                "ok": False,
                "engine": "mineru2_5",
                "error": f"worker_unhealthy:mineru_cli_exit_{proc.returncode}",
                "lane_signature": "mineru_cli_fallback",
                "fallback_reason": api_error or "mineru_native_api_failed",
                **_safe_latency_tail({"stdout": proc.stdout, "stderr": proc.stderr}),
            }
        artifacts = _collect_text_artifacts(workspace)
        payload = _canonicalize_ocr_payload(
            engine="mineru2_5",
            model=_mineru_model_id(),
            markdown=str(artifacts.get("markdown") or ""),
            pages=int(artifacts.get("pages_detected") or max_pages or 0),
            route="OCR97_mineru_native",
            reason="mineru_cli_fallback",
            lane_signature="mineru_cli_fallback",
            fallback_reason=api_error or "mineru_native_api_failed",
            raw_tables=[row for row in list(artifacts.get("tables") or []) if isinstance(row, dict)],
            bbox=[row for row in list(artifacts.get("bbox") or []) if isinstance(row, dict)],
        )
        if payload.get("ok"):
            payload.update(_safe_latency_tail({"stdout": proc.stdout, "stderr": proc.stderr}))
            return payload
        return {
            "ok": False,
            "engine": "mineru2_5",
            "error": "worker_unhealthy:mineru_cli_empty_output",
            "lane_signature": "mineru_cli_fallback",
            "fallback_reason": api_error or "mineru_native_api_failed",
            **_safe_latency_tail({"stdout": proc.stdout, "stderr": proc.stderr}),
        }
    except subprocess.TimeoutExpired:
        return {
            "ok": False,
            "engine": "mineru2_5",
            "error": "engine_timeout",
            "lane_signature": "mineru_cli_fallback",
            "fallback_reason": api_error or "mineru_native_api_failed",
        }
    except Exception as exc:
        return {
            "ok": False,
            "engine": "mineru2_5",
            "error": f"worker_unhealthy:mineru_cli_failed:{type(exc).__name__}:{exc}",
            "lane_signature": "mineru_cli_fallback",
            "fallback_reason": api_error or "mineru_native_api_failed",
        }
    finally:
        shutil.rmtree(workspace, ignore_errors=True)


def _extract_olmocr2(path: Path, goal: str, max_chars: int, max_pages: int) -> Dict[str, Any]:
    status = _olmocr_backend_status()
    if not status.get("worker_callable"):
        return {"ok": False, "engine": "olmocr2", "error": "engine_not_installed"}
    if not status.get("model_assets_present"):
        return {"ok": False, "engine": "olmocr2", "error": "model_assets_missing"}
    timeout_sec = max(60, min(int(os.getenv("OCR97_OLMOCR2_TIMEOUT_SEC", "900")), 3600))
    api_error = ""
    try:
        import olmocr.pipeline as olm_pipeline  # type: ignore

        workspace = Path(tempfile.mkdtemp(prefix="OCR97_olmocr_api_"))
        try:
            args = types.SimpleNamespace(
                workspace=str(workspace),
                pdfs=[str(path)],
                model=str(os.getenv("OCR97_OLMOCR2_MODEL_ID", _olmocr_model_id())).strip(),
                workspace_profile="",
                pdf_profile="",
                pages_per_group=1,
                max_page_retries=max(1, int(os.getenv("OCR97_OLMOCR2_MAX_PAGE_RETRIES", "1"))),
                max_page_error_rate=float(os.getenv("OCR97_OLMOCR2_MAX_PAGE_ERROR_RATE", "1.0")),
                workers=1,
                max_concurrent_requests=1,
                max_server_ready_timeout=max(30, int(os.getenv("OCR97_OLMOCR2_MAX_SERVER_READY_TIMEOUT", "180"))),
                apply_filter=False,
                stats=False,
                markdown=True,
                target_longest_image_dim=max(1024, int(os.getenv("OCR97_OLMOCR2_TARGET_LONGEST_DIM", "2048"))),
                target_anchor_text_len=max(0, int(os.getenv("OCR97_OLMOCR2_TARGET_ANCHOR_TEXT_LEN", "0"))),
                guided_decoding=_truthy(os.getenv("OCR97_OLMOCR2_GUIDED_DECODING", "0"), default=False),
                disk_logging=False,
                server=str(os.getenv("OCR97_OLMOCR2_SERVER_URL", "")).strip() or None,
                api_key=str(os.getenv("OCR97_OLMOCR2_API_KEY", "")).strip() or None,
                gpu_memory_utilization=float(os.getenv("OCR97_OLMOCR2_GPU_MEMORY_UTILIZATION", "0.7")),
                max_model_len=max(2048, int(os.getenv("OCR97_OLMOCR2_MAX_MODEL_LEN", "16384"))),
                tensor_parallel_size=max(1, int(os.getenv("OCR97_OLMOCR2_TP", "1"))),
                data_parallel_size=max(1, int(os.getenv("OCR97_OLMOCR2_DP", "1"))),
                port=max(1024, int(os.getenv("OCR97_OLMOCR2_PORT", "12345"))),
                beaker=False,
                beaker_workspace="",
                beaker_cluster="",
                beaker_gpus="",
                beaker_priority="normal",
            )
            try:
                asyncio.run(olm_pipeline.process_single_pdf(args, worker_id=0, pdf_orig_path=str(path), local_pdf_path=str(path)))
            except Exception as inner_exc:
                raise RuntimeError(f"olmocr_api_process_failed:{type(inner_exc).__name__}:{inner_exc}") from inner_exc
            artifacts = _collect_text_artifacts(workspace)
            payload = _canonicalize_ocr_payload(
                engine="olmocr2",
                model=_olmocr_model_id(),
                markdown=str(artifacts.get("markdown") or ""),
                pages=int(artifacts.get("pages_detected") or max_pages or 0),
                route="OCR97_olmocr_native",
                reason="olmocr_native_api",
                lane_signature="olmocr_native_api",
                raw_tables=[row for row in list(artifacts.get("tables") or []) if isinstance(row, dict)],
                bbox=[row for row in list(artifacts.get("bbox") or []) if isinstance(row, dict)],
            )
            if payload.get("ok"):
                return payload
            api_error = "worker_unhealthy:olmocr_api_empty_output"
        finally:
            shutil.rmtree(workspace, ignore_errors=True)
    except Exception as exc:
        api_error = f"worker_unhealthy:olmocr_api_failed:{type(exc).__name__}:{exc}"

    workspace = Path(tempfile.mkdtemp(prefix="OCR97_olmocr_cli_"))
    try:
        command = _olmocr_cli_command(path, workspace)
        proc = subprocess.run(command, capture_output=True, text=True, timeout=timeout_sec)
        if proc.returncode != 0:
            return {
                "ok": False,
                "engine": "olmocr2",
                "error": f"worker_unhealthy:olmocr_cli_exit_{proc.returncode}",
                "lane_signature": "olmocr_cli_fallback",
                "fallback_reason": api_error or "olmocr_native_api_failed",
                **_safe_latency_tail({"stdout": proc.stdout, "stderr": proc.stderr}),
            }
        artifacts = _collect_text_artifacts(workspace)
        payload = _canonicalize_ocr_payload(
            engine="olmocr2",
            model=_olmocr_model_id(),
            markdown=str(artifacts.get("markdown") or ""),
            pages=int(artifacts.get("pages_detected") or max_pages or 0),
            route="OCR97_olmocr_native",
            reason="olmocr_cli_fallback",
            lane_signature="olmocr_cli_fallback",
            fallback_reason=api_error or "olmocr_native_api_failed",
            raw_tables=[row for row in list(artifacts.get("tables") or []) if isinstance(row, dict)],
            bbox=[row for row in list(artifacts.get("bbox") or []) if isinstance(row, dict)],
        )
        if payload.get("ok"):
            payload.update(_safe_latency_tail({"stdout": proc.stdout, "stderr": proc.stderr}))
            return payload
        return {
            "ok": False,
            "engine": "olmocr2",
            "error": "worker_unhealthy:olmocr_cli_empty_output",
            "lane_signature": "olmocr_cli_fallback",
            "fallback_reason": api_error or "olmocr_native_api_failed",
            **_safe_latency_tail({"stdout": proc.stdout, "stderr": proc.stderr}),
        }
    except subprocess.TimeoutExpired:
        return {
            "ok": False,
            "engine": "olmocr2",
            "error": "engine_timeout",
            "lane_signature": "olmocr_cli_fallback",
            "fallback_reason": api_error or "olmocr_native_api_failed",
        }
    except Exception as exc:
        return {
            "ok": False,
            "engine": "olmocr2",
            "error": f"worker_unhealthy:olmocr_cli_failed:{type(exc).__name__}:{exc}",
            "lane_signature": "olmocr_cli_fallback",
            "fallback_reason": api_error or "olmocr_native_api_failed",
        }
    finally:
        shutil.rmtree(workspace, ignore_errors=True)


def _load_got_runtime() -> Dict[str, Any]:
    global _GOT_RUNTIME_LOCK, _GOT_RUNTIME
    runtime_module = ocr_local_inference._load()
    _GOT_RUNTIME_LOCK = runtime_module._GOT_RUNTIME_LOCK
    _GOT_RUNTIME = runtime_module._GOT_RUNTIME
    runtime = ocr_local_inference.load_got_runtime()
    _GOT_RUNTIME["processor"] = runtime.get("processor")
    _GOT_RUNTIME["model"] = runtime.get("model")
    _GOT_RUNTIME["loaded_model_id"] = str(runtime.get("model_id") or _got_model_id())
    _GOT_RUNTIME["last_error"] = str(runtime.get("error") or "")
    _GOT_RUNTIME["backend"] = str(runtime.get("backend") or ("transformers_got_ocr2" if runtime.get("ok") else "load_failed"))
    return runtime


def _got_extract_transformers(path: Path, goal: str, max_chars: int, max_pages: int) -> Dict[str, Any]:
    result = ocr_local_inference.got_extract_transformers(
        path,
        goal,
        max_chars,
        max_pages,
        normalize_markdown=ocr_dual_tool._normalize_markdown_layout,
        normalize_text=ocr_dual_tool._normalize_text,
    )
    if _got_unload_after_request():
        with _GOT_RUNTIME_LOCK:
            _GOT_RUNTIME["processor"] = None
            _GOT_RUNTIME["model"] = None
            _GOT_RUNTIME["backend"] = "unloaded_after_request"
        try:
            import gc

            gc.collect()
        except Exception:
            pass
        runtime_torch = getattr(ocr_local_inference, "torch", None)
        if GOT_TRANSFORMERS_AVAILABLE and runtime_torch is not None:
            try:
                if runtime_torch.cuda.is_available():
                    runtime_torch.cuda.empty_cache()
            except Exception:
                pass
    return result


def _got_extract_with_fallback(path: Path, goal: str, max_chars: int, max_pages: int) -> Dict[str, Any]:
    if not _got_force_qwen_fallback():
        got = _got_extract_transformers(path=path, goal=goal, max_chars=max_chars, max_pages=max_pages)
        if got.get("ok"):
            return got
        got_error = str(got.get("error") or "got_ocr2_failed")
    else:
        got_error = "got_forced_qwen_fallback"
    fallback = ocr_dual_tool.ocr_dual(
        {
            "path": str(path),
            "goal": goal,
            "engine": "gb10_qwen_ocr",
            "max_pages": max_pages,
            "max_chars": max_chars,
            "route_mode": os.getenv("OCR97_OCR_ROUTE_MODE", "quality_first"),
            "gb10_enabled": True,
            "use_gateway": False,
        }
    )
    if fallback.get("ok"):
        fallback = dict(fallback)
        fallback["engine"] = "gb10_got_ocr2+qwen_fallback"
        fallback["route"] = "OCR97_got_qwen_fallback"
        fallback["fallback_reason"] = got_error
        fallback["reason"] = "got_unavailable_qwen_fallback"
        return fallback
    local = ocr_dual_tool.ocr_dual(
        {
            "path": str(path),
            "goal": goal,
            "engine": "rapidocr",
            "max_pages": max_pages,
            "max_chars": max_chars,
            "route_mode": "balanced",
            "gb10_enabled": False,
            "use_gateway": False,
        }
    )
    if local.get("ok"):
        local = dict(local)
        local["engine"] = "gb10_got_ocr2+rapidocr_fallback"
        local["route"] = "OCR97_got_rapidocr_fallback"
        local["fallback_reason"] = f"{got_error}|{fallback.get('error')}"
        local["reason"] = "got_and_qwen_failed_rapidocr_recovered"
        return local
    tess = ocr_dual_tool.ocr_dual(
        {
            "path": str(path),
            "goal": goal,
            "engine": "tesseract",
            "max_pages": max_pages,
            "max_chars": max_chars,
            "route_mode": "balanced",
            "gb10_enabled": False,
            "use_gateway": False,
        }
    )
    if tess.get("ok"):
        tess = dict(tess)
        tess["engine"] = "gb10_got_ocr2+tesseract_fallback"
        tess["route"] = "OCR97_got_tesseract_fallback"
        tess["fallback_reason"] = f"{got_error}|{fallback.get('error')}|{local.get('error')}"
        tess["reason"] = "got_qwen_rapidocr_failed_tesseract_recovered"
        return tess
    return {
        "ok": False,
        "engine": "gb10_got_ocr2",
        "error": f"got_and_fallbacks_failed:{got_error}|qwen={fallback.get('error')}|rapidocr={local.get('error')}|tesseract={tess.get('error')}",
        "route": "OCR97_got_qwen_fallback",
    }


def _image_to_png_b64(image: "Image.Image") -> str:
    buffer = BytesIO()
    image.save(buffer, format="PNG")
    return base64.b64encode(buffer.getvalue()).decode("ascii")


def _image_from_b64(raw: str) -> Optional["Image.Image"]:
    token = str(raw or "").strip()
    if not token or not PIL_AVAILABLE:
        return None
    if "," in token and token.lower().startswith("data:image/"):
        token = token.split(",", 1)[1]
    try:
        data = base64.b64decode(token)
        return Image.open(BytesIO(data)).convert("RGB")
    except Exception:
        return None


def _load_image_input(source_path: str = "", image_b64: str = "") -> Optional["Image.Image"]:
    image = _image_from_b64(image_b64)
    if image is not None:
        return image
    path = Path(str(source_path or "").strip())
    if not path.exists():
        return None
    return _render_table_source_image(path)


def _markdown_table_rows(text: str, max_rows: int = 32) -> list[list[str]]:
    rows: list[list[str]] = []
    for raw_line in str(text or "").splitlines():
        line = raw_line.strip()
        if not line:
            continue
        if "|" in line and line.count("|") >= 2:
            cells = [part.strip() for part in line.split("|")]
            cells = [cell for cell in cells if cell]
            if not cells:
                continue
            if all(set(cell) <= {"-", ":"} for cell in cells):
                continue
            rows.append(cells)
            continue
        if "\t" in line:
            cells = [part.strip() for part in line.split("\t") if part.strip()]
            if len(cells) >= 2:
                rows.append(cells)
                continue
        parts = [part.strip() for part in line.split("  ") if part.strip()]
        if len(parts) >= 2:
            rows.append(parts)
    return rows[:max_rows]


def _synthesized_axis_boxes(table_box: Dict[str, float], count: int, axis: str) -> list[Dict[str, float]]:
    total = max(1, int(count))
    boxes: list[Dict[str, float]] = []
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


def _patch_paddlex_opencv_gate() -> Optional[str]:
    if not CV2_AVAILABLE:
        return "cv2_unavailable"
    try:
        import paddlex.utils.deps as paddlex_deps
    except Exception as exc:
        return f"paddlex_deps_unavailable:{type(exc).__name__}:{exc}"

    original = paddlex_deps.is_dep_available

    def _patched(dep, /, check_version=False):
        if dep == "opencv-contrib-python":
            return True
        return original(dep, check_version)

    paddlex_deps.is_dep_available = _patched
    for module_name in (
        "paddlex.inference.common.reader.image_reader",
        "paddlex.inference.utils.io.readers",
        "paddlex.inference.utils.io.writers",
        "paddlex.inference.models.common.vision.processors",
        "paddlex.inference.models.common.vision.funcs",
    ):
        try:
            module = importlib.import_module(module_name)
            setattr(module, "cv2", cv2)
        except Exception:
            continue
    return ""


def _load_docunet_runtime() -> Dict[str, Any]:
    if not (PADDLE_DOCPREPROCESSOR_AVAILABLE and PIL_AVAILABLE):
        return {"ok": False, "error": "docunet_runtime_unavailable"}
    with _DOCUNET_RUNTIME_LOCK:
        if _DOCUNET_RUNTIME.get("pipeline") is not None:
            return {"ok": True, "pipeline": _DOCUNET_RUNTIME.get("pipeline"), "model_id": str(_DOCUNET_RUNTIME.get("loaded_model_id") or _docunet_model_id())}
        try:
            os.environ.setdefault("DISABLE_MODEL_SOURCE_CHECK", "True")
            patch_error = _patch_paddlex_opencv_gate()
            if patch_error:
                raise RuntimeError(patch_error)
            pipeline_obj = DocPreprocessor(
                use_doc_orientation_classify=False,
                use_doc_unwarping=True,
                device=_docunet_device(),
            )
            _DOCUNET_RUNTIME["pipeline"] = pipeline_obj
            _DOCUNET_RUNTIME["loaded_model_id"] = _docunet_model_id()
            _DOCUNET_RUNTIME["last_error"] = ""
            _DOCUNET_RUNTIME["backend"] = "docunet_paddle_uvdoc"
            return {"ok": True, "pipeline": pipeline_obj, "model_id": _docunet_model_id()}
        except Exception as exc:
            _DOCUNET_RUNTIME["pipeline"] = None
            _DOCUNET_RUNTIME["loaded_model_id"] = _docunet_model_id()
            _DOCUNET_RUNTIME["last_error"] = f"docunet_load_failed:{type(exc).__name__}:{exc}"
            _DOCUNET_RUNTIME["backend"] = "load_failed"
            return {"ok": False, "error": _DOCUNET_RUNTIME["last_error"]}


def _docunet_service_rectify(source_path: str = "", image_b64: str = "") -> Dict[str, Any]:
    runtime = _load_docunet_runtime()
    if not runtime.get("ok"):
        return {"ok": False, "error": str(runtime.get("error") or "docunet_runtime_unavailable")}
    image = _load_image_input(source_path=source_path, image_b64=image_b64)
    if image is None:
        return {"ok": False, "error": "source_image_required"}
    pipeline_obj = runtime.get("pipeline")
    try:
        image_array = ocr_dual_tool.np.array(image.convert("RGB"))[:, :, ::-1]
        results = list(pipeline_obj.predict([image_array], use_doc_orientation_classify=False, use_doc_unwarping=True))
        if not results:
            return {"ok": False, "error": "docunet_no_output"}
        item = results[0]
        output_img = item["output_img"]
        rectified = Image.fromarray(output_img.astype("uint8"))
        return {
            "ok": True,
            "mode": "docunet",
            "model": str(runtime.get("model_id") or _docunet_model_id()),
            "backend": str(_DOCUNET_RUNTIME.get("backend") or ""),
            "width": rectified.size[0],
            "height": rectified.size[1],
            "image_b64": _image_to_png_b64(rectified),
        }
    except Exception as exc:
        _DOCUNET_RUNTIME["last_error"] = f"docunet_infer_failed:{type(exc).__name__}:{exc}"
        return {"ok": False, "error": _DOCUNET_RUNTIME["last_error"]}


def _install_torchvision_functional_tensor_shim() -> None:
    if "torchvision.transforms.functional_tensor" in sys.modules:
        return
    try:
        import torchvision.transforms.functional as tv_functional
    except Exception:
        return
    shim = types.ModuleType("torchvision.transforms.functional_tensor")
    for name in dir(tv_functional):
        setattr(shim, name, getattr(tv_functional, name))
    sys.modules["torchvision.transforms.functional_tensor"] = shim


def _realesrgan_model_spec() -> Dict[str, Any]:
    model_name = _realesrgan_model_name().lower()
    if model_name == "realesrgan_x2plus":
        from basicsr.archs.rrdbnet_arch import RRDBNet

        return {
            "name": "RealESRGAN_x2plus",
            "scale": 2,
            "weights": "https://github.com/xinntao/Real-ESRGAN/releases/download/v0.2.1/RealESRGAN_x2plus.pth",
            "network": RRDBNet(num_in_ch=3, num_out_ch=3, num_feat=64, num_block=23, num_grow_ch=32, scale=2),
        }
    from basicsr.archs.rrdbnet_arch import RRDBNet

    return {
        "name": "RealESRGAN_x4plus",
        "scale": 4,
        "weights": "https://github.com/xinntao/Real-ESRGAN/releases/download/v0.1.0/RealESRGAN_x4plus.pth",
        "network": RRDBNet(num_in_ch=3, num_out_ch=3, num_feat=64, num_block=23, num_grow_ch=32, scale=4),
    }


def _load_realesrgan_runtime() -> Dict[str, Any]:
    with _REALESRGAN_RUNTIME_LOCK:
        if _REALESRGAN_RUNTIME.get("upsampler") is not None:
            return {"ok": True, "upsampler": _REALESRGAN_RUNTIME.get("upsampler"), "model_id": str(_REALESRGAN_RUNTIME.get("loaded_model_id") or "")}
        try:
            _install_torchvision_functional_tensor_shim()
            from realesrgan import RealESRGANer

            spec = _realesrgan_model_spec()
            upsampler = None
            backend_device = ""
            load_errors: list[str] = []
            for device_name in [_realesrgan_device(), "cpu"]:
                if device_name in {backend_device}:
                    continue
                try:
                    upsampler = RealESRGANer(
                        scale=int(spec["scale"]),
                        model_path=str(spec["weights"]),
                        model=spec["network"],
                        tile=_realesrgan_tile(),
                        tile_pad=10,
                        pre_pad=0,
                        half=device_name == "cuda",
                        device=torch.device(device_name),
                    )
                    backend_device = device_name
                    break
                except Exception as exc:
                    load_errors.append(f"{device_name}:{type(exc).__name__}:{exc}")
                    upsampler = None
            if upsampler is None:
                raise RuntimeError("|".join(load_errors) or "realesrgan_init_failed")
            _REALESRGAN_RUNTIME["upsampler"] = upsampler
            _REALESRGAN_RUNTIME["loaded_model_id"] = str(spec["name"])
            _REALESRGAN_RUNTIME["last_error"] = ""
            _REALESRGAN_RUNTIME["backend"] = f"realesrgan_{backend_device}"
            return {"ok": True, "upsampler": upsampler, "model_id": str(spec["name"])}
        except Exception as exc:
            _REALESRGAN_RUNTIME["upsampler"] = None
            _REALESRGAN_RUNTIME["loaded_model_id"] = _realesrgan_model_name()
            _REALESRGAN_RUNTIME["last_error"] = f"realesrgan_load_failed:{type(exc).__name__}:{exc}"
            _REALESRGAN_RUNTIME["backend"] = "load_failed"
            return {"ok": False, "error": _REALESRGAN_RUNTIME["last_error"]}


def _realesrgan_service_upscale(source_path: str = "", image_b64: str = "", outscale: Optional[float] = None) -> Dict[str, Any]:
    runtime = _load_realesrgan_runtime()
    if not runtime.get("ok"):
        return {"ok": False, "error": str(runtime.get("error") or "realesrgan_runtime_unavailable")}
    image = _load_image_input(source_path=source_path, image_b64=image_b64)
    if image is None:
        return {"ok": False, "error": "source_image_required"}
    try:
        bgr = ocr_dual_tool.np.array(image.convert("RGB"))[:, :, ::-1]
        upsampler = runtime.get("upsampler")
        target_scale = float(outscale) if outscale is not None else _realesrgan_outscale()
        output, _ = upsampler.enhance(bgr, outscale=target_scale)
        rgb = Image.fromarray(output[:, :, ::-1].astype("uint8"))
        return {
            "ok": True,
            "mode": "realesrgan",
            "model": str(runtime.get("model_id") or _realesrgan_model_name()),
            "backend": str(_REALESRGAN_RUNTIME.get("backend") or ""),
            "width": rgb.size[0],
            "height": rgb.size[1],
            "image_b64": _image_to_png_b64(rgb),
            "outscale": target_scale,
        }
    except Exception as exc:
        _REALESRGAN_RUNTIME["last_error"] = f"realesrgan_infer_failed:{type(exc).__name__}:{exc}"
        return {"ok": False, "error": _REALESRGAN_RUNTIME["last_error"]}


def _load_finbert_runtime() -> Dict[str, Any]:
    global _FINBERT_RUNTIME_LOCK, _FINBERT_RUNTIME
    runtime_module = ocr_local_inference._load()
    _FINBERT_RUNTIME_LOCK = runtime_module._FINBERT_RUNTIME_LOCK
    _FINBERT_RUNTIME = runtime_module._FINBERT_RUNTIME
    runtime = ocr_local_inference.load_finbert_runtime()
    _FINBERT_RUNTIME["pipeline"] = runtime.get("pipeline")
    _FINBERT_RUNTIME["loaded_model_id"] = str(runtime.get("model_id") or _finbert_model_id())
    _FINBERT_RUNTIME["last_error"] = str(runtime.get("error") or "")
    _FINBERT_RUNTIME["backend"] = "transformers_finbert" if runtime.get("ok") else "load_failed"
    return runtime


def _finbert_service_eval(text: str) -> Dict[str, Any]:
    result = ocr_local_inference.finbert_eval(text)
    if not result.get("ok"):
        _FINBERT_RUNTIME["last_error"] = str(result.get("error") or "finbert_runtime_unavailable")
    return result


def _load_tableformer_runtime() -> Dict[str, Any]:
    global _TABLEFORMER_RUNTIME_LOCK, _TABLEFORMER_RUNTIME
    runtime_module = ocr_local_inference._load()
    _TABLEFORMER_RUNTIME_LOCK = runtime_module._TABLE_RUNTIME_LOCK
    _TABLEFORMER_RUNTIME = runtime_module._TABLE_RUNTIME
    runtime = ocr_local_inference.load_tableformer_runtime()
    _TABLEFORMER_RUNTIME["processor"] = runtime.get("processor")
    _TABLEFORMER_RUNTIME["model"] = runtime.get("model")
    _TABLEFORMER_RUNTIME["loaded_model_id"] = str(runtime.get("model_id") or _tableformer_model_id())
    _TABLEFORMER_RUNTIME["last_error"] = str(runtime.get("error") or "")
    _TABLEFORMER_RUNTIME["backend"] = (
        f"transformers_tableformer_{_tableformer_device()}" if runtime.get("ok") else "load_failed"
    )
    return runtime


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
        return Image.open(source_path).convert("RGB")
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


def _extract_cell_text(cell_image: "Image.Image") -> str:
    if not PYTESSERACT_AVAILABLE:
        return ""
    try:
        text = pytesseract.image_to_string(cell_image, config="--psm 7")
    except Exception:
        return ""
    return ocr_dual_tool._normalize_text(text, max_chars=200)


def _tableformer_service_reconstruct(source_path: str, text: str = "") -> Dict[str, Any]:
    result = ocr_local_inference.tableformer_reconstruct(
        source_path,
        text=text,
        normalize_text=ocr_dual_tool._normalize_text,
    )
    if not result.get("ok"):
        _TABLEFORMER_RUNTIME["last_error"] = str(result.get("error") or "tableformer_runtime_unavailable")
    return result


@dataclass
class _GatewayState:
    max_concurrency: int
    running: int = 0
    queued: int = 0
    processed: int = 0
    failures: int = 0
    last_error: str = ""
    last_heartbeat: str = ""
    warmed_models: set[str] = field(default_factory=set)
    lane_events: Dict[str, list[Dict[str, Any]]] = field(default_factory=dict)
    lock: threading.RLock = field(default_factory=threading.RLock)
    semaphore: threading.BoundedSemaphore = field(init=False)

    def __post_init__(self) -> None:
        self.semaphore = threading.BoundedSemaphore(max(1, int(self.max_concurrency)))


def register_gb10_ocr_gateway_routes(app, instance_name: str, upload_dir: Optional[Path] = None) -> None:
    upload_root = _resolve_upload_dir(instance_name, upload_dir) / "gateway"
    state = _GatewayState(max_concurrency=max(1, int(os.getenv("OCR97_OCR_GATEWAY_MAX_CONCURRENCY", "2"))))
    default_timeout_sec = max(30, int(os.getenv("OCR97_OCR_GATEWAY_TIMEOUT_SEC", "180")))
    install_manifest_path = _install_metadata_path(instance_name, upload_root)
    ollama_url = str(
        os.getenv("OCR97_OCR_GB10_OLLAMA_URL")
        or os.getenv("OCR97_GB10_QWEN_OLLAMA_URL")
        or os.getenv("OLLAMA_URL_FALLBACK")
        or os.getenv("OLLAMA_URL")
        or "http://127.0.0.1:11434"
    ).strip()
    warm_interval_sec = max(60, int(os.getenv("OCR97_OCR_GATEWAY_PREWARM_INTERVAL_SEC", "300")))
    prewarm_on_start = _truthy(os.getenv("OCR97_OCR_GATEWAY_PREWARM_ON_STARTUP", "0"), default=False)
    prewarm_enabled = _truthy(os.getenv("OCR97_OCR_GATEWAY_PREWARM_ENABLED", "0"), default=False)
    warm_state: Dict[str, Any] = {
        "enabled": prewarm_enabled,
        "interval_sec": warm_interval_sec,
        "last_run": "",
        "last_success": "",
        "next_run": "",
        "failures": 0,
        "warmed_engines": [],
        "last_error": "",
        "startup_done": False,
    }
    warm_lock = threading.Lock()
    metrics_window_hours = max(1, int(os.getenv("OCR97_OCR_SLO_WINDOW_HOURS", "24")))
    smoke_required = _truthy(os.getenv("OCR97_OCR_SMOKE_REQUIRED", "1"), default=True)
    smoke_report_path = Path(str(os.getenv("OCR97_OCR_SMOKE_REPORT_PATH", "")).strip() or (upload_root.parent / "ocr_smoke_report.json"))
    p95_caps_by_class = {
        "layout": max(1000, int(os.getenv("OCR97_OCR_SLO_P95_LAYOUT_MS", "30000"))),
        "dense_scan": max(1000, int(os.getenv("OCR97_OCR_SLO_P95_DENSE_SCAN_MS", "30000"))),
        "structure_parser": max(1000, int(os.getenv("OCR97_OCR_SLO_P95_STRUCTURE_MS", "45000"))),
        "linearization": max(1000, int(os.getenv("OCR97_OCR_SLO_P95_LINEARIZATION_MS", "45000"))),
        "semantic_cleanup": max(1000, int(os.getenv("OCR97_OCR_SLO_P95_SEMANTIC_MS", "20000"))),
        "compat_fallback": max(1000, int(os.getenv("OCR97_OCR_SLO_P95_COMPAT_MS", "15000"))),
        "native_text": max(1000, int(os.getenv("OCR97_OCR_SLO_P95_NATIVE_TEXT_MS", "8000"))),
        "image_preprocessor": max(1000, int(os.getenv("OCR97_OCR_SLO_P95_IMAGE_PREPROCESSOR_MS", "120000"))),
        "unknown": max(1000, int(os.getenv("OCR97_OCR_SLO_P95_UNKNOWN_MS", "30000"))),
    }

    def _iso_to_dt(value: Any) -> Optional[datetime]:
        raw = str(value or "").strip()
        if not raw:
            return None
        try:
            if raw.endswith("Z"):
                raw = raw[:-1] + "+00:00"
            return datetime.fromisoformat(raw)
        except Exception:
            return None

    def _load_smoke_report() -> Dict[str, Any]:
        data = _safe_json_load(smoke_report_path)
        if not data:
            return {
                "ok": False,
                "pass": False,
                "required": smoke_required,
                "path": str(smoke_report_path),
                "last_run": "",
                "engines": {},
                "reason": "smoke_report_missing",
            }
        if "path" not in data:
            data["path"] = str(smoke_report_path)
        data["required"] = smoke_required
        return data

    def _record_lane_event(engine: str, *, ok: bool, latency_ms: float, fallback_used: bool, timed_out: bool) -> None:
        now = datetime.now(timezone.utc)
        cutoff = now - timedelta(hours=metrics_window_hours)
        with state.lock:
            rows = list(state.lane_events.get(engine) or [])
            rows = [row for row in rows if (_iso_to_dt(row.get("ts")) or now) >= cutoff]
            rows.append(
                {
                    "ts": now.replace(microsecond=0).isoformat().replace("+00:00", "Z"),
                    "ok": bool(ok),
                    "latency_ms": float(max(0.0, latency_ms)),
                    "fallback_used": bool(fallback_used),
                    "timed_out": bool(timed_out),
                }
            )
            state.lane_events[engine] = rows[-3000:]

    def _lane_slo(engine: str) -> Dict[str, Any]:
        now = datetime.now(timezone.utc)
        cutoff = now - timedelta(hours=metrics_window_hours)
        with state.lock:
            rows = [row for row in list(state.lane_events.get(engine) or []) if (_iso_to_dt(row.get("ts")) or now) >= cutoff]
        count = len(rows)
        if count <= 0:
            return {
                "window_hours": metrics_window_hours,
                "last_24h_count": 0,
                "success_rate": 0.0,
                "timeout_rate": 0.0,
                "fallback_rate": 0.0,
                "p95_latency_ms": 0.0,
            }
        latencies = sorted(float(row.get("latency_ms") or 0.0) for row in rows)
        p95_index = max(0, min(len(latencies) - 1, int(round(0.95 * len(latencies) + 0.499)) - 1))
        p95 = float(latencies[p95_index]) if latencies else 0.0
        success = sum(1 for row in rows if bool(row.get("ok")))
        timeouts = sum(1 for row in rows if bool(row.get("timed_out")))
        fallbacks = sum(1 for row in rows if bool(row.get("fallback_used")))
        return {
            "window_hours": metrics_window_hours,
            "last_24h_count": count,
            "success_rate": round(success / float(count), 4),
            "timeout_rate": round(timeouts / float(count), 4),
            "fallback_rate": round(fallbacks / float(count), 4),
            "p95_latency_ms": round(p95, 2),
        }

    def _build_install_manifest() -> Dict[str, Any]:
        existing = _safe_json_load(install_manifest_path)
        now = _utc_iso()
        manifest = {
            "updated_at": now,
            "python": sys.version.split(" ")[0],
            "engines": {
                "gb10_paddleocr_vl": {
                    "version": diag.package_version("paddleocr") if PADDLE_OCR_AVAILABLE else "",
                    "model_id": _paddle_model_id(),
                    "model_dir": str(_paddle_model_dir()),
                    "model_hash": _hash_dir_metadata(_paddle_model_dir()),
                    "status": _paddle_backend_status(),
                },
                "mineru2_5": {
                    "version": diag.package_version("mineru") if _module_available("mineru") else "",
                    "model_id": _mineru_model_id(),
                    "model_dir": str(_mineru_model_dir()),
                    "model_hash": _hash_dir_metadata(_mineru_model_dir()),
                    "status": _mineru_backend_status(),
                },
                "olmocr2": {
                    "version": diag.package_version("olmocr") if _module_available("olmocr") else "",
                    "model_id": _olmocr_model_id(),
                    "model_dir": str(_olmocr_model_dir()),
                    "model_hash": _hash_dir_metadata(_olmocr_model_dir()),
                    "status": _olmocr_backend_status(),
                },
            },
            "source": "OCR97_gb10_ocr_gateway",
        }
        if existing:
            manifest["previous_updated_at"] = str(existing.get("updated_at") or "")
        _safe_json_write(install_manifest_path, manifest)
        return manifest

    def _compute_cold_start_ms(engine: str) -> int:
        name = str(engine or "").strip().lower()
        if name == "gb10_got_ocr2":
            return 4500
        if name == "gb10_paddleocr_vl":
            return 3500
        if name in {"mineru2_5", "olmocr2"}:
            return 6000
        if name == "gb10_qwen_ocr":
            return 2500
        if name in {"finbert", "tableformer"}:
            return 1800
        return 900

    def _run_prewarm(warm_model: str = "") -> Dict[str, Any]:
        results: Dict[str, Any] = {"ok": True, "engines": {}, "model": ""}
        now = _utc_iso()
        try:
            got = _load_got_runtime()
            results["engines"]["gb10_got_ocr2"] = {"ok": bool(got.get("ok")), "error": str(got.get("error") or "")}
            finbert = _load_finbert_runtime()
            results["engines"]["finbert"] = {"ok": bool(finbert.get("ok")), "error": str(finbert.get("error") or "")}
            table = _load_tableformer_runtime()
            results["engines"]["tableformer"] = {"ok": bool(table.get("ok")), "error": str(table.get("error") or "")}
            model = str(warm_model or os.getenv("OCR97_OCR_GATEWAY_PREWARM_MODEL", "")).strip() or _infer_ollama_model(ollama_url, preferred=DEFAULT_GB10_QWEN_OCR_MODEL)
            warm = _prewarm_model(ollama_url, model, timeout_sec=default_timeout_sec)
            results["engines"]["gb10_qwen_ocr"] = {"ok": bool(warm.get("ok")), "error": str(warm.get("error") or ""), "model": str(warm.get("model") or model)}
            results["model"] = str(warm.get("model") or model)
            if not all(bool((row or {}).get("ok")) for row in dict(results.get("engines") or {}).values()):
                results["ok"] = False
        except Exception as exc:
            results = {"ok": False, "engines": {}, "error": f"prewarm_failed:{type(exc).__name__}:{exc}", "model": ""}
        with warm_lock:
            warm_state["last_run"] = now
            warm_state["next_run"] = (
                (datetime.now(timezone.utc) + timedelta(seconds=warm_interval_sec))
                .replace(microsecond=0)
                .isoformat()
                .replace("+00:00", "Z")
            )
            if results.get("ok"):
                warm_state["last_success"] = now
                warm_state["warmed_engines"] = sorted([name for name, row in dict(results.get("engines") or {}).items() if bool((row or {}).get("ok"))])
                warm_state["last_error"] = ""
            else:
                warm_state["failures"] = int(warm_state.get("failures") or 0) + 1
                warm_state["last_error"] = str(results.get("error") or "")
        return results

    def _start_prewarm_loop() -> None:
        if not prewarm_enabled:
            return

        def _loop() -> None:
            if prewarm_on_start:
                _run_prewarm()
                with warm_lock:
                    warm_state["startup_done"] = True
            while True:
                time.sleep(max(30, warm_interval_sec))
                _run_prewarm()

        thread = threading.Thread(target=_loop, name="OCR97-ocr-prewarm", daemon=True)
        thread.start()

    def _default_smoke_fixtures() -> list[Path]:
        raw = str(os.getenv("OCR97_OCR_SMOKE_FIXTURES", "")).strip()
        rows: list[Path] = []
        if raw:
            for token in re.split(r"[;,]", raw):
                path = Path(token.strip())
                if path.exists() and path.is_file():
                    rows.append(path)
        if rows:
            return rows
        fallback_root = Path(__file__).resolve().parent / "fixtures"
        if fallback_root.exists():
            for item in sorted(fallback_root.glob("*.pdf"))[:4]:
                if item.exists():
                    rows.append(item)
        return rows

    def _run_smoke_suite() -> Dict[str, Any]:
        fixtures = _default_smoke_fixtures()
        started = _utc_iso()
        if not fixtures:
            report = {
                "ok": False,
                "pass": False,
                "last_run": started,
                "required": smoke_required,
                "reason": "smoke_fixture_missing",
                "engines": {},
                "fixtures": [],
                "path": str(smoke_report_path),
            }
            _safe_json_write(smoke_report_path, report)
            return report
        engines = ["gb10_paddleocr_vl", "gb10_got_ocr2", "mineru2_5", "olmocr2", "gb10_qwen_ocr"]
        engine_rows: Dict[str, Any] = {}
        passes = 0
        for idx, engine in enumerate(engines):
            fixture = fixtures[min(idx, len(fixtures) - 1)]
            row = _run_engine(
                fixture,
                goal="OCR smoke validation",
                engine=engine,
                max_pages=2,
                max_chars=12000,
                route_mode="quality_first",
                region_retry_policy={},
                bypass_ready_gate=True,
            )
            quality = dict(row.get("quality") or {})
            engine_rows[engine] = {
                "ok": bool(row.get("ok")),
                "lane_signature": str(row.get("lane_signature") or ""),
                "latency_ms": float(row.get("latency_ms") or 0.0),
                "chars": int(quality.get("chars") or len(str(row.get("markdown") or row.get("text") or "").strip())),
                "fallback_reason": str(row.get("fallback_reason") or row.get("error") or ""),
                "fixture": str(fixture),
            }
            if bool(row.get("ok")):
                passes += 1
        report = {
            "ok": passes == len(engines),
            "pass": passes == len(engines),
            "required": smoke_required,
            "last_run": started,
            "engines": engine_rows,
            "fixtures": [str(item) for item in fixtures],
            "path": str(smoke_report_path),
        }
        _safe_json_write(smoke_report_path, report)
        return report

    install_manifest: Dict[str, Any] = _build_install_manifest()
    _start_prewarm_loop()

    def _engine_snapshot(engine: str) -> Dict[str, Any]:
        health = _engine_health(engine)
        runtime_flags = _runtime_loaded_flags()
        lane_class = _engine_class(engine)
        slo = _lane_slo(engine)
        smoke = _load_smoke_report()
        smoke_rows = dict(smoke.get("engines") or {})
        smoke_row = dict(smoke_rows.get(engine) or {})
        smoke_pass = bool(smoke_row.get("pass")) or bool(smoke_row.get("ok"))
        p95_cap = int(p95_caps_by_class.get(lane_class, p95_caps_by_class.get("unknown", 30000)))
        timeout_ok = float(slo.get("timeout_rate") or 0.0) < 0.05
        fallback_ok = float(slo.get("fallback_rate") or 0.0) < 0.20
        p95_ok = float(slo.get("p95_latency_ms") or 0.0) <= float(p95_cap)
        smoke_ok = smoke_pass or not smoke_required
        health_ready = bool(health.get("ready"))
        ready_gate = {
            "mode": "balanced",
            "pass": bool(health_ready and smoke_ok and timeout_ok and fallback_ok and p95_ok),
            "smoke_required": bool(smoke_required),
            "smoke_pass": bool(smoke_pass),
            "timeout_rate_ok": bool(timeout_ok),
            "fallback_rate_ok": bool(fallback_ok),
            "p95_ok": bool(p95_ok),
            "p95_cap_ms": int(p95_cap),
        }
        return {
            "name": engine,
            "class": lane_class,
            "ready": bool(ready_gate.get("pass")),
            "health_ready": health_ready,
            "reason": str(health.get("reason") or ""),
            "health": health,
            "runtime_loaded": bool(runtime_flags.get(engine)),
            "cold_start_estimate_ms": _compute_cold_start_ms(engine),
            "native_api_supported": _engine_native_api_supported(engine),
            "cli_supported": _engine_cli_supported(engine),
            "lane_signature_modes": _engine_signature_modes(engine),
            "slo": slo,
            "ready_gate": ready_gate,
            "smoke": {
                "pass": bool(smoke_pass),
                "last_run": str(smoke.get("last_run") or ""),
                "path": str(smoke.get("path") or ""),
                "details": smoke_row,
            },
        }

    def _engine_ready(engine: str) -> bool:
        return bool(_engine_snapshot(engine).get("ready"))

    def _engine_class(engine: str) -> str:
        mapping = {
            "gb10_paddleocr_vl": "layout",
            "gb10_got_ocr2": "dense_scan",
            "mineru2_5": "structure_parser",
            "olmocr2": "linearization",
            "gb10_qwen_ocr": "semantic_cleanup",
            "rapidocr": "compat_fallback",
            "tesseract": "compat_fallback",
            "native_pdf_text": "native_text",
            "local_image_best": "image_router",
            "local_image_preprocessed_best": "image_preprocessor",
        }
        return mapping.get(str(engine or "").strip().lower(), "unknown")

    def _engine_native_api_supported(engine: str) -> bool:
        name = str(engine or "").strip().lower()
        if name == "mineru2_5":
            return _module_available("mineru.cli.client")
        if name == "olmocr2":
            return _module_available("olmocr.pipeline")
        return False

    def _engine_cli_supported(engine: str) -> bool:
        name = str(engine or "").strip().lower()
        if name == "mineru2_5":
            return bool(shutil.which(str(os.getenv("OCR97_MINERU2_5_CMD", "mineru")).strip() or "mineru"))
        if name == "olmocr2":
            return bool(shutil.which(str(os.getenv("OCR97_OLMOCR2_CMD", "olmocr")).strip() or "olmocr"))
        return False

    def _engine_signature_modes(engine: str) -> list[str]:
        name = str(engine or "").strip().lower()
        if name == "mineru2_5":
            return ["mineru_native_api", "mineru_cli_fallback"]
        if name == "olmocr2":
            return ["olmocr_native_api", "olmocr_cli_fallback"]
        if name == "gb10_paddleocr_vl":
            return ["paddleocr_vl_worker"]
        if name == "gb10_got_ocr2":
            return ["got_ocr2_worker"]
        if name == "gb10_qwen_ocr":
            return ["qwen_ocr_worker"]
        if name == "rapidocr":
            return ["rapidocr_worker"]
        if name == "tesseract":
            return ["tesseract_worker"]
        if name == "native_pdf_text":
            return ["native_pdf_text"]
        if name == "local_image_best":
            return ["tesseract_worker", "rapidocr_worker"]
        if name == "local_image_preprocessed_best":
            return ["pillow_preprocess", "tesseract_worker", "rapidocr_worker"]
        return [f"{name}_worker"] if name else []

    def _default_lane_signature(engine: str, row: Dict[str, Any]) -> str:
        if str(row.get("lane_signature") or "").strip():
            return str(row.get("lane_signature") or "").strip()
        name = str(engine or "").strip().lower()
        if name == "mineru2_5":
            return "mineru_native_unknown"
        if name == "olmocr2":
            return "olmocr_native_unknown"
        if name == "gb10_paddleocr_vl":
            return "paddleocr_vl_worker"
        if name == "gb10_got_ocr2":
            return "got_ocr2_worker"
        if name == "native_pdf_text":
            return "native_pdf_text"
        if name == "local_image_best":
            return "local_image_best"
        if name == "local_image_preprocessed_best":
            return "local_image_preprocessed_best"
        if name == "gb10_qwen_ocr":
            return "qwen_ocr_worker"
        if name == "rapidocr":
            return "rapidocr_worker"
        if name == "tesseract":
            return "tesseract_worker"
        return f"{name or 'unknown'}_worker"

    def _normalize_result(
        raw: Dict[str, Any],
        *,
        attempts: list[Dict[str, Any]],
        chain: list[str],
        route_mode: str,
        doc_class: str,
        fallback_reason: str = "",
        document_features: Optional[Dict[str, Any]] = None,
        visual_controls: Optional[list[Dict[str, Any]]] = None,
        feature_detection: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        text = str(raw.get("text") or "").strip()
        markdown = str(raw.get("markdown") or text).strip()
        receipt_fields = list(raw.get("receipt_fields") or [])
        receipt_fields_used = bool(raw.get("receipt_fields_used"))
        if bool(raw.get("ok")) and markdown and not receipt_fields:
            receipt_fields = receipt_fields_from_candidates(
                [
                    {
                        "ok": True,
                        "engine": str(raw.get("engine") or ""),
                        "preprocess": str(raw.get("selected_preprocess") or raw.get("preprocess") or ""),
                        "markdown": markdown,
                        "selection_score": float((raw.get("quality") or {}).get("score") or 0.0),
                    }
                ]
            )
            if receipt_fields:
                markdown = append_receipt_fields(markdown, receipt_fields)
                text = markdown
                receipt_fields_used = True
        quality = dict(raw.get("quality") or {})
        if not quality:
            quality = _quality_bundle(markdown, raw.get("confidence"))
        blocks = list(raw.get("blocks") or _blocks_from_text(markdown))
        tables = list(raw.get("tables") or _tables_from_text(markdown))
        reading_order = list(raw.get("reading_order") or [item.get("id") for item in blocks if isinstance(item, dict)])
        bbox = list(raw.get("bbox") or [])
        out = {
            "ok": bool(raw.get("ok")) and bool(markdown),
            "engine": str(raw.get("engine") or ""),
            "model": str(raw.get("model") or ""),
            "lane_signature": str(raw.get("lane_signature") or ""),
            "router": str(raw.get("router") or ""),
            "selected_engine": str(raw.get("selected_engine") or ""),
            "selected_preprocess": str(raw.get("selected_preprocess") or ""),
            "route": str(raw.get("route") or ""),
            "reason": str(raw.get("reason") or ""),
            "error": str(raw.get("error") or ""),
            "confidence": raw.get("confidence"),
            "pages": int(raw.get("pages") or 0),
            "text": text,
            "markdown": markdown,
            "blocks": blocks,
            "tables": tables,
            "reading_order": reading_order,
            "bbox": bbox,
            "engine_chain": chain,
            "attempts": attempts,
            "local_image_candidates": list(raw.get("local_image_candidates") or []),
            "field_consensus": list(raw.get("field_consensus") or []),
            "field_consensus_used": bool(raw.get("field_consensus_used")),
            "receipt_fields": receipt_fields,
            "receipt_fields_used": receipt_fields_used,
            "phase2": dict(raw.get("phase2") or {}),
            "quality": {
                "score": float(quality.get("score") or 0.0),
                "chars": int(quality.get("chars") or len(markdown)),
                "confidence": quality.get("confidence"),
                "structure_score": float(quality.get("structure_score") or 0.0),
                "numeric_fidelity_score": float(quality.get("numeric_fidelity_score") or 0.0),
                "table_rows": int(quality.get("table_rows") or 0),
                "finance_consistency": dict(quality.get("finance_consistency") or {}),
                "finbert_eval": dict(quality.get("finbert_eval") or {}),
                "table_reconstruction": dict(quality.get("table_reconstruction") or {}),
                "route_mode": route_mode,
                "doc_class": doc_class,
            },
            "fallback_reason": fallback_reason,
            "document_features": dict(document_features or raw.get("document_features") or {}),
            "layout_regions": list((document_features or raw.get("document_features") or {}).get("layout_regions") or raw.get("layout_regions") or []),
            "visual_controls": list(visual_controls or raw.get("visual_controls") or []),
            "feature_detection": dict(feature_detection or raw.get("feature_detection") or {}),
        }
        return out

    def _run_with_timeout(fn, *, timeout_sec: int, fallback_engine: str) -> Dict[str, Any]:
        holder: Dict[str, Any] = {"result": {"ok": False, "engine": fallback_engine, "error": "worker_unhealthy"}}

        def _task() -> None:
            try:
                holder["result"] = fn()
            except Exception as exc:  # pragma: no cover - defensive
                holder["result"] = {"ok": False, "engine": fallback_engine, "error": f"worker_unhealthy:{type(exc).__name__}:{exc}"}

        thread = threading.Thread(target=_task, daemon=True)
        thread.start()
        thread.join(timeout=max(1, int(timeout_sec)))
        if thread.is_alive():
            return {"ok": False, "engine": fallback_engine, "error": "engine_timeout"}
        return dict(holder.get("result") or {})

    def _run_engine(
        path: Path,
        goal: str,
        engine: str,
        max_pages: int,
        max_chars: int,
        route_mode: str,
        region_retry_policy: Optional[Dict[str, Any]] = None,
        bypass_ready_gate: bool = False,
    ) -> Dict[str, Any]:
        start = time.perf_counter()
        timeout_sec = max(15, min(default_timeout_sec, int(os.getenv("OCR97_OCR_ENGINE_TIMEOUT_SEC", str(default_timeout_sec)))))
        snapshot = _engine_snapshot(engine)
        if not snapshot.get("ready") and not bypass_ready_gate:
            gate = dict(snapshot.get("ready_gate") or {})
            gate_reason = []
            if not bool(snapshot.get("health_ready")):
                gate_reason.append(str(snapshot.get("reason") or "worker_unhealthy"))
            if not bool(gate.get("smoke_pass")) and bool(gate.get("smoke_required")):
                gate_reason.append("smoke_gate_failed")
            if not bool(gate.get("timeout_rate_ok")):
                gate_reason.append("timeout_rate_high")
            if not bool(gate.get("fallback_rate_ok")):
                gate_reason.append("fallback_rate_high")
            if not bool(gate.get("p95_ok")):
                gate_reason.append("p95_latency_high")
            result: Dict[str, Any] = {
                "ok": False,
                "engine": engine,
                "error": "worker_unhealthy:" + ",".join([item for item in gate_reason if item]) if gate_reason else "worker_unhealthy",
                "ready_gate": gate,
            }
        elif engine == "gb10_paddleocr_vl":
            result = _run_with_timeout(
                lambda: _extract_paddleocr_vl(path, goal=goal, max_chars=max_chars, max_pages=max_pages),
                timeout_sec=timeout_sec,
                fallback_engine=engine,
            )
        elif engine == "mineru2_5":
            result = _run_with_timeout(
                lambda: _extract_mineru2_5(path, goal=goal, max_chars=max_chars, max_pages=max_pages),
                timeout_sec=timeout_sec,
                fallback_engine=engine,
            )
        elif engine == "olmocr2":
            result = _run_with_timeout(
                lambda: _extract_olmocr2(path, goal=goal, max_chars=max_chars, max_pages=max_pages),
                timeout_sec=timeout_sec,
                fallback_engine=engine,
            )
        elif engine == "native_pdf_text":
            result = _native_pdf_text_extract(path, max_pages=max_pages, max_chars=max_chars)
        elif engine == "local_image_best":
            result = _local_image_best_extract(
                path,
                goal=goal,
                max_pages=max_pages,
                max_chars=max_chars,
                route_mode=route_mode,
                region_retry_policy=region_retry_policy,
            )
        elif engine == "local_image_preprocessed_best":
            result = _local_image_preprocessed_best_extract(
                path,
                goal=goal,
                max_pages=max_pages,
                max_chars=max_chars,
                route_mode=route_mode,
                region_retry_policy=region_retry_policy,
            )
        else:
            result = ocr_dual_tool.ocr_dual(
                {
                    "path": str(path),
                    "goal": goal,
                    "engine": engine,
                    "max_pages": max_pages,
                    "max_chars": max_chars,
                    "route_mode": route_mode,
                    "gb10_enabled": True,
                    "use_gateway": False,
                    "region_retry_policy": dict(region_retry_policy or {}),
                }
            )
        elapsed_ms = round((time.perf_counter() - start) * 1000.0, 2)
        quality = dict(result.get("quality") or {})
        if not quality:
            quality = _quality_bundle(str(result.get("text") or ""), result.get("confidence"))
        lane_signature = _default_lane_signature(engine, result)
        fallback_used = bool(str(result.get("fallback_reason") or "").strip())
        timed_out = str(result.get("error") or "").strip().startswith("engine_timeout")
        _record_lane_event(
            engine,
            ok=bool(result.get("ok")),
            latency_ms=elapsed_ms,
            fallback_used=fallback_used,
            timed_out=timed_out,
        )
        return {
            **dict(result or {}),
            "engine": str(result.get("engine") or engine),
            "lane_signature": lane_signature,
            "latency_ms": elapsed_ms,
            "quality": quality,
        }

    def _resolve_path_from_request(prefix: str) -> tuple[Optional[Path], Optional[Any]]:
        upload_root.mkdir(parents=True, exist_ok=True)
        file = request.files.get("file")
        json_payload = request.get_json(silent=True) if request.is_json else {}
        path_raw = str(request.form.get("path") or ((json_payload or {}).get("path") if isinstance(json_payload, dict) else "") or "").strip()
        local_path: Optional[Path] = None
        if file is not None:
            suffix = Path(file.filename or "").suffix
            local_path = upload_root / f"{prefix}_{uuid.uuid4().hex}{suffix}"
            try:
                file.save(local_path)
            except Exception as exc:
                return None, jsonify({"ok": False, "error": f"save_failed:{exc}"})
        elif path_raw:
            candidate = Path(path_raw)
            if not candidate.exists():
                return None, jsonify({"ok": False, "error": f"path_not_found:{path_raw}"})
            local_path = candidate
        if local_path is None:
            return None, jsonify({"ok": False, "error": "file_or_path_required"})
        return local_path, None

    @app.route("/ocr/got/extract", methods=["POST"])
    def ocr_got_extract():
        if _auth_required() and not _auth_ok():
            return jsonify({"ok": False, "error": "unauthorized"}), 401
        local_path, error_resp = _resolve_path_from_request("got_ocr")
        if local_path is None:
            return error_resp, 400
        json_payload = request.get_json(silent=True) if request.is_json else {}

        goal = str(request.form.get("goal") or ((json_payload or {}).get("goal") if isinstance(json_payload, dict) else "") or "").strip()
        max_pages_raw = request.form.get("max_pages") or (((json_payload or {}).get("max_pages")) if isinstance(json_payload, dict) else None)
        max_chars_raw = request.form.get("max_chars") or (((json_payload or {}).get("max_chars")) if isinstance(json_payload, dict) else None)
        try:
            max_pages = max(1, min(int(max_pages_raw or 4), 16))
            max_chars = max(1000, min(int(max_chars_raw or 20000), 80000))
        except Exception:
            return jsonify({"ok": False, "error": "invalid_numeric_param"}), 400

        response = _got_extract_with_fallback(local_path, goal=goal, max_chars=max_chars, max_pages=max_pages)
        response = dict(response or {})
        response["provider"] = "OCR97_got_ocr2_service"
        response["ts"] = _utc_iso()
        status = 200 if response.get("ok") else 422
        return jsonify(response), status

    @app.route("/ocr/got/health", methods=["GET"])
    def ocr_got_health():
        if _auth_required() and not _auth_ok():
            return jsonify({"ok": False, "error": "unauthorized"}), 401
        runtime_loaded = bool(_GOT_RUNTIME.get("model") is not None and _GOT_RUNTIME.get("processor") is not None)
        payload = {
            "ok": True,
            "provider": "OCR97_got_ocr2_service",
            "ts": _utc_iso(),
            "backend_enabled": _got_backend_enabled(),
            "force_qwen_fallback": _got_force_qwen_fallback(),
            "unload_after_request": _got_unload_after_request(),
            "transformers_available": GOT_TRANSFORMERS_AVAILABLE,
            "pil_available": PIL_AVAILABLE,
            "runtime_loaded": runtime_loaded,
            "model_id": _got_model_id(),
            "device_pref": _got_device(),
            "loaded_model_id": str(_GOT_RUNTIME.get("loaded_model_id") or ""),
            "last_error": str(_GOT_RUNTIME.get("last_error") or ""),
            "backend": str(_GOT_RUNTIME.get("backend") or ""),
            "qwen_fallback_ready": _engine_ready("gb10_qwen_ocr"),
        }
        return jsonify(payload), 200

    @app.route("/ocr/paddle/extract", methods=["POST"])
    def ocr_paddle_extract():
        if _auth_required() and not _auth_ok():
            return jsonify({"ok": False, "error": "unauthorized"}), 401
        local_path, error_resp = _resolve_path_from_request("paddle_ocr")
        if local_path is None:
            return error_resp, 400
        json_payload = request.get_json(silent=True) if request.is_json else {}
        goal = str(request.form.get("goal") or ((json_payload or {}).get("goal") if isinstance(json_payload, dict) else "") or "").strip()
        max_pages = max(1, min(int(request.form.get("max_pages") or ((json_payload or {}).get("max_pages") or 4)), 16))
        max_chars = max(1000, min(int(request.form.get("max_chars") or ((json_payload or {}).get("max_chars") or 20000)), 80000))
        result = _extract_paddleocr_vl(local_path, goal=goal, max_chars=max_chars, max_pages=max_pages)
        result = dict(result or {})
        result["provider"] = "OCR97_paddleocr_vl_service"
        result["ts"] = _utc_iso()
        return jsonify(result), 200 if result.get("ok") else 422

    @app.route("/ocr/paddle/health", methods=["GET"])
    def ocr_paddle_health():
        if _auth_required() and not _auth_ok():
            return jsonify({"ok": False, "error": "unauthorized"}), 401
        status = _paddle_backend_status()
        return jsonify(
            {
                "ok": True,
                "provider": "OCR97_paddleocr_vl_service",
                "ts": _utc_iso(),
                "runtime_loaded": bool(_PADDLE_VL_RUNTIME.get("pipeline") is not None),
                "model_id": _paddle_model_id(),
                "loaded_model_id": str(_PADDLE_VL_RUNTIME.get("loaded_model_id") or ""),
                "last_error": str(_PADDLE_VL_RUNTIME.get("last_error") or ""),
                "backend": str(_PADDLE_VL_RUNTIME.get("backend") or ""),
                "status": status,
            }
        ), 200

    @app.route("/ocr/mineru/extract", methods=["POST"])
    def ocr_mineru_extract():
        if _auth_required() and not _auth_ok():
            return jsonify({"ok": False, "error": "unauthorized"}), 401
        local_path, error_resp = _resolve_path_from_request("mineru_ocr")
        if local_path is None:
            return error_resp, 400
        json_payload = request.get_json(silent=True) if request.is_json else {}
        goal = str(request.form.get("goal") or ((json_payload or {}).get("goal") if isinstance(json_payload, dict) else "") or "").strip()
        max_pages = max(1, min(int(request.form.get("max_pages") or ((json_payload or {}).get("max_pages") or 4)), 16))
        max_chars = max(1000, min(int(request.form.get("max_chars") or ((json_payload or {}).get("max_chars") or 20000)), 80000))
        result = _extract_mineru2_5(local_path, goal=goal, max_chars=max_chars, max_pages=max_pages)
        result = dict(result or {})
        result["provider"] = "OCR97_mineru2_5_service"
        result["ts"] = _utc_iso()
        return jsonify(result), 200 if result.get("ok") else 422

    @app.route("/ocr/mineru/health", methods=["GET"])
    def ocr_mineru_health():
        if _auth_required() and not _auth_ok():
            return jsonify({"ok": False, "error": "unauthorized"}), 401
        return jsonify({"ok": True, "provider": "OCR97_mineru2_5_service", "ts": _utc_iso(), "status": _mineru_backend_status(), "model_id": _mineru_model_id()}), 200

    @app.route("/ocr/olmocr/extract", methods=["POST"])
    def ocr_olmocr_extract():
        if _auth_required() and not _auth_ok():
            return jsonify({"ok": False, "error": "unauthorized"}), 401
        local_path, error_resp = _resolve_path_from_request("olmocr_ocr")
        if local_path is None:
            return error_resp, 400
        json_payload = request.get_json(silent=True) if request.is_json else {}
        goal = str(request.form.get("goal") or ((json_payload or {}).get("goal") if isinstance(json_payload, dict) else "") or "").strip()
        max_pages = max(1, min(int(request.form.get("max_pages") or ((json_payload or {}).get("max_pages") or 4)), 16))
        max_chars = max(1000, min(int(request.form.get("max_chars") or ((json_payload or {}).get("max_chars") or 20000)), 80000))
        result = _extract_olmocr2(local_path, goal=goal, max_chars=max_chars, max_pages=max_pages)
        result = dict(result or {})
        result["provider"] = "OCR97_olmocr2_service"
        result["ts"] = _utc_iso()
        return jsonify(result), 200 if result.get("ok") else 422

    @app.route("/ocr/olmocr/health", methods=["GET"])
    def ocr_olmocr_health():
        if _auth_required() and not _auth_ok():
            return jsonify({"ok": False, "error": "unauthorized"}), 401
        return jsonify({"ok": True, "provider": "OCR97_olmocr2_service", "ts": _utc_iso(), "status": _olmocr_backend_status(), "model_id": _olmocr_model_id()}), 200

    @app.route("/ocr/finbert/eval", methods=["POST"])
    def ocr_finbert_eval():
        if _auth_required() and not _auth_ok():
            return jsonify({"ok": False, "error": "unauthorized"}), 401
        json_payload = request.get_json(silent=True) if request.is_json else {}
        text = str(request.form.get("text") or ((json_payload or {}).get("text") if isinstance(json_payload, dict) else "") or "").strip()
        result = _finbert_service_eval(text)
        result = dict(result or {})
        result["provider"] = "OCR97_finbert_service"
        result["ts"] = _utc_iso()
        return jsonify(result), 200 if result.get("ok") else 422

    @app.route("/ocr/finbert/health", methods=["GET"])
    def ocr_finbert_health():
        if _auth_required() and not _auth_ok():
            return jsonify({"ok": False, "error": "unauthorized"}), 401
        payload = {
            "ok": True,
            "provider": "OCR97_finbert_service",
            "ts": _utc_iso(),
            "transformers_available": PHASE2_TRANSFORMERS_AVAILABLE,
            "runtime_loaded": bool(_FINBERT_RUNTIME.get("pipeline") is not None),
            "model_id": _finbert_model_id(),
            "device": _finbert_device(),
            "loaded_model_id": str(_FINBERT_RUNTIME.get("loaded_model_id") or ""),
            "last_error": str(_FINBERT_RUNTIME.get("last_error") or ""),
            "backend": str(_FINBERT_RUNTIME.get("backend") or ""),
        }
        return jsonify(payload), 200

    @app.route("/ocr/table/reconstruct", methods=["POST"])
    def ocr_table_reconstruct():
        if _auth_required() and not _auth_ok():
            return jsonify({"ok": False, "error": "unauthorized"}), 401
        json_payload = request.get_json(silent=True) if request.is_json else {}
        source_path = str(request.form.get("source_path") or ((json_payload or {}).get("source_path") if isinstance(json_payload, dict) else "") or "").strip()
        text = str(request.form.get("text") or ((json_payload or {}).get("text") if isinstance(json_payload, dict) else "") or "").strip()
        result = _tableformer_service_reconstruct(source_path, text=text)
        result = dict(result or {})
        result["provider"] = "OCR97_tableformer_service"
        result["ts"] = _utc_iso()
        return jsonify(result), 200 if result.get("ok") else 422

    @app.route("/ocr/table/health", methods=["GET"])
    def ocr_table_health():
        if _auth_required() and not _auth_ok():
            return jsonify({"ok": False, "error": "unauthorized"}), 401
        payload = {
            "ok": True,
            "provider": "OCR97_tableformer_service",
            "ts": _utc_iso(),
            "transformers_available": PHASE2_TRANSFORMERS_AVAILABLE,
            "pymupdf_available": PYMUPDF_AVAILABLE,
            "pytesseract_available": PYTESSERACT_AVAILABLE,
            "runtime_loaded": bool(_TABLEFORMER_RUNTIME.get("processor") is not None and _TABLEFORMER_RUNTIME.get("model") is not None),
            "model_id": _tableformer_model_id(),
            "device": _tableformer_device(),
            "loaded_model_id": str(_TABLEFORMER_RUNTIME.get("loaded_model_id") or ""),
            "last_error": str(_TABLEFORMER_RUNTIME.get("last_error") or ""),
            "backend": str(_TABLEFORMER_RUNTIME.get("backend") or ""),
        }
        return jsonify(payload), 200

    @app.route("/ocr/docunet/rectify", methods=["POST"])
    def ocr_docunet_rectify():
        if _auth_required() and not _auth_ok():
            return jsonify({"ok": False, "error": "unauthorized"}), 401
        json_payload = request.get_json(silent=True) if request.is_json else {}
        source_path = str(request.form.get("source_path") or ((json_payload or {}).get("source_path") if isinstance(json_payload, dict) else "") or "").strip()
        image_b64 = str(request.form.get("image_b64") or ((json_payload or {}).get("image_b64") if isinstance(json_payload, dict) else "") or "").strip()
        result = _docunet_service_rectify(source_path=source_path, image_b64=image_b64)
        result = dict(result or {})
        result["provider"] = "OCR97_docunet_service"
        result["ts"] = _utc_iso()
        return jsonify(result), 200 if result.get("ok") else 422

    @app.route("/ocr/docunet/health", methods=["GET"])
    def ocr_docunet_health():
        if _auth_required() and not _auth_ok():
            return jsonify({"ok": False, "error": "unauthorized"}), 401
        payload = {
            "ok": True,
            "provider": "OCR97_docunet_service",
            "ts": _utc_iso(),
            "paddle_docpreprocessor_available": PADDLE_DOCPREPROCESSOR_AVAILABLE,
            "cv2_available": CV2_AVAILABLE,
            "runtime_loaded": bool(_DOCUNET_RUNTIME.get("pipeline") is not None),
            "model_id": _docunet_model_id(),
            "device": _docunet_device(),
            "loaded_model_id": str(_DOCUNET_RUNTIME.get("loaded_model_id") or ""),
            "last_error": str(_DOCUNET_RUNTIME.get("last_error") or ""),
            "backend": str(_DOCUNET_RUNTIME.get("backend") or ""),
        }
        return jsonify(payload), 200

    @app.route("/ocr/realesrgan/upscale", methods=["POST"])
    def ocr_realesrgan_upscale():
        if _auth_required() and not _auth_ok():
            return jsonify({"ok": False, "error": "unauthorized"}), 401
        json_payload = request.get_json(silent=True) if request.is_json else {}
        source_path = str(request.form.get("source_path") or ((json_payload or {}).get("source_path") if isinstance(json_payload, dict) else "") or "").strip()
        image_b64 = str(request.form.get("image_b64") or ((json_payload or {}).get("image_b64") if isinstance(json_payload, dict) else "") or "").strip()
        outscale_raw = request.form.get("outscale") or (((json_payload or {}).get("outscale")) if isinstance(json_payload, dict) else None)
        try:
            outscale = float(outscale_raw) if outscale_raw is not None else None
        except Exception:
            return jsonify({"ok": False, "error": "invalid_outscale"}), 400
        result = _realesrgan_service_upscale(source_path=source_path, image_b64=image_b64, outscale=outscale)
        result = dict(result or {})
        result["provider"] = "OCR97_realesrgan_service"
        result["ts"] = _utc_iso()
        return jsonify(result), 200 if result.get("ok") else 422

    @app.route("/ocr/realesrgan/health", methods=["GET"])
    def ocr_realesrgan_health():
        if _auth_required() and not _auth_ok():
            return jsonify({"ok": False, "error": "unauthorized"}), 401
        payload = {
            "ok": True,
            "provider": "OCR97_realesrgan_service",
            "ts": _utc_iso(),
            "runtime_loaded": bool(_REALESRGAN_RUNTIME.get("upsampler") is not None),
            "model_id": _realesrgan_model_name(),
            "device": _realesrgan_device(),
            "outscale_default": _realesrgan_outscale(),
            "tile": _realesrgan_tile(),
            "loaded_model_id": str(_REALESRGAN_RUNTIME.get("loaded_model_id") or ""),
            "last_error": str(_REALESRGAN_RUNTIME.get("last_error") or ""),
            "backend": str(_REALESRGAN_RUNTIME.get("backend") or ""),
        }
        return jsonify(payload), 200

    @app.route("/ocr/extract", methods=["POST"])
    def ocr_extract():
        if _auth_required() and not _auth_ok():
            return jsonify({"ok": False, "error": "unauthorized"}), 401
        local_path, error_resp = _resolve_path_from_request("gateway_ocr")
        if local_path is None:
            return error_resp, 400
        json_payload = request.get_json(silent=True) if request.is_json else {}

        goal = str(request.form.get("goal") or ((json_payload or {}).get("goal") if isinstance(json_payload, dict) else "") or "").strip()
        model = str(request.form.get("model") or ((json_payload or {}).get("model") if isinstance(json_payload, dict) else "") or "auto").strip().lower()
        route_mode = str(request.form.get("route_mode") or ((json_payload or {}).get("route_mode") if isinstance(json_payload, dict) else "") or os.getenv("OCR97_OCR_ROUTE_MODE", "quality_first")).strip().lower()
        explicit_lane = model not in {"", "auto", "gb10_auto"}
        requested_lane_strict = _truthy(
            request.form.get("requested_lane_strict")
            or ((json_payload or {}).get("requested_lane_strict") if isinstance(json_payload, dict) else None),
            default=explicit_lane,
        )
        max_pages_raw = request.form.get("max_pages") or (((json_payload or {}).get("max_pages")) if isinstance(json_payload, dict) else None)
        max_chars_raw = request.form.get("max_chars") or (((json_payload or {}).get("max_chars")) if isinstance(json_payload, dict) else None)
        timeout_raw = request.form.get("timeout_sec") or (((json_payload or {}).get("timeout_sec")) if isinstance(json_payload, dict) else None)
        prewarm = _truthy(request.form.get("prewarm") or (((json_payload or {}).get("prewarm")) if isinstance(json_payload, dict) else None), default=False)
        warm_hint = str(request.form.get("warm_hint") or ((json_payload or {}).get("warm_hint") if isinstance(json_payload, dict) else "") or "").strip()
        region_retry_policy = dict(((json_payload or {}).get("region_retry_policy")) if isinstance(json_payload, dict) and isinstance((json_payload or {}).get("region_retry_policy"), dict) else {})
        bypass_ready_gate = _truthy(
            request.form.get("bypass_ready_gate")
            or ((json_payload or {}).get("bypass_ready_gate") if isinstance(json_payload, dict) else None),
            default=False,
        ) and _truthy(os.getenv("OCR97_OCR_ALLOW_READY_GATE_BYPASS", "0"), default=False)

        try:
            max_pages = max(1, min(int(max_pages_raw or 4), 16))
            max_chars = max(1000, min(int(max_chars_raw or 20000), 80000))
            timeout_sec = max(30, min(int(timeout_raw or default_timeout_sec), 900))
        except Exception:
            return jsonify({"ok": False, "error": "invalid_numeric_param"}), 400

        with state.lock:
            state.queued += 1
        acquired = state.semaphore.acquire(timeout=max(5, min(timeout_sec, 45)))
        with state.lock:
            state.queued = max(0, state.queued - 1)
        if not acquired:
            return jsonify({"ok": False, "error": "gateway_busy", "queue_depth": state.queued}), 429

        with state.lock:
            state.running += 1
            state.last_heartbeat = _utc_iso()
        try:
            if prewarm:
                warm_model = str(request.form.get("warm_model") or (((json_payload or {}).get("warm_model")) if isinstance(json_payload, dict) else "") or warm_hint).strip()
                warm = _run_prewarm(warm_model=warm_model)
                if warm.get("ok"):
                    with state.lock:
                        state.warmed_models.add(str(warm.get("model") or warm_model).strip())
            document_features = ocr_local_inference.classify_document_features(local_path, goal=goal, max_pages=max_pages)
            visual_controls = list(document_features.get("visual_controls") or [])
            feature_detection = {
                "mode": str(document_features.get("mode") or "lightweight_heuristic"),
                "ok": bool(document_features.get("ok")),
                "confidence_reason": str(document_features.get("confidence_reason") or ""),
                "warnings": list(document_features.get("warnings") or []),
            }
            doc_class = _classify_doc_type(local_path, goal, document_features=document_features)
            if explicit_lane and requested_lane_strict:
                warnings = list(feature_detection.get("warnings") or [])
                if doc_class == "handwritten" and model not in {"gb10_got_ocr2", "gb10_qwen_ocr"}:
                    warnings.append("detected_handwriting_but_requested_lane_is_strict")
                if doc_class == "chart_or_figure" and model not in {"gb10_qwen_ocr", "gb10_paddleocr_vl"}:
                    warnings.append("detected_chart_or_figure_but_requested_lane_is_strict")
                if doc_class == "forms_or_checkboxes" and model not in {"gb10_qwen_ocr", "gb10_got_ocr2", "rapidocr", "tesseract"}:
                    warnings.append("detected_visual_controls_but_requested_lane_is_strict")
                feature_detection["warnings"] = warnings
            if explicit_lane and requested_lane_strict:
                chain = [model]
            else:
                chain = _engine_chain(doc_class, route_mode, model, document_features=document_features)
            if not gb10_default_enabled() and not (explicit_lane and requested_lane_strict):
                chain = [engine for engine in chain if not str(engine).startswith("gb10_")] + ["rapidocr", "tesseract"]
            attempts: list[Dict[str, Any]] = []
            best: Dict[str, Any] = {}
            best_score = -1.0
            fallback_reason = ""
            target_score = 0.72 if route_mode != "balanced" else 0.60
            for engine in chain:
                row = _run_engine(
                    local_path,
                    goal=goal,
                    engine=engine,
                    max_pages=max_pages,
                    max_chars=max_chars,
                    route_mode=route_mode,
                    region_retry_policy=region_retry_policy,
                    bypass_ready_gate=bypass_ready_gate,
                )
                q = dict(row.get("quality") or {})
                attempt_row = {
                    "engine": str(row.get("engine") or engine),
                    "model": str(row.get("model") or ""),
                    "ok": bool(row.get("ok")),
                    "latency_ms": float(row.get("latency_ms") or 0.0),
                    "chars": int(q.get("chars") or len(str(row.get("text") or ""))),
                    "confidence": row.get("confidence"),
                    "structure_score": float(q.get("structure_score") or 0.0),
                    "numeric_fidelity_score": float(q.get("numeric_fidelity_score") or 0.0),
                    "lane_signature": str(row.get("lane_signature") or ""),
                    "fallback_reason": str(row.get("fallback_reason") or row.get("error") or ""),
                }
                attempts.append(attempt_row)
                score = float(q.get("score") or 0.0)
                if row.get("ok") and score > best_score:
                    best = row
                    best_score = score
                native_chars = int(q.get("chars") or len(str(row.get("text") or "")))
                if (
                    engine == "native_pdf_text"
                    and row.get("ok")
                    and doc_class in {"digital_pdf", "table_dense"}
                    and native_chars >= 300
                    and (
                        float(q.get("structure_score") or 0.0) >= 0.05
                        or float(q.get("numeric_fidelity_score") or 0.0) >= 0.20
                    )
                ):
                    best = row
                    best_score = score
                    break
                if row.get("ok") and score >= target_score and float(q.get("numeric_fidelity_score") or 0.0) >= 0.50:
                    best = row
                    best_score = score
                    break
                if not row.get("ok") and not fallback_reason:
                    fallback_reason = str(row.get("fallback_reason") or row.get("error") or "")

            if not best:
                failure = _normalize_result(
                    {"ok": False, "engine": "", "error": "all_engines_failed"},
                    attempts=attempts,
                    chain=chain,
                    route_mode=route_mode,
                    doc_class=doc_class,
                    fallback_reason=fallback_reason or "all_engines_failed",
                    document_features=document_features,
                    visual_controls=visual_controls,
                    feature_detection=feature_detection,
                )
                with state.lock:
                    state.failures += 1
                    state.last_error = str(failure.get("error") or "all_engines_failed")
                    state.processed += 1
                    state.last_heartbeat = _utc_iso()
                return jsonify(failure), 422

            response = _normalize_result(
                best,
                attempts=attempts,
                chain=chain,
                route_mode=route_mode,
                doc_class=doc_class,
                fallback_reason=fallback_reason,
                document_features=document_features,
                visual_controls=visual_controls,
                feature_detection=feature_detection,
            )
            response["ok"] = bool(response.get("ok"))
            response["provider"] = "OCR97_gb10_gateway"
            response["queue_depth"] = state.queued
            response["running"] = state.running
            response["warm_hint"] = warm_hint
            response["requested_lane_strict"] = bool(requested_lane_strict)
            response["bypass_ready_gate"] = bool(bypass_ready_gate)
            _low_thresh = float(os.getenv("OCR97_LOW_CONFIDENCE_THRESHOLD", "0.45"))
            _high_thresh = float(os.getenv("OCR97_HIGH_CONFIDENCE_THRESHOLD", "0.72"))
            response["low_confidence"] = best_score < _low_thresh
            response["confidence_tier"] = (
                "high" if best_score >= _high_thresh
                else "medium" if best_score >= _low_thresh
                else "low"
            )
            if response["low_confidence"]:
                _rej: list[str] = []
                _q = dict(response.get("quality") or {})
                if int(_q.get("chars") or 0) < 30:
                    _rej.append("very_short_extraction")
                if float(_q.get("structure_score") or 0.0) < 0.10:
                    _rej.append("no_structure_detected")
                if len(attempts) > 1 and not any(a.get("ok") for a in attempts[:-1]):
                    _rej.append("all_primary_engines_failed")
                response["rejection_reason"] = ",".join(_rej) if _rej else "composite_score_below_threshold"
            else:
                response["rejection_reason"] = None
            if region_retry_policy:
                response["region_retry_policy"] = region_retry_policy
            with state.lock:
                state.processed += 1
                state.last_heartbeat = _utc_iso()
            return jsonify(response), 200 if response.get("ok") else 422
        except Exception as exc:
            with state.lock:
                state.failures += 1
                state.last_error = f"{type(exc).__name__}:{exc}"
                state.last_heartbeat = _utc_iso()
            return jsonify({"ok": False, "error": f"ocr_extract_failed:{type(exc).__name__}:{exc}"}), 500
        finally:
            with state.lock:
                state.running = max(0, state.running - 1)
            state.semaphore.release()

    @app.route("/ocr/prewarm", methods=["POST"])
    def ocr_prewarm():
        if _auth_required() and not _auth_ok():
            return jsonify({"ok": False, "error": "unauthorized"}), 401
        json_payload = request.get_json(silent=True) if request.is_json else {}
        warm_model = str(request.form.get("warm_model") or ((json_payload or {}).get("warm_model") if isinstance(json_payload, dict) else "") or "").strip()
        result = _run_prewarm(warm_model=warm_model)
        if result.get("ok"):
            with state.lock:
                if str(result.get("model") or "").strip():
                    state.warmed_models.add(str(result.get("model") or "").strip())
        return jsonify({"ok": bool(result.get("ok")), "provider": "OCR97_gb10_gateway", "ts": _utc_iso(), "prewarm": result}), 200 if result.get("ok") else 422

    @app.route("/ocr/smoke", methods=["GET", "POST"])
    def ocr_smoke():
        if _auth_required() and not _auth_ok():
            return jsonify({"ok": False, "error": "unauthorized"}), 401
        if request.method == "POST":
            report = _run_smoke_suite()
            return jsonify({"ok": bool(report.get("ok")), "provider": "OCR97_gb10_gateway", "ts": _utc_iso(), "smoke": report}), 200 if report.get("ok") else 422
        report = _load_smoke_report()
        status = 200 if report else 404
        return jsonify({"ok": bool(report.get("ok")), "provider": "OCR97_gb10_gateway", "ts": _utc_iso(), "smoke": report}), status

    @app.route("/ocr/health", methods=["GET"])
    def ocr_health():
        if _auth_required() and not _auth_ok():
            return jsonify({"ok": False, "error": "unauthorized"}), 401
        deep_check = _truthy(request.args.get("deep"), default=False)
        if deep_check:
            try:
                ollama_resp = requests.get(f"{ollama_url}/api/tags", timeout=4)
                ollama_ok = bool(ollama_resp.ok)
                ollama_data = ollama_resp.json() if ollama_ok else {}
            except Exception as exc:
                ollama_ok = False
                ollama_data = {"error": f"{type(exc).__name__}:{exc}"}
        else:
            ollama_ok = None
            ollama_data = {}
        with state.lock:
            engine_rows = {
                "gb10_paddleocr_vl": _engine_snapshot("gb10_paddleocr_vl"),
                "gb10_got_ocr2": _engine_snapshot("gb10_got_ocr2"),
                "mineru2_5": _engine_snapshot("mineru2_5"),
                "olmocr2": _engine_snapshot("olmocr2"),
                "gb10_qwen_ocr": _engine_snapshot("gb10_qwen_ocr"),
                "local_image_best": _engine_snapshot("local_image_best"),
                "local_image_preprocessed_best": _engine_snapshot("local_image_preprocessed_best"),
                "rapidocr": _engine_snapshot("rapidocr"),
                "tesseract": _engine_snapshot("tesseract"),
            }
            install_runtime = _safe_json_load(install_manifest_path) or install_manifest
            smoke_runtime = _load_smoke_report()
            payload = {
                "ok": True,
                "provider": "OCR97_gb10_gateway",
                "ts": _utc_iso(),
                "max_concurrency": state.max_concurrency,
                "running": state.running,
                "queue_depth": state.queued,
                "processed": state.processed,
                "failures": state.failures,
                "last_error": state.last_error,
                "last_heartbeat": state.last_heartbeat,
                "warmed_models": sorted(model for model in state.warmed_models if model),
                "diagnostic_mode": "lightweight_no_model_import",
                "ollama": {
                    "ok": ollama_ok,
                    "checked": bool(deep_check),
                    "url": ollama_url,
                    "models": [str((item or {}).get("name") or "") for item in list(ollama_data.get("models") or [])],
                },
                "engines": {
                    "gb10_paddleocr_vl": bool(engine_rows["gb10_paddleocr_vl"]["ready"]),
                    "gb10_got_ocr2": bool(engine_rows["gb10_got_ocr2"]["ready"]),
                    "mineru2_5": bool(engine_rows["mineru2_5"]["ready"]),
                    "olmocr2": bool(engine_rows["olmocr2"]["ready"]),
                    "gb10_qwen_ocr": bool(engine_rows["gb10_qwen_ocr"]["ready"]),
                    "local_image_best": bool(engine_rows["local_image_best"]["ready"]),
                    "local_image_preprocessed_best": bool(engine_rows["local_image_preprocessed_best"]["ready"]),
                    "rapidocr": bool(engine_rows["rapidocr"]["ready"]),
                    "tesseract": bool(engine_rows["tesseract"]["ready"]),
                },
                "feature_detection": {
                    "document_feature_classifier": {
                        "ready": bool(PIL_AVAILABLE),
                        "mode": "lightweight_heuristic",
                        "runtime_loaded": False,
                        "pillow_available": bool(PIL_AVAILABLE),
                        "opencv_available": bool(CV2_AVAILABLE),
                    },
                    "visual_control_detector": {
                        "ready": bool(PIL_AVAILABLE and CV2_AVAILABLE),
                        "mode": "opencv_contour_heuristic",
                        "runtime_loaded": False,
                        "pillow_available": bool(PIL_AVAILABLE),
                        "opencv_available": bool(CV2_AVAILABLE),
                    },
                    "layout_model_classifier": {
                        "ready": False,
                        "enabled": _truthy(os.getenv("OCR97_LAYOUT_MODEL_ENABLED", "0"), default=False),
                        "runtime_loaded": False,
                        "mode": "optional_model_disabled_by_default",
                    },
                },
                "engine_details": engine_rows,
                "install_metadata": install_runtime,
                "warm_state": dict(warm_state),
                "smoke": smoke_runtime,
                "slo_policy": {
                    "mode": "balanced",
                    "window_hours": metrics_window_hours,
                    "timeout_rate_max": 0.05,
                    "fallback_rate_max": 0.20,
                    "p95_caps_by_class": p95_caps_by_class,
                    "smoke_required": smoke_required,
                },
                "got_service": {
                    "extract": "/ocr/got/extract",
                    "health": "/ocr/got/health",
                    "runtime_loaded": bool(_GOT_RUNTIME.get("model") is not None and _GOT_RUNTIME.get("processor") is not None),
                    "model_id": _got_model_id(),
                    "last_error": str(_GOT_RUNTIME.get("last_error") or ""),
                },
                "phase2_services": {
                    "finbert_eval": "/ocr/finbert/eval",
                    "finbert_health": "/ocr/finbert/health",
                    "table_reconstruct": "/ocr/table/reconstruct",
                    "table_health": "/ocr/table/health",
                    "docunet_rectify": "/ocr/docunet/rectify",
                    "docunet_health": "/ocr/docunet/health",
                    "realesrgan_upscale": "/ocr/realesrgan/upscale",
                    "realesrgan_health": "/ocr/realesrgan/health",
                    "smoke": "/ocr/smoke",
                    "finbert_runtime_loaded": bool(_FINBERT_RUNTIME.get("pipeline") is not None),
                    "table_runtime_loaded": bool(_TABLEFORMER_RUNTIME.get("processor") is not None and _TABLEFORMER_RUNTIME.get("model") is not None),
                    "docunet_runtime_loaded": bool(_DOCUNET_RUNTIME.get("pipeline") is not None),
                    "realesrgan_runtime_loaded": bool(_REALESRGAN_RUNTIME.get("upsampler") is not None),
                    "finbert_model_id": _finbert_model_id(),
                    "table_model_id": _tableformer_model_id(),
                    "docunet_model_id": _docunet_model_id(),
                    "realesrgan_model_id": _realesrgan_model_name(),
                    "finbert_last_error": str(_FINBERT_RUNTIME.get("last_error") or ""),
                    "table_last_error": str(_TABLEFORMER_RUNTIME.get("last_error") or ""),
                    "docunet_last_error": str(_DOCUNET_RUNTIME.get("last_error") or ""),
                    "realesrgan_last_error": str(_REALESRGAN_RUNTIME.get("last_error") or ""),
                },
            }
        return jsonify(payload), 200

    @app.route("/ocr/capabilities", methods=["GET"])
    def ocr_capabilities():
        if _auth_required() and not _auth_ok():
            return jsonify({"ok": False, "error": "unauthorized"}), 401
        install_runtime = _safe_json_load(install_manifest_path) or install_manifest
        smoke_runtime = _load_smoke_report()
        engines = [
            {
                "class": "feature_detection",
                "name": "document_feature_classifier",
                "ready": bool(PIL_AVAILABLE),
                "reason": "lightweight_pil_opencv_heuristics" if PIL_AVAILABLE else "pil_unavailable",
                "mode": "lightweight_heuristic",
                "runtime_loaded": False,
            },
            {
                "class": "feature_detection",
                "name": "visual_control_detector",
                "ready": bool(PIL_AVAILABLE and CV2_AVAILABLE),
                "reason": "opencv_contour_heuristic" if PIL_AVAILABLE and CV2_AVAILABLE else "pil_or_opencv_unavailable",
                "mode": "opencv_contour_heuristic",
                "runtime_loaded": False,
            },
            {
                "class": "feature_detection",
                "name": "layout_model_classifier",
                "ready": False,
                "reason": "optional_layout_model_disabled_by_default",
                "enabled": _truthy(os.getenv("OCR97_LAYOUT_MODEL_ENABLED", "0"), default=False),
                "runtime_loaded": False,
            },
            {"class": "layout", **_engine_snapshot("gb10_paddleocr_vl")},
            {"class": "dense_scan", **_engine_snapshot("gb10_got_ocr2")},
            {"class": "structure_parser", **_engine_snapshot("mineru2_5")},
            {"class": "linearization", **_engine_snapshot("olmocr2")},
            {"class": "semantic_cleanup", **_engine_snapshot("gb10_qwen_ocr")},
            {"class": "image_router", **_engine_snapshot("local_image_best")},
            {"class": "image_preprocessor", **_engine_snapshot("local_image_preprocessed_best")},
            {"class": "compat_fallback", **_engine_snapshot("rapidocr")},
            {"class": "compat_fallback", **_engine_snapshot("tesseract")},
        ]
        return jsonify(
            {
                "ok": True,
                "provider": "OCR97_gb10_gateway",
                "diagnostic_mode": "lightweight_no_model_import",
                "route_mode_default": os.getenv("OCR97_OCR_ROUTE_MODE", "quality_first"),
                "max_concurrency": state.max_concurrency,
                "timeout_sec_default": default_timeout_sec,
                "queue_depth": state.queued,
                "running": state.running,
                "engines": engines,
                "install_metadata": install_runtime,
                "warm_state": dict(warm_state),
                "smoke": smoke_runtime,
                "slo_policy": {
                    "mode": "balanced",
                    "window_hours": metrics_window_hours,
                    "timeout_rate_max": 0.05,
                    "fallback_rate_max": 0.20,
                    "p95_caps_by_class": p95_caps_by_class,
                    "smoke_required": smoke_required,
                },
                "doc_classes": ["digital_pdf", "scanned_pdf", "photo", "handwritten", "table_dense", "chart_or_figure", "forms_or_checkboxes"],
                "routes": {
                    "extract": "/ocr/extract",
                    "health": "/ocr/health",
                    "capabilities": "/ocr/capabilities",
                    "prewarm": "/ocr/prewarm",
                    "smoke": "/ocr/smoke",
                    "got_extract": "/ocr/got/extract",
                    "got_health": "/ocr/got/health",
                    "finbert_eval": "/ocr/finbert/eval",
                    "finbert_health": "/ocr/finbert/health",
                    "table_reconstruct": "/ocr/table/reconstruct",
                    "table_health": "/ocr/table/health",
                    "docunet_rectify": "/ocr/docunet/rectify",
                    "docunet_health": "/ocr/docunet/health",
                    "realesrgan_upscale": "/ocr/realesrgan/upscale",
                    "realesrgan_health": "/ocr/realesrgan/health",
                },
                "lifecycle": {
                    "supports_prewarm": True,
                    "max_concurrency": state.max_concurrency,
                    "queue_depth": state.queued,
                    "timeout_sec_default": default_timeout_sec,
                    "heartbeat": state.last_heartbeat,
                },
            }
        ), 200
