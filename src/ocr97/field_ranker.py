from __future__ import annotations

import json
import os
import re
from functools import lru_cache
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Optional

FIELD_ALIASES: Dict[str, List[str]] = {
    "invoice_number": ["invoice number", "invoice"],
    "subtotal": ["subtotal"],
    "tax": ["tax"],
    "total": ["total", "amount due", "payment due"],
    "assets": ["assets"],
    "liabilities": ["liabilities"],
    "equity": ["equity"],
    "opening_balance": ["opening balance"],
    "deposits": ["deposits"],
    "closing_balance": ["closing balance"],
    "cash": ["cash"],
    "market_value": ["market value"],
    "gross_pay": ["gross pay"],
    "deductions": ["deductions"],
    "net_pay": ["net pay"],
    "agi": ["adjusted gross income"],
    "taxable_income": ["taxable income"],
    "tax_due": ["tax due"],
    "account": ["account"],
    "due_date": ["due date", "date"],
    "date": ["date", "receipt date", "statement date"],
    "revenue": ["revenue"],
    "cost": ["cost"],
    "margin": ["margin"],
    "principal": ["principal"],
    "interest_rate": ["interest rate"],
}

DEFAULT_WEIGHTS: Dict[str, float] = {
    "bias": 0.0,
    "candidate_confidence": 1.0,
    "near_requested_label": 0.7,
    "label_context": 0.2,
    "line_has_field_name": 0.15,
    "value_near_field_alias": 1.0,
    "immediate_after_field_alias": 0.75,
    "labels_between_alias_and_value": -0.5,
    "has_currency": 0.02,
    "money_type_without_currency": -1.5,
    "identifier_text_has_digit": 0.9,
    "identifier_text_without_digit": -1.2,
    "has_percent": 0.02,
    "type_percent_match": 0.4,
    "iso_date_value": 0.25,
    "line_is_table_like": -0.1,
}

_LABEL_BETWEEN_RE = re.compile(r"\b[A-Za-z][A-Za-z0-9 ]{2,35}\s*:")


def normalize_key(value: Any) -> str:
    return re.sub(r"[^a-z0-9]+", " ", " ".join(str(value or "").strip().lower().split())).strip()


def _alias_pattern(alias: Any) -> str:
    parts: List[str] = []
    for char in str(alias or ""):
        lower = char.lower()
        if char.isspace():
            parts.append(r"\s*")
        elif lower in {"l", "i"}:
            parts.append(r"[li1|]")
        elif lower == "o":
            parts.append(r"[o0]")
        elif lower == "s":
            parts.append(r"[s5]")
        elif lower == "v":
            parts.append(r"[vV]")
        else:
            parts.append(re.escape(char))
    return "".join(parts)


def _field_name(field: Mapping[str, Any]) -> str:
    return str(field.get("name") or field.get("field") or "")


def candidate_features(candidate: Mapping[str, Any], field: Mapping[str, Any]) -> Dict[str, float]:
    source_line = str(candidate.get("source_line") or "")
    reason = str(candidate.get("reason") or "")
    field_name = _field_name(field)
    field_type = str(field.get("type") or "text")
    line_key = normalize_key(source_line)
    value = str(candidate.get("value") or "")
    value_start = source_line.find(value)
    aliases = list(field.get("aliases") or []) or FIELD_ALIASES.get(field_name, [field_name.replace("_", " ")])
    best_distance_score = 0.0
    immediate_after_label = 0.0
    labels_between = 0.0
    for alias in aliases:
        alias_key = str(alias or "")
        if not alias_key or value_start < 0:
            continue
        pattern = rf"(?<![A-Za-z0-9]){_alias_pattern(alias_key)}(?![A-Za-z0-9])"
        for match in re.finditer(pattern, source_line, flags=re.IGNORECASE):
            if match.end() > value_start:
                continue
            gap = source_line[match.end() : value_start]
            distance = len(gap.strip())
            if distance <= 3:
                immediate_after_label = 1.0
            if _LABEL_BETWEEN_RE.search(gap):
                labels_between = 1.0
            best_distance_score = max(best_distance_score, 1.0 / (1.0 + float(distance)))
    return {
        "bias": 1.0,
        "candidate_confidence": float(candidate.get("confidence") or 0.0),
        "near_requested_label": 1.0 if "near requested label" in reason else 0.0,
        "label_context": 1.0 if "label/context" in reason else 0.0,
        "line_has_field_name": 1.0 if normalize_key(field_name).replace(" ", "") in line_key.replace(" ", "") else 0.0,
        "value_near_field_alias": best_distance_score,
        "immediate_after_field_alias": immediate_after_label,
        "labels_between_alias_and_value": labels_between,
        "has_currency": 1.0 if "$" in value else 0.0,
        "money_type_without_currency": 1.0 if field_type in {"money", "number"} and "$" not in value and "%" not in value else 0.0,
        "identifier_text_has_digit": 1.0 if field_name in {"invoice_number", "account"} and any(ch.isdigit() for ch in value) else 0.0,
        "identifier_text_without_digit": 1.0 if field_name in {"invoice_number", "account"} and not any(ch.isdigit() for ch in value) else 0.0,
        "has_percent": 1.0 if "%" in value else 0.0,
        "type_percent_match": 1.0 if field_type == "percent" and "%" in value else 0.0,
        "iso_date_value": 1.0 if re.fullmatch(r"\d{4}[-/]\d{1,2}[-/]\d{1,2}", value.strip()) else 0.0,
        "line_is_table_like": 1.0 if source_line.count("|") >= 2 else 0.0,
    }


def dot(weights: Mapping[str, float], features: Mapping[str, float]) -> float:
    return sum(float(weights.get(name, 0.0)) * float(value) for name, value in features.items())


def model_weights(model: Optional[Mapping[str, Any]] = None) -> Dict[str, float]:
    if not model:
        return dict(DEFAULT_WEIGHTS)
    return {str(key): float(value) for key, value in dict(model.get("weights") or DEFAULT_WEIGHTS).items()}


def load_model(path: Path) -> Dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"field_ranker_model_must_be_object:{path}")
    payload.setdefault("weights", dict(DEFAULT_WEIGHTS))
    return payload


@lru_cache(maxsize=8)
def _load_model_cached(path_text: str) -> Dict[str, Any]:
    return load_model(Path(path_text))


def configured_model() -> Optional[Dict[str, Any]]:
    path = os.getenv("OCR97_HELIX97_FIELD_RANKER_MODEL") or os.getenv("HELIX97_FIELD_RANKER_MODEL")
    if not path:
        return None
    model_path = Path(path).expanduser()
    if not model_path.exists():
        return None
    return _load_model_cached(str(model_path.resolve()))


def score_candidate(candidate: Mapping[str, Any], field: Mapping[str, Any], model: Optional[Mapping[str, Any]] = None) -> float:
    return dot(model_weights(model), candidate_features(candidate, field))


def rerank_candidates(
    candidates: Iterable[Mapping[str, Any]],
    field: Mapping[str, Any],
    *,
    model: Optional[Mapping[str, Any]] = None,
) -> List[Dict[str, Any]]:
    rows = [dict(candidate) for candidate in candidates]
    if not rows:
        return []
    chosen_model = model if model is not None else configured_model()
    if not chosen_model:
        return rows
    for row in rows:
        row["learned_rank_score"] = round(score_candidate(row, field, chosen_model), 6)
        row["learned_ranker_used"] = True
    return sorted(
        rows,
        key=lambda row: (
            -float(row.get("learned_rank_score") or 0.0),
            -float(row.get("confidence") or 0.0),
            int(row.get("line_index") or 0),
            str(row.get("value") or ""),
        ),
    )
