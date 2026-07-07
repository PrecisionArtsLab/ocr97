from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Any, Dict, Iterable, List, Mapping, Optional


_MONEY_RE = re.compile(r"(?:\$|USD|RM|MYR)?\s*\(?-?\d[\d,\s<]*(?:[.\-]\d{1,4})?%?\)?", re.IGNORECASE)
_ISO_DATE_RE = re.compile(r"\b(?:19|20)\d{2}[-/.]\d{1,2}[-/.]\d{1,2}\b")
_US_DATE_RE = re.compile(r"\b\d{1,2}[-/.]\d{1,2}[-/.](?:\d{2}|\d{4})\b")
_MONTH_DATE_RE = re.compile(
    r"\b("
    r"jan(?:uary)?|feb(?:ruary)?|mar(?:ch)?|apr(?:il)?|may|jun(?:e)?|jul(?:y)?|"
    r"aug(?:ust)?|sep(?:t(?:ember)?)?|oct(?:ober)?|nov(?:ember)?|dec(?:ember)?"
    r")\s+\d{1,2},?\s+(?:19|20)\d{2}\b",
    re.IGNORECASE,
)
_ADDRESS_HINT_RE = re.compile(
    r"\b(?:street|st\.?|avenue|ave\.?|road|rd\.?|lane|ln\.?|drive|dr\.?|boulevard|blvd\.?|"
    r"jalan|lot|suite|ste\.?|unit|po box|city|state|zip|postal)\b|\b\d{5}(?:-\d{4})?\b",
    re.IGNORECASE,
)
_INTERVENING_LABEL_RE = re.compile(r"\b[A-Z][A-Za-z0-9 ]{2,35}\s*:")
_MONTHS = {
    "jan": 1,
    "january": 1,
    "feb": 2,
    "february": 2,
    "mar": 3,
    "march": 3,
    "apr": 4,
    "april": 4,
    "may": 5,
    "jun": 6,
    "june": 6,
    "jul": 7,
    "july": 7,
    "aug": 8,
    "august": 8,
    "sep": 9,
    "sept": 9,
    "september": 9,
    "oct": 10,
    "october": 10,
    "nov": 11,
    "november": 11,
    "dec": 12,
    "december": 12,
}


@dataclass(frozen=True)
class FieldCandidate:
    field: str
    value: str
    normalized_value: str
    source_line: str
    line_index: int
    confidence: float
    reason: str

    def as_dict(self) -> Dict[str, Any]:
        return {
            "field": self.field,
            "value": self.value,
            "normalized_value": self.normalized_value,
            "source_line": self.source_line,
            "line_index": self.line_index,
            "confidence": round(float(self.confidence), 3),
            "reason": self.reason,
        }


def normalize_space(value: Any) -> str:
    return " ".join(str(value or "").strip().split())


def normalize_key(value: Any) -> str:
    return re.sub(r"[^a-z0-9]+", " ", normalize_space(value).lower()).strip()


def normalize_number(value: Any) -> str:
    raw = str(value or "").strip()
    match = _MONEY_RE.search(raw)
    if not match:
        return ""
    token = match.group(0)
    negative = token.startswith("(") and token.endswith(")")
    token = re.sub(r"(?<=\d)<(?=\d{3}(?:\D|$))", ",", token)
    token = re.sub(r"(?<=\d)-(?=\d{2}(?:\D|$))", ".", token)
    token = token.replace("$", "").replace("USD", "").replace("RM", "").replace("MYR", "")
    token = token.replace(",", "").replace(" ", "").replace("(", "").replace(")", "").replace("%", "")
    try:
        number = float(token)
    except Exception:
        return ""
    if negative:
        number = -number
    if abs(number - round(number)) < 0.000001:
        return str(int(round(number)))
    return f"{number:.4f}".rstrip("0").rstrip(".")


def normalize_date(value: Any) -> str:
    raw = str(value or "").strip()
    if not raw:
        return ""
    month_date_raw = _MONTH_DATE_RE.search(raw)
    if month_date_raw:
        parts = re.split(r"[\s,]+", month_date_raw.group(0).strip())
        if len(parts) >= 3:
            month = _MONTHS.get(parts[0].lower()[:3], _MONTHS.get(parts[0].lower()))
            day = int(parts[1])
            year = int(parts[2])
            if month and 1900 <= year <= 2099 and 1 <= day <= 31:
                return f"{year:04d}-{month:02d}-{day:02d}"
    clean = raw.translate(str.maketrans({"O": "0", "o": "0", "I": "1", "l": "1", "|": "1"}))
    iso = _ISO_DATE_RE.search(clean)
    if iso:
        parts = re.split(r"[-/.]", iso.group(0))
        year, month, day = int(parts[0]), int(parts[1]), int(parts[2])
        if 1900 <= year <= 2099 and 1 <= month <= 12 and 1 <= day <= 31:
            return f"{year:04d}-{month:02d}-{day:02d}"
    month_date = _MONTH_DATE_RE.search(clean)
    if month_date:
        parts = re.split(r"[\s,]+", month_date.group(0).strip())
        if len(parts) >= 3:
            month = _MONTHS.get(parts[0].lower()[:3], _MONTHS.get(parts[0].lower()))
            day = int(parts[1])
            year = int(parts[2])
            if month and 1900 <= year <= 2099 and 1 <= day <= 31:
                return f"{year:04d}-{month:02d}-{day:02d}"
    date = _US_DATE_RE.search(clean)
    if date:
        first, second, year_raw = re.split(r"[-/.]", date.group(0))
        year = int(year_raw)
        if year < 100:
            year += 2000
        a, b = int(first), int(second)
        # Prefer month/day for US-like business docs; fall back to day/month when only that is valid.
        if 1 <= a <= 12 and 1 <= b <= 31:
            month, day = a, b
        elif 1 <= b <= 12 and 1 <= a <= 31:
            month, day = b, a
        else:
            return ""
        if 1900 <= year <= 2099:
            return f"{year:04d}-{month:02d}-{day:02d}"
    return ""


def normalize_for_type(value: Any, field_type: str) -> str:
    kind = str(field_type or "text").lower().strip()
    if kind in {"money", "number", "percent"}:
        return normalize_number(value)
    if kind == "date":
        return normalize_date(value)
    return normalize_key(value)


def _lines(text: str) -> List[str]:
    return [normalize_space(line) for line in str(text or "").splitlines() if normalize_space(line)]


def _label_score(line: str, aliases: Iterable[Any]) -> float:
    key = normalize_key(line)
    best = 0.0
    for alias in aliases:
        alias_text = normalize_space(alias)
        alias_key = normalize_key(alias)
        if not alias_key:
            continue
        exact_pattern = rf"(?<![A-Za-z0-9]){_alias_pattern(alias_text)}(?![A-Za-z0-9])"
        exact_present = bool(re.search(exact_pattern, line, flags=re.IGNORECASE))
        if exact_present:
            best = max(best, 0.50 if key.startswith(alias_key) else 0.42)
            continue
        if len(alias_key) <= 7 and not re.search(r"[:#=/]", str(alias or "")):
            continue
        if re.search(r"[:#=/]", str(alias or "")):
            continue
        if alias_key in key:
            best = max(best, 0.35)
        if key.startswith(alias_key):
            best = max(best, 0.50)
    return best


def _alias_pattern(alias_text: Any) -> str:
    parts: List[str] = []
    for char in str(alias_text or ""):
        lower = char.lower()
        if char.isspace():
            parts.append(r"\s*")
        elif lower in {"l", "i"}:
            parts.append(r"[li1|]")
        elif lower == "o":
            parts.append(r"[o0qQ]")
        elif lower == "s":
            parts.append(r"[s5]")
        elif lower == "v":
            parts.append(r"[vV]")
        else:
            parts.append(re.escape(char))
    return "".join(parts)


def _label_proximity_score(line: str, aliases: Iterable[Any], value_start: int) -> float:
    best = 0.0
    for alias in aliases:
        alias_text = normalize_space(alias)
        if not alias_text:
            continue
        pattern = rf"(?<![A-Za-z0-9]){_alias_pattern(alias_text)}(?![A-Za-z0-9])"
        for match in re.finditer(pattern, line, flags=re.IGNORECASE):
            if match.end() > value_start:
                continue
            gap = line[match.end() : value_start]
            distance = len(gap.strip())
            if distance <= 3:
                score = 0.42
            elif distance <= 16:
                score = 0.32
            elif distance <= 45:
                score = 0.18
            else:
                score = 0.06
            if _INTERVENING_LABEL_RE.search(gap):
                score -= 0.20
            best = max(best, score)
    return max(0.0, best)


def field_candidates(text: str, field: Mapping[str, Any]) -> List[Dict[str, Any]]:
    name = str(field.get("name") or "")
    aliases = list(field.get("aliases") or [name])
    field_type = str(field.get("type") or "text")
    kind = field_type.lower().strip()
    rows = _lines(text)
    candidates: List[FieldCandidate] = []

    for idx, line in enumerate(rows):
        label_bonus = _label_score(line, aliases)
        if kind in {"money", "number", "percent"}:
            for match in _MONEY_RE.finditer(line):
                raw = match.group(0).strip()
                normalized = normalize_number(raw)
                if not normalized:
                    continue
                proximity_bonus = _label_proximity_score(line, aliases, match.start())
                confidence = min(0.98, 0.20 + label_bonus + proximity_bonus + (0.10 if "$" in raw or "%" in raw else 0.0))
                if re.search(r"\b(?:subtotal|tax|change|unit price|qty|quantity)\b", line, re.IGNORECASE) and normalize_key(name) in {"total", "amount due", "balance due"}:
                    confidence -= 0.20
                if confidence >= 0.25:
                    reason = "numeric candidate from label/context"
                    if proximity_bonus:
                        reason = "numeric candidate near requested label"
                    candidates.append(FieldCandidate(name, raw, normalized, line, idx, max(0.05, confidence), reason))
        elif kind == "date":
            for pattern in (_ISO_DATE_RE, _US_DATE_RE, _MONTH_DATE_RE):
                for match in pattern.finditer(line):
                    raw = match.group(0).strip()
                    normalized = normalize_date(raw)
                    if not normalized:
                        continue
                    proximity_bonus = _label_proximity_score(line, aliases, match.start())
                    iso_bonus = 0.16 if _ISO_DATE_RE.fullmatch(raw) else 0.0
                    confidence = min(0.98, 0.30 + label_bonus + proximity_bonus + iso_bonus + (0.08 if re.search(r"\b(date|due|issued|paid)\b", line, re.IGNORECASE) else 0.0))
                    reason = "date candidate from label/context"
                    if proximity_bonus:
                        reason = "date candidate near requested label"
                    candidates.append(FieldCandidate(name, raw, normalized, line, idx, confidence, reason))
        elif normalize_key(name) == "address" or "address" in {normalize_key(alias) for alias in aliases}:
            if _ADDRESS_HINT_RE.search(line):
                block = [line]
                for follow in rows[idx + 1 : idx + 3]:
                    if re.search(r"\b(?:date|total|tax|invoice|receipt|phone|tel|email)\b", follow, re.IGNORECASE):
                        break
                    if _ADDRESS_HINT_RE.search(follow):
                        block.append(follow)
                value = " ".join(block)
                candidates.append(FieldCandidate(name, value, normalize_key(value), line, idx, 0.65 + min(0.2, len(block) * 0.05), "address block candidate"))
        else:
            label_bonus = _label_score(line, aliases)
            if label_bonus <= 0:
                continue
            parts = re.split(r"[:=\|]|\s{2,}", line, maxsplit=1)
            value = normalize_space(parts[-1] if len(parts) > 1 else line)
            if len(parts) == 1:
                for alias in aliases:
                    pattern = rf"(?<![A-Za-z0-9]){_alias_pattern(alias)}(?![A-Za-z0-9])"
                    match = re.search(pattern, line, flags=re.IGNORECASE)
                    if match:
                        tail = normalize_space(line[match.end() :])
                        if tail:
                            value = tail
                            break
            if value and normalize_key(value) not in {normalize_key(alias) for alias in aliases}:
                candidates.append(FieldCandidate(name, value, normalize_key(value), line, idx, min(0.90, 0.35 + label_bonus), "text candidate from label"))

    by_value: Dict[str, FieldCandidate] = {}
    for candidate in candidates:
        existing = by_value.get(candidate.normalized_value)
        if existing is None or candidate.confidence > existing.confidence:
            by_value[candidate.normalized_value] = candidate
    rows = [item.as_dict() for item in sorted(by_value.values(), key=lambda row: (-row.confidence, row.line_index, row.value))[:8]]
    try:
        from .field_ranker import rerank_candidates

        rows = rerank_candidates(rows, {"name": name, "aliases": aliases, "type": field_type})
    except Exception:
        pass
    return rows[:8]


def classify_failure(*, expected: Any, candidates: Iterable[Mapping[str, Any]], matched: bool, field_type: str) -> str:
    if matched:
        return "matched"
    rows = list(candidates or [])
    if not rows:
        return "field_not_found"
    expected_norm = normalize_for_type(expected, field_type)
    if not expected_norm:
        return "ground_truth_unusable"
    for row in rows:
        if str(row.get("normalized_value") or "") == expected_norm:
            return "candidate_found_but_not_selected"
    if str(field_type or "").lower() in {"money", "number", "percent"}:
        return "numeric_candidate_wrong"
    if str(field_type or "").lower() == "date":
        return "date_candidate_wrong_or_ambiguous"
    return "text_candidate_wrong"
