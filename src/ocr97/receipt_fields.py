from __future__ import annotations

import re
from collections import defaultdict
from difflib import SequenceMatcher
from typing import Any, Dict, Iterable, Mapping, Optional


_CORP_SUFFIX = r"(?:SDN\s*(?:BHD|BHO|BND|EHD|GHD)|S/B|S\.B\.|\$B|SB|ENTERPRISE|TRADING|MARKETING|RESTAURANT|MART|STORE|HYPERMARKET)"
_STOP_WORDS = {
    "TAX",
    "INVOICE",
    "RECEIPT",
    "CASH",
    "CUSTOMER",
    "DATE",
    "TOTAL",
    "SUBTOTAL",
    "CHANGE",
    "AMOUNT",
    "GST",
    "SST",
    "TEL",
    "PHONE",
    "FAX",
    "CO",
    "NO",
    "REG",
}


def normalize_receipt_key(value: Any) -> str:
    raw = str(value or "").upper()
    raw = raw.replace("$B", " SB ")
    raw = re.sub(r"\bS\s*/\s*B\b", " SB ", raw)
    raw = re.sub(r"\bS\.?\s*B\.?\b", " SB ", raw)
    raw = re.sub(r"\bSDN\s+(?:EHD|BHO|BND|GHD|8HD|BHD)\b", " SDN BHD ", raw)
    raw = re.sub(r"[^A-Z0-9]+", " ", raw)
    return " ".join(raw.split())


def normalize_receipt_date(value: Any) -> str:
    raw = str(value or "").strip()
    if not raw:
        return ""
    trans = str.maketrans(
        {
            "O": "0",
            "o": "0",
            "Q": "0",
            "D": "0",
            "I": "1",
            "l": "1",
            "|": "1",
            "S": "5",
            "s": "5",
            "B": "8",
            "e": "2",
            "Z": "2",
            "z": "2",
        }
    )
    clean = raw.translate(trans)
    match = re.search(r"(\d{1,2})\s*[/-]\s*(\d{1,2})\s*[/-]?\s*((?:20)?\d{2,4})", clean)
    if not match:
        match = re.search(r"(\d{4})\s*[/-]\s*(\d{1,2})\s*[/-]\s*(\d{1,2})", clean)
        if not match:
            return ""
        year, month, day = (int(match.group(1)), int(match.group(2)), int(match.group(3)))
    else:
        day, month, year = (int(match.group(1)), int(match.group(2)), int(match.group(3)))
        if year < 100:
            year += 2000
    if 2035 < year <= 2099:
        year = 2010 + (year % 10)
    if not (1 <= day <= 31 and 1 <= month <= 12 and 2010 <= year <= 2035):
        return ""
    return f"{day:02d}/{month:02d}/{year:04d}"


def receipt_date_variants(value: Any) -> set[str]:
    normalized = normalize_receipt_date(value)
    if not normalized:
        return set()
    day, month, year = normalized.split("/")
    return {
        normalized,
        f"{day}-{month}-{year}",
        f"{day}.{month}.{year}",
        f"{int(day)}/{int(month)}/{year}",
        f"{day}/{month}/{year[-2:]}",
    }


def _normalize_line(value: Any) -> str:
    line = str(value or "").replace("\t", " ").replace("|", " ")
    line = line.replace("\\", " ").replace("_", " ")
    return " ".join(line.split())


def _line_tokens(line: str) -> list[str]:
    return re.findall(r"[A-Z0-9$./&'-]+", line.upper())


def _company_score(value: str, *, line_index: int, source_score: float) -> float:
    key = normalize_receipt_key(value)
    tokens = key.split()
    if not tokens:
        return 0.0
    alpha_tokens = [token for token in tokens if re.search(r"[A-Z]", token)]
    suffix_bonus = 8.0 if re.search(r"\b(?:SDN BHD|SB|ENTERPRISE|TRADING|MARKETING|RESTAURANT|MART|STORE)\b", key) else 0.0
    stop_penalty = sum(3.0 for token in tokens if token in _STOP_WORDS)
    address_penalty = 8.0 if re.search(r"\b(?:JALAN|LOT|NO|KM|BANDAR|TAMAN|SELANGOR|JOHOR|KUALA|IPOH)\b", key) else 0.0
    length_penalty = max(0.0, (len(tokens) - 7) * 1.5)
    top_bonus = max(0.0, 8.0 - (line_index * 1.2))
    return max(0.0, len(alpha_tokens) * 2.0 + suffix_bonus + top_bonus + min(10.0, source_score / 12.0) - stop_penalty - address_penalty - length_penalty)


def _company_candidates_from_text(text: str) -> Iterable[tuple[str, int]]:
    lines = [_normalize_line(line) for line in str(text or "").splitlines()]
    lines = [line for line in lines if line]
    for idx, line in enumerate(lines[:12]):
        upper = line.upper().replace("$B", " SB ")
        upper = re.sub(r"\bS\s*/\s*B\b", " SB ", upper)
        upper = re.sub(r"\bSDN\s+(?:EHD|BHO|BND|GHD|8HD|BHD)\b", " SDN BHD ", upper)
        suffix_match = re.search(rf"([A-Z0-9$&./' -]{{2,90}}?)\s+({_CORP_SUFFIX})\b", upper)
        if suffix_match:
            candidate = f"{suffix_match.group(1)} {suffix_match.group(2)}"
            yield _trim_company_candidate(candidate), idx
        registration_match = re.search(
            r"\b(?:DE|THE)?\s*([A-Z][A-Z&' -]{5,60}?)\s+(?:JM|ROC|CO\s*NO|COMPANY\s*REG|REG\s*NO|NO\.?\s*\d|LOT|JALAN|BANDAR|TAMAN)\b",
            upper,
        )
        if registration_match:
            yield _trim_company_candidate(registration_match.group(1)), idx
        tokens = _line_tokens(upper)
        if not tokens:
            continue
        if idx <= 3 and 1 <= len(tokens) <= 7 and not any(token in _STOP_WORDS for token in tokens[:2]):
            if sum(1 for token in tokens if re.search(r"[A-Z]", token)) >= 2:
                yield _trim_company_candidate(" ".join(tokens)), idx


def _trim_company_candidate(candidate: str) -> str:
    key = normalize_receipt_key(candidate)
    tokens = key.split()
    while tokens and (tokens[0] in _STOP_WORDS or re.fullmatch(r"\d+", tokens[0]) or tokens[0] in {"YY", "Y"}):
        tokens.pop(0)
    suffix_positions = [
        idx
        for idx, token in enumerate(tokens)
        if token in {"SB", "ENTERPRISE", "TRADING", "MARKETING", "RESTAURANT", "MART", "STORE"} or (token == "BHD" and idx > 0 and tokens[idx - 1] == "SDN")
    ]
    if suffix_positions:
        end = suffix_positions[0] + 1
        if tokens[suffix_positions[0]] == "BHD" and suffix_positions[0] > 0:
            start = max(0, suffix_positions[0] - 4)
            tokens = tokens[start:end]
        else:
            start = max(0, suffix_positions[0] - 5)
            tokens = tokens[start:end]
    return " ".join(tokens)


def _date_candidates_from_text(text: str) -> Iterable[tuple[str, int]]:
    lines = [_normalize_line(line) for line in str(text or "").splitlines()]
    patterns = [
        r"([0-9OoQDISsl|B]{1,2})\s*[/-]\s*([0-9OoQDISsl|Be]{1,2})\s*[/-]?\s*((?:20)?[0-9OoQDISsl|B]{2,4})",
        r"((?:20)[0-9OoQDISsl|B]{2})\s*[/-]\s*([0-9OoQDISsl|B]{1,2})\s*[/-]\s*([0-9OoQDISsl|B]{1,2})",
    ]
    for idx, line in enumerate(lines):
        context_bonus = 1 if re.search(r"\b(?:DATE|TARIKH|DTE)\b", line, flags=re.IGNORECASE) else 0
        for pattern in patterns:
            for match in re.finditer(pattern, line):
                token = match.group(0)
                window_start = max(0, match.start() - 28)
                window_end = min(len(line), match.end() + 18)
                window = line[window_start:window_end]
                has_date_context = bool(re.search(r"\b(?:DATE|TARIKH|DTE|DUE DATE)\b", window, flags=re.IGNORECASE))
                has_explicit_year = bool(re.search(r"20[0-9OoQDISsl|B]{2}", token))
                if not has_date_context and not has_explicit_year:
                    continue
                normalized = normalize_receipt_date(token)
                if normalized:
                    yield normalized, max(0, idx - context_bonus)


def _total_candidates_from_text(text: str) -> Iterable[tuple[str, int, float]]:
    lines = [_normalize_line(line) for line in str(text or "").splitlines()]
    total_context = re.compile(
        r"\b(?:GRAND\s+TOTAL|TOTAL\s+AMOUNT|AMOUNT\s+PAYABLE|NET\s+TOTAL|BALANCE\s+DUE|TOTAL)\b",
        flags=re.IGNORECASE,
    )
    reject_context = re.compile(
        r"\b(?:SUB\s*TOTAL|SUBTOTAL|TOTAL\s+ITEM|ITEM\s+DISCOUNT|QTY|QUANTITY|CHANGE|ROUND\s+AMT)\b",
        flags=re.IGNORECASE,
    )
    money_pattern = re.compile(r"(?:RM|MYR|\$)?\s*([0-9]{1,3}(?:[, ]?[0-9]{3})*|[0-9]+)[.,]([0-9]{2})\b", flags=re.IGNORECASE)
    for idx, line in enumerate(lines):
        if not total_context.search(line) or reject_context.search(line):
            continue
        values = []
        for match in money_pattern.finditer(line):
            whole = re.sub(r"[^0-9]", "", match.group(1))
            cents = match.group(2)
            if not whole:
                continue
            value = f"{int(whole)}.{cents}"
            values.append(value)
        if not values:
            continue
        value = values[-1]
        upper = line.upper()
        score = 12.0
        if "GRAND TOTAL" in upper or "TOTAL AMOUNT" in upper or re.search(r"\bTOTAL\s*[:=]", upper):
            score += 8.0
        if idx < 30:
            score += max(0.0, 5.0 - (idx * 0.08))
        yield value, idx, score


def _address_candidates_from_text(text: str) -> Iterable[tuple[str, int]]:
    lines = [_normalize_line(line) for line in str(text or "").splitlines()]
    lines = [line for line in lines if line]
    for idx, line in enumerate(lines[:18]):
        upper = line.upper()
        if not re.search(r"\b(?:NO\.?|LOT|JALAN|BANDAR|TAMAN|KG|KM)\b", upper):
            continue
        parts = [upper]
        for following in lines[idx + 1 : idx + 4]:
            follow_upper = following.upper()
            if re.search(r"\b(?:TEL|FAX|GST|TAX\s+INVOICE|INVOICE\s+NO|CASHIER|DATE|RECEIPT)\b", follow_upper):
                break
            if re.search(r"\b(?:JALAN|BANDAR|TAMAN|KG|KM|JOHOR|SELANGOR|KUALA|IPOH|BAHRU|KALI|MASAI|PERMAS|JAYA)\b|\b\d{5}\b", follow_upper):
                parts.append(follow_upper)
        candidate = " ".join(parts)
        key = normalize_receipt_key(candidate)
        if len(key.split()) >= 5 and re.search(r"\b(?:JALAN|BANDAR|JOHOR|SELANGOR|KUALA|IPOH|BAHRU|PERMAS|JAYA)\b", key):
            yield key, idx


def _address_score(value: str, *, line_index: int, source_score: float) -> float:
    key = normalize_receipt_key(value)
    tokens = key.split()
    if len(tokens) < 5:
        return 0.0
    geo_bonus = sum(2.0 for token in tokens if token in {"JALAN", "BANDAR", "JOHOR", "SELANGOR", "KUALA", "IPOH", "BAHRU", "PERMAS", "JAYA", "LOT", "KG", "KM"})
    postal_bonus = 5.0 if re.search(r"\b\d{5}\b", key) else 0.0
    top_bonus = max(0.0, 6.0 - (line_index * 0.6))
    return min(36.0, len(tokens) * 1.2 + geo_bonus + postal_bonus + top_bonus + min(8.0, source_score / 16.0))


def _bucket_score(items: list[Mapping[str, Any]], value_key: str) -> float:
    support = len(items)
    best = max(float(item.get("selection_score") or 0.0) for item in items)
    return support * 20.0 + min(15.0, best / 8.0)


def receipt_fields_from_candidates(candidates: Iterable[Mapping[str, Any]]) -> list[Dict[str, Any]]:
    buckets: dict[tuple[str, str], list[Dict[str, Any]]] = defaultdict(list)
    for row in candidates:
        if not row.get("ok"):
            continue
        text = str(row.get("markdown") or row.get("text") or "")
        source = {
            "engine": str(row.get("engine") or ""),
            "preprocess": str(row.get("preprocess") or ""),
            "selection_score": float(row.get("_selection_score") or row.get("selection_score") or 0.0),
        }
        for company, line_index in _company_candidates_from_text(text):
            key = normalize_receipt_key(company)
            if len(key.split()) < 2:
                continue
            score = _company_score(company, line_index=line_index, source_score=source["selection_score"])
            if score <= 6.0:
                continue
            buckets[("company", key)].append({**source, "value": company, "line_index": line_index, "candidate_score": score})
        for date, line_index in _date_candidates_from_text(text):
            buckets[("date", date)].append({**source, "value": date, "line_index": line_index, "candidate_score": 10.0})
        for total, line_index, candidate_score in _total_candidates_from_text(text):
            buckets[("total", total)].append({**source, "value": total, "line_index": line_index, "candidate_score": candidate_score})
        for address, line_index in _address_candidates_from_text(text):
            score = _address_score(address, line_index=line_index, source_score=source["selection_score"])
            if score <= 10.0:
                continue
            buckets[("address", normalize_receipt_key(address))].append({**source, "value": address, "line_index": line_index, "candidate_score": score})

    rows: list[Dict[str, Any]] = []
    for (field, value_key), items in buckets.items():
        if field == "company":
            score = _bucket_score(items, value_key) + max(float(item.get("candidate_score") or 0.0) for item in items)
            value = _best_company_display(value_key, [str(item.get("value") or "") for item in items])
        elif field == "address":
            score = _bucket_score(items, value_key) + max(float(item.get("candidate_score") or 0.0) for item in items)
            value = _best_address_display([str(item.get("value") or "") for item in items])
        elif field == "total":
            score = _bucket_score(items, value_key) + max(float(item.get("candidate_score") or 0.0) for item in items)
            value = value_key
        else:
            score = _bucket_score(items, value_key)
            value = value_key
        confidence = min(1.0, 0.35 + (len(items) * 0.12) + (score / 180.0))
        rows.append(
            {
                "field": field,
                "value": value,
                "normalized_value": value_key,
                "confidence": round(confidence, 3),
                "support": len(items),
                "sources": [{k: v for k, v in item.items() if k not in {"value", "candidate_score"}} for item in items[:8]],
                "score": round(score, 3),
                "repair_type": f"{field}_evidence_consensus" if len(items) > 1 else f"{field}_single_evidence_candidate",
                "unsupported_tokens": [],
            }
        )

    best_by_field: dict[str, Dict[str, Any]] = {}
    for row in sorted(rows, key=lambda item: (-float(item["score"]), -int(item["support"]), str(item["normalized_value"]))):
        best_by_field.setdefault(str(row["field"]), row)
    return sorted(best_by_field.values(), key=lambda item: str(item["field"]))


def _best_company_display(value_key: str, raw_values: Iterable[str]) -> str:
    normalized = normalize_receipt_key(value_key)
    variants = [normalize_receipt_key(value) for value in raw_values if normalize_receipt_key(value)]
    if not variants:
        return normalized
    return max(variants, key=lambda item: (SequenceMatcher(None, item, normalized).ratio(), len(item)))


def _best_address_display(raw_values: Iterable[str]) -> str:
    variants = [normalize_receipt_key(value) for value in raw_values if normalize_receipt_key(value)]
    if not variants:
        return ""
    return max(variants, key=lambda item: (len(set(item.split())), len(item)))


def append_receipt_fields(markdown: str, receipt_fields: Iterable[Mapping[str, Any]]) -> str:
    base = str(markdown or "").strip()
    rows = []
    for item in receipt_fields:
        confidence = float(item.get("confidence") or 0.0)
        support = int(item.get("support") or 0)
        if confidence < 0.5 and support < 2:
            continue
        field = str(item.get("field") or "")
        value = str(item.get("value") or "").strip()
        if field == "company" and value:
            rows.append(f"Company: {value}")
        elif field == "date" and value:
            rows.append(f"Date: {value}")
        elif field == "address" and value:
            rows.append(f"Address: {value}")
        elif field == "total" and value:
            rows.append(f"Total: {value}")
    if not rows:
        return base
    appendix = "Receipt fields:\n" + "\n".join(rows)
    if appendix in base:
        return base
    return (base + "\n\n" + appendix).strip() if base else appendix


def receipt_field_value(receipt_fields: Iterable[Mapping[str, Any]], field: str) -> str:
    for item in receipt_fields:
        if str(item.get("field") or "") == field:
            return str(item.get("value") or "")
    return ""

