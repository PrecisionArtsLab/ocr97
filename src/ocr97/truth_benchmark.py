from __future__ import annotations

import argparse
import json
import re
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Optional

from .field_evidence import classify_failure, field_candidates, normalize_for_type


_VALUE_SEPARATORS = r"[:=\|\t]"


def _normalize_space(value: Any) -> str:
    return " ".join(str(value or "").strip().split())


def _normalize_key(value: Any) -> str:
    raw = _normalize_space(value).lower()
    return re.sub(r"[^a-z0-9]+", " ", raw).strip()


def _ocr_text_key(value: Any) -> str:
    raw = _normalize_space(value).lower()
    raw = raw.translate(str.maketrans({"0": "o", "q": "o", "1": "l", "|": "l"}))
    return re.sub(r"[^a-z0-9]+", "", raw)


def _identifier_digits(value: Any) -> str:
    return "".join(re.findall(r"\d+", str(value or "")))


def _identifier_alpha(value: Any) -> str:
    raw = str(value or "").lower().translate(str.maketrans({"0": "o", "q": "o", "1": "i", "|": "i", "l": "i"}))
    return re.sub(r"[^a-z]+", "", raw)


def _number(value: Any) -> Optional[float]:
    raw = str(value or "").strip()
    if not raw:
        return None
    raw = raw.replace(";", ",")
    raw = re.sub(r"(?<=\d)<(?=\d{3}(?:\D|$))", ",", raw)
    raw = re.sub(r"(?<=\d)-(?=\d{2}(?:\D|$))", ".", raw)
    match = re.search(r"\(?-?\$?\d(?:[\d,\s<]*\d)?(?:[.\-]\d+)?%?\)?", raw)
    if not match:
        return None
    token = match.group(0)
    negative = token.startswith("(") and token.endswith(")")
    token = re.sub(r"\s+[1ilI|](?:\s+[1ilI|])*$", "", token)
    token = re.sub(r"(?<=\d)<(?=\d{3}(?:\D|$))", ",", token)
    token = re.sub(r"(?<=\d)-(?=\d{2}(?:\D|$))", ".", token)
    token = token.replace(";", ",")
    token = re.sub(r"(?<=\d)[,\s]+(?=\d{3}(?:\D|$))", "", token)
    raw = token.replace("$", "").replace(",", "").replace("%", "").replace("(", "").replace(")", "").replace(" ", "")
    if raw in {"", "-", ".", "-."}:
        return None
    try:
        num = float(raw)
    except Exception:
        return None
    return -num if negative else num


def _normalize_date(value: Any) -> str:
    normalized = normalize_for_type(value, "date")
    if normalized:
        return normalized
    raw = _normalize_space(value).lower()
    return re.sub(r"[^0-9a-z-]+", "", raw.replace("/", "-").replace(".", "-"))


def _value_matches(actual: Any, expected: Any, field_type: str, tolerance: float = 0.0) -> bool:
    kind = str(field_type or "text").strip().lower()
    if kind in {"money", "number", "percent"}:
        got = _number(actual)
        want = _number(expected)
        if got is None or want is None:
            return False
        return abs(got - want) <= float(tolerance or 0.0)
    if kind == "date":
        return _normalize_date(actual) == _normalize_date(expected)
    actual_key = _normalize_key(actual)
    expected_key = _normalize_key(expected)
    if actual_key == expected_key or actual_key.startswith(expected_key + " "):
        return True
    actual_ocr_key = _ocr_text_key(actual)
    expected_ocr_key = _ocr_text_key(expected)
    if actual_ocr_key == expected_ocr_key:
        return True
    if len(expected_ocr_key) >= 4 and actual_ocr_key.startswith(expected_ocr_key):
        return True
    expected_digits = _identifier_digits(expected)
    actual_digits = _identifier_digits(actual)
    if len(expected_digits) >= 4 and actual_digits.startswith(expected_digits):
        expected_alpha = _identifier_alpha(expected)
        actual_alpha = _identifier_alpha(actual)
        if not expected_alpha or any(ch in actual_alpha for ch in expected_alpha[:2]):
            return True
    return len(expected_key) >= 3 and re.search(rf"(?<![a-z0-9]){re.escape(expected_key)}(?![a-z0-9])", actual_key) is not None


def _number_partial_score(actual: Any, expected: Any, tolerance: float = 0.0) -> float:
    got = _number(actual)
    want = _number(expected)
    if got is None or want is None:
        return 0.0
    if abs(got - want) <= float(tolerance or 0.0):
        return 1.0
    if want == 0.0:
        return 1.0 if got == 0.0 else 0.0
    rel = abs(got - want) / abs(want)
    if rel <= 0.001:
        return 1.0
    if rel <= 0.01:
        return 0.75
    if rel <= 0.05:
        return 0.40
    return 0.0


def _alias_pattern(alias_text: str) -> str:
    parts: List[str] = []
    for ch in str(alias_text or ""):
        lower = ch.lower()
        if ch.isspace():
            parts.append(r"\s*")
        elif lower in {"l", "i"}:
            parts.append(r"[li1|]")
        elif lower == "o":
            parts.append(r"[o0qQ]")
        elif lower == "s":
            parts.append(r"[s5]")
        else:
            parts.append(re.escape(ch))
    return "".join(parts)


def _candidate_values(text: str, aliases: Iterable[Any]) -> List[str]:
    rows = [line.strip() for line in str(text or "").splitlines() if line.strip()]
    candidates: List[str] = []
    for alias in aliases:
        alias_text = _normalize_space(alias)
        if not alias_text:
            continue
        alias_pattern = _alias_pattern(alias_text)
        for row in rows:
            match = re.search(rf"\b{alias_pattern}\b\s*{_VALUE_SEPARATORS}\s*([^|\t]+)", row, flags=re.IGNORECASE)
            if match:
                candidates.append(match.group(1).strip())
                continue
            table_match = re.search(rf"\|\s*{alias_pattern}\s*\|\s*([^|]+)\|?", row, flags=re.IGNORECASE)
            if table_match:
                candidates.append(table_match.group(1).strip())
                continue
            pseudo_table_match = re.search(
                rf"\b{alias_pattern}\b\s+(?:[1ilI|]\s+)+([$]?\(?-?[0-9][0-9,\s<]*(?:[.\-][0-9]+)?%?\)?)",
                row,
                flags=re.IGNORECASE,
            )
            if pseudo_table_match:
                candidates.append(pseudo_table_match.group(1).strip())
                continue
            loose = re.search(rf"\b{alias_pattern}\b\s*[:=\-]?\s*([$]?\(?-?[0-9][0-9,\s<]*(?:[.\-][0-9]+)?%?\)?)", row, flags=re.IGNORECASE)
            if loose:
                candidates.append(loose.group(1).strip())
    return candidates


def _score_field(text: str, field: Mapping[str, Any]) -> Dict[str, Any]:
    aliases = list(field.get("aliases") or [field.get("name")])
    field_name_key = _normalize_key(field.get("name") or "")
    if field_name_key in {"amount due", "balance due"}:
        aliases.extend(["Total", "Grand Total", "Balance Due", "Payment Due"])
    elif field_name_key == "net amount":
        aliases.extend(["NetAmount"])
    expected = field.get("expected")
    field_type = str(field.get("type") or "text")
    tolerance = float(field.get("tolerance") or 0.0)
    candidates = _candidate_values(text, aliases)
    evidence_candidates = field_candidates(text, field)
    matched_value = ""
    partial_score = 0.0
    is_numeric = str(field_type or "text").strip().lower() in {"money", "number", "percent"}
    selected_candidate = evidence_candidates[0] if evidence_candidates else None
    for row in evidence_candidates:
        selected_value = str(row.get("value") or row.get("normalized_value") or "")
        if _value_matches(selected_value, expected, field_type, tolerance=tolerance):
            matched_value = selected_value
            partial_score = 1.0
            break
        if is_numeric:
            ps = _number_partial_score(selected_value, expected, tolerance=tolerance)
            if ps > partial_score:
                partial_score = ps
                matched_value = selected_value if ps > 0 else matched_value
    if partial_score < 1.0:
        for candidate in candidates[:8]:
            if _value_matches(candidate, expected, field_type, tolerance=tolerance):
                matched_value = candidate
                partial_score = 1.0
                break
            if is_numeric:
                ps = _number_partial_score(candidate, expected, tolerance=tolerance)
                if ps > partial_score:
                    partial_score = ps
                    matched_value = candidate if ps > 0 else matched_value
    exact_match = partial_score >= 1.0
    failure_bucket = classify_failure(
        expected=expected,
        candidates=evidence_candidates,
        matched=exact_match,
        field_type=field_type,
    )
    return {
        "name": str(field.get("name") or aliases[0] if aliases else ""),
        "expected": expected,
        "type": field_type,
        "matched": exact_match,
        "matched_value": matched_value if exact_match else "",
        "partial_score": round(partial_score, 3),
        "candidates": candidates[:8],
        "ranked_candidates": evidence_candidates,
        "failure_bucket": failure_bucket,
        "source_evidence": evidence_candidates[0] if evidence_candidates else None,
    }


def _required_token_score(text: str, tokens: Iterable[Any]) -> Dict[str, Any]:
    raw = str(text or "").lower()
    raw_compact = _normalize_space(raw)
    raw_squashed = re.sub(r"[^a-z0-9]+", "", raw)
    raw_ocr_key = _ocr_text_key(raw)
    required = [_normalize_space(token) for token in tokens if _normalize_space(token)]
    missing: List[str] = []
    for token in required:
        token_l = token.lower()
        token_squashed = re.sub(r"[^a-z0-9]+", "", token_l)
        token_ocr_key = _ocr_text_key(token_l)
        if token_l in raw or token_l in raw_compact:
            continue
        if token_squashed and token_squashed in raw_squashed:
            continue
        if token_ocr_key and token_ocr_key in raw_ocr_key:
            continue
        missing.append(token)
    total = len(required)
    score = 100 if total == 0 else int(round(((total - len(missing)) / float(total)) * 100.0))
    return {"score": score, "required": required, "missing": missing}


_FINANCIAL_LABEL_ROW = re.compile(
    r"^[A-Z][A-Za-z &/()]{2,35}(?:\s*[:]\s*|\s{2,})\$?-?[\d,]+(?:\.\d+)?(?:\s|$)",
)

def _table_row_count(text: str) -> int:
    count = 0
    for row in str(text or "").splitlines():
        clean = row.strip()
        if not clean:
            continue
        pipeish = re.sub(r"(?<=\s)[1ilI|](?=\s)", "|", clean)
        if "|" in clean and clean.count("|") >= 2:
            if re.search(r"\|\s*-+\s*\|", clean):
                continue
            count += max(1, clean.count("|") // 3)
        elif "|" in pipeish and pipeish.count("|") >= 3:
            count += max(1, pipeish.count("|") // 3)
        elif "\t" in clean and len(clean.split("\t")) >= 2:
            count += 1
        elif _FINANCIAL_LABEL_ROW.match(clean):
            count += 1
    return count


def score_case(case: Mapping[str, Any], extracted_text: Optional[str] = None) -> Dict[str, Any]:
    text = str(case.get("sample_text") if extracted_text is None else extracted_text or "")
    fields = [dict(item) for item in list(case.get("expected_fields") or []) if isinstance(item, dict)]
    field_results = [_score_field(text, field) for field in fields]
    field_total = len(field_results)
    field_hits = sum(1 for row in field_results if row["matched"])
    failure_buckets: Dict[str, int] = {}
    for row in field_results:
        bucket = str(row.get("failure_bucket") or "unknown")
        if bucket != "matched":
            failure_buckets[bucket] = failure_buckets.get(bucket, 0) + 1
    field_score = 100 if field_total == 0 else int(round(
        (sum(float(row.get("partial_score") or (1.0 if row["matched"] else 0.0)) for row in field_results) / float(field_total)) * 100.0
    ))

    token_result = _required_token_score(text, case.get("required_tokens") or [])
    expected_rows = case.get("expected_table_rows")
    actual_rows = _table_row_count(text)
    if expected_rows is None:
        table_score = 100
    else:
        expected = max(0, int(expected_rows))
        table_score = 100 if actual_rows >= expected else int(round((actual_rows / float(max(1, expected))) * 100.0))

    overall = int(round((field_score * 0.70) + (int(token_result["score"]) * 0.15) + (table_score * 0.15)))
    return {
        "id": str(case.get("id") or ""),
        "label": str(case.get("label") or ""),
        "score": overall,
        "field_score": field_score,
        "field_hits": field_hits,
        "field_total": field_total,
        "fields": field_results,
        "failure_buckets": failure_buckets,
        "required_token_score": int(token_result["score"]),
        "missing_tokens": list(token_result["missing"]),
        "table_row_score": table_score,
        "expected_table_rows": expected_rows,
        "actual_table_rows": actual_rows,
    }


def score_manifest(manifest: Mapping[str, Any], extracted_text_by_id: Optional[Mapping[str, str]] = None) -> Dict[str, Any]:
    text_map = dict(extracted_text_by_id or {})
    cases = [dict(item) for item in list(manifest.get("cases") or []) if isinstance(item, dict)]
    results = [
        score_case(
            case,
            extracted_text=text_map[str(case.get("id") or "")] if str(case.get("id") or "") in text_map else None,
        )
        for case in cases
    ]
    avg = 0 if not results else int(round(sum(int(row["score"]) for row in results) / float(len(results))))
    failure_buckets: Dict[str, int] = {}
    for row in results:
        for bucket, count in dict(row.get("failure_buckets") or {}).items():
            failure_buckets[str(bucket)] = failure_buckets.get(str(bucket), 0) + int(count)
    field_totals: Dict[str, Dict[str, int]] = {}
    for row in results:
        for field in list(row.get("fields") or []):
            name = str(field.get("name") or "unknown")
            field_totals.setdefault(name, {"hits": 0, "total": 0})
            field_totals[name]["total"] += 1
            if field.get("matched"):
                field_totals[name]["hits"] += 1
    field_accuracy = {
        name: {
            "hits": totals["hits"],
            "total": totals["total"],
            "accuracy": 100 if totals["total"] == 0 else round((totals["hits"] / float(totals["total"])) * 100.0, 2),
        }
        for name, totals in sorted(field_totals.items())
    }
    return {
        "name": str(manifest.get("name") or "ocr97_truth_benchmark"),
        "case_count": len(results),
        "score_avg": avg,
        "field_accuracy": field_accuracy,
        "failure_buckets": failure_buckets,
        "results": results,
    }


def benchmark_claim_readiness(result: Mapping[str, Any]) -> Dict[str, Any]:
    score = int(result.get("score_avg") or 0)
    failures = dict(result.get("failure_buckets") or {})
    case_count = int(result.get("case_count") or 0)
    has_failures = sum(int(v) for v in failures.values()) > 0
    if score >= 97 and not has_failures and case_count >= 100:
        level = "paper_strong"
        reason = "Score is at or above 97 on a large enough manifest with no field failures."
    elif score >= 90 and case_count >= 25:
        level = "technical_report_ready"
        reason = "Score is strong enough for a narrow technical report, but needs larger/repeated proof for a 97 claim."
    elif score >= 80:
        level = "engineering_progress"
        reason = "Score is useful, but remaining failures should drive the next improvement pass."
    else:
        level = "not_ready"
        reason = "Score is below the useful benchmark threshold or lacks enough terminal evidence."
    return {
        "level": level,
        "reason": reason,
        "score_avg": score,
        "case_count": case_count,
        "failure_buckets": failures,
    }


def load_manifest(path: Path) -> Dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError("truth_manifest_root_must_be_object")
    payload.setdefault("cases", [])
    return payload


def main(argv: Optional[List[str]] = None) -> int:
    parser = argparse.ArgumentParser(description="Score OCR97 extracted text against a field-level truth manifest.")
    parser.add_argument("--manifest", required=True, help="Path to a truth benchmark manifest.")
    parser.add_argument("--output", default="", help="Optional output JSON path.")
    args = parser.parse_args(argv)

    result = score_manifest(load_manifest(Path(args.manifest).expanduser()))
    readiness = benchmark_claim_readiness(result)
    if args.output:
        output = Path(args.output).expanduser()
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"name": result["name"], "case_count": result["case_count"], "score_avg": result["score_avg"], "readiness": readiness}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
