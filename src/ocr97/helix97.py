from __future__ import annotations

import argparse
import json
import os
import re
import time
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Optional

import requests

from .field_evidence import normalize_for_type, normalize_key
from .field_ranker import DEFAULT_WEIGHTS, candidate_features as ranker_candidate_features, dot as ranker_dot, rerank_candidates


PIPELINE_NAME = "Helix97"
SCHEMA_VERSION = "helix97.v1"
DEFAULT_GB10_URL = "http://host.docker.internal:11434"
DEFAULT_GB10_MODEL = "qwen2.5:72b"
DEFAULT_FIELD_RANKER_MIN_ACCURACY = 0.90
DEFAULT_CLEAN_MANIFEST_MIN_SCORE = 99
DEFAULT_CLEAN_MANIFEST_MAX_FAILURES = 0
DEFAULT_STRICT_MATRIX_MIN_DELTA = 0

_FIELD_ALIASES: Dict[str, List[str]] = {
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
    "revenue": ["revenue"],
    "cost": ["cost"],
    "margin": ["margin"],
    "principal": ["principal"],
    "interest_rate": ["interest rate"],
}
_LABEL_BETWEEN_RE = re.compile(r"\b[A-Za-z][A-Za-z0-9 ]{2,35}\s*:")


def _read_json(path: Path) -> Dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"json_root_must_be_object:{path}")
    return payload


def _safe_int(value: Any, *, default: int = 0) -> int:
    try:
        return int(value)
    except Exception:
        return default


def _safe_float(value: Any, *, default: float = 0.0) -> float:
    try:
        return float(value)
    except Exception:
        return default


def _coerce_evidence_path(path: Optional[Path]) -> Optional[Path]:
    if not path:
        return None
    candidate = path.expanduser()
    return candidate if candidate.exists() else None


def _write_jsonl(path: Path, rows: Iterable[Mapping[str, Any]]) -> int:
    path.parent.mkdir(parents=True, exist_ok=True)
    count = 0
    with path.open("w", encoding="utf-8", newline="\n") as handle:
        for row in rows:
            handle.write(json.dumps(dict(row), ensure_ascii=False, sort_keys=True) + "\n")
            count += 1
    return count


def _read_jsonl(path: Path) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    if not path.exists():
        return rows
    for line in path.read_text(encoding="utf-8").splitlines():
        clean = line.strip()
        if clean:
            row = json.loads(clean)
            if isinstance(row, dict):
                rows.append(row)
    return rows


def _artifact_text(row: Mapping[str, Any]) -> str:
    artifact_path = str(row.get("artifact_path") or "").strip()
    if artifact_path:
        path = Path(artifact_path)
        if path.exists():
            try:
                payload = _read_json(path)
                return str(payload.get("extracted_text") or payload.get("markdown") or payload.get("text") or "")
            except Exception:
                return ""
    return str(row.get("extracted_text") or row.get("markdown") or row.get("text") or "")


def _positive_candidate_index(field: Mapping[str, Any], expected_normalized: str) -> int:
    for index, candidate in enumerate(list(field.get("ranked_candidates") or [])):
        if str(candidate.get("normalized_value") or "") == expected_normalized:
            return index
    return -1


def _field_record(
    *,
    comparison_path: Path,
    engine: Mapping[str, Any],
    case_row: Mapping[str, Any],
    field: Mapping[str, Any],
    gb10_url: str,
    gb10_model: str,
) -> Dict[str, Any]:
    score = dict(case_row.get("score") or {})
    field_type = str(field.get("type") or "text")
    expected = field.get("expected")
    expected_normalized = normalize_for_type(expected, field_type)
    ranked = list(field.get("ranked_candidates") or [])
    selected = dict(ranked[0]) if ranked else None
    source_evidence = dict(field.get("source_evidence") or selected or {})
    return {
        "schema_version": SCHEMA_VERSION,
        "pipeline": PIPELINE_NAME,
        "stage": "field_ranker_correction",
        "created_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "source": {
            "comparison_path": str(comparison_path),
            "artifact_path": str(case_row.get("artifact_path") or ""),
            "input_path": str(case_row.get("input_path") or ""),
            "engine": str(engine.get("engine") or ""),
            "variant": "",
        },
        "case": {
            "id": str(score.get("id") or case_row.get("id") or ""),
            "label": str(score.get("label") or ""),
            "overall_score": int(score.get("score") or 0),
            "field_score": int(score.get("field_score") or 0),
        },
        "field": {
            "name": str(field.get("name") or ""),
            "type": field_type,
            "expected": expected,
            "expected_normalized": expected_normalized,
            "matched": bool(field.get("matched")),
            "partial_score": float(field.get("partial_score") or 0.0),
            "failure_bucket": str(field.get("failure_bucket") or ""),
        },
        "ocr": {
            "text": _artifact_text(case_row),
            "ranked_candidates": ranked,
            "selected_candidate": selected,
            "source_evidence": source_evidence,
        },
        "training_target": {
            "task": "select_correct_field_candidate",
            "correct_value": expected,
            "correct_normalized_value": expected_normalized,
            "positive_candidate_index": _positive_candidate_index(field, expected_normalized),
        },
        "gb10": {
            "intended_url": gb10_url,
            "intended_model": gb10_model,
            "role": "teacher_rationale_and_synthetic_variants",
            "status": "not_called",
        },
        "raw_ocr_training_gate": {
            "decision": "defer",
            "reason": "Extraction, ranking, correction, and layout-region layers must mature before raw OCR recognizer training.",
        },
    }


def collect_failure_records(
    comparison_path: Path,
    *,
    output_dir: Path,
    engine_name: str = "ocr97",
    gb10_url: str = DEFAULT_GB10_URL,
    gb10_model: str = DEFAULT_GB10_MODEL,
    include_partial: bool = True,
) -> Dict[str, Any]:
    payload = _read_json(comparison_path)
    rows: List[Dict[str, Any]] = []
    for engine in list(payload.get("engines") or []):
        if str(engine.get("engine") or "").lower() != engine_name.lower():
            continue
        for case_row in list(engine.get("results") or []):
            score = dict(case_row.get("score") or {})
            for field in list(score.get("fields") or []):
                matched = bool(field.get("matched"))
                partial = float(field.get("partial_score") or 0.0)
                if matched and not (include_partial and 0.0 < partial < 1.0):
                    continue
                rows.append(
                    _field_record(
                        comparison_path=comparison_path,
                        engine=engine,
                        case_row=case_row,
                        field=field,
                        gb10_url=gb10_url,
                        gb10_model=gb10_model,
                    )
                )
    output_dir.mkdir(parents=True, exist_ok=True)
    dataset_path = output_dir / "helix97_field_failures.jsonl"
    count = _write_jsonl(dataset_path, rows)
    manifest = {
        "pipeline": PIPELINE_NAME,
        "schema_version": SCHEMA_VERSION,
        "stage": "failure_capture",
        "source_comparison": str(comparison_path),
        "engine": engine_name,
        "record_count": count,
        "dataset_path": str(dataset_path),
        "gb10": {
            "intended_url": gb10_url,
            "intended_model": gb10_model,
            "role": "teacher_rationale_and_synthetic_variants",
        },
    }
    (output_dir / "helix97_manifest.json").write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    return manifest


def collect_failure_records_many(
    comparison_paths: Iterable[Path],
    *,
    output_dir: Path,
    engine_name: str = "ocr97",
    gb10_url: str = DEFAULT_GB10_URL,
    gb10_model: str = DEFAULT_GB10_MODEL,
    include_partial: bool = True,
) -> Dict[str, Any]:
    rows: List[Dict[str, Any]] = []
    sources: List[str] = []
    for comparison_path in comparison_paths:
        path = comparison_path.expanduser()
        if not path.exists():
            continue
        sources.append(str(path))
        payload = _read_json(path)
        for engine in list(payload.get("engines") or []):
            if str(engine.get("engine") or "").lower() != engine_name.lower():
                continue
            for case_row in list(engine.get("results") or []):
                score = dict(case_row.get("score") or {})
                for field in list(score.get("fields") or []):
                    matched = bool(field.get("matched"))
                    partial = float(field.get("partial_score") or 0.0)
                    if matched and not (include_partial and 0.0 < partial < 1.0):
                        continue
                    rows.append(
                        _field_record(
                            comparison_path=path,
                            engine=engine,
                            case_row=case_row,
                            field=field,
                            gb10_url=gb10_url,
                            gb10_model=gb10_model,
                        )
                    )
    output_dir.mkdir(parents=True, exist_ok=True)
    dataset_path = output_dir / "helix97_field_failures.jsonl"
    count = _write_jsonl(dataset_path, rows)
    manifest = {
        "pipeline": PIPELINE_NAME,
        "schema_version": SCHEMA_VERSION,
        "stage": "multi_failure_capture",
        "source_comparisons": sources,
        "engine": engine_name,
        "record_count": count,
        "dataset_path": str(dataset_path),
        "gb10": {
            "intended_url": gb10_url,
            "intended_model": gb10_model,
            "role": "teacher_rationale_and_synthetic_variants",
        },
    }
    (output_dir / "helix97_manifest.json").write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    return manifest


def _candidate_features(candidate: Mapping[str, Any], record: Mapping[str, Any]) -> Dict[str, float]:
    source_line = str(candidate.get("source_line") or "")
    reason = str(candidate.get("reason") or "")
    field = dict(record.get("field") or {})
    field_name = str(field.get("name") or "")
    field_type = str(field.get("type") or "text")
    line_key = normalize_key(source_line)
    value = str(candidate.get("value") or "")
    value_start = source_line.find(value)
    aliases = _FIELD_ALIASES.get(field_name, [field_name.replace("_", " ")])
    best_distance_score = 0.0
    immediate_after_label = 0.0
    labels_between = 0.0
    for alias in aliases:
        alias_key = str(alias or "")
        if not alias_key or value_start < 0:
            continue
        for match in re.finditer(re.escape(alias_key), source_line, flags=re.IGNORECASE):
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
        "has_percent": 1.0 if "%" in value else 0.0,
        "type_percent_match": 1.0 if field_type == "percent" and "%" in value else 0.0,
        "iso_date_value": 1.0 if re.fullmatch(r"\d{4}[-/]\d{1,2}[-/]\d{1,2}", value.strip()) else 0.0,
        "line_is_table_like": 1.0 if source_line.count("|") >= 2 else 0.0,
    }


def _dot(weights: Mapping[str, float], features: Mapping[str, float]) -> float:
    return sum(float(weights.get(name, 0.0)) * float(value) for name, value in features.items())


def _positive_index(record: Mapping[str, Any]) -> int:
    raw = (record.get("training_target") or {}).get("positive_candidate_index")
    try:
        return int(raw)
    except Exception:
        return -1


def train_field_ranker(
    dataset_path: Path,
    *,
    output_dir: Path,
    epochs: int = 8,
    learning_rate: float = 0.25,
    gb10_url: str = DEFAULT_GB10_URL,
    gb10_model: str = DEFAULT_GB10_MODEL,
) -> Dict[str, Any]:
    records = _read_jsonl(dataset_path)
    weights: Dict[str, float] = dict(DEFAULT_WEIGHTS)
    trainable = [
        row
        for row in records
        if _positive_index(row) >= 0
        and len(((row.get("ocr") or {}).get("ranked_candidates") or [])) >= 2
    ]
    for _epoch in range(max(1, int(epochs))):
        for record in trainable:
            candidates = list((record.get("ocr") or {}).get("ranked_candidates") or [])
            positive = _positive_index(record)
            field = dict(record.get("field") or {})
            scored = [(ranker_dot(weights, ranker_candidate_features(candidate, field)), index, candidate) for index, candidate in enumerate(candidates)]
            predicted = max(scored, key=lambda item: (item[0], -item[1]))[1]
            if predicted == positive:
                continue
            pos_features = ranker_candidate_features(candidates[positive], field)
            pred_features = ranker_candidate_features(candidates[predicted], field)
            for name in set(pos_features) | set(pred_features):
                weights[name] = round(
                    float(weights.get(name, 0.0))
                    + (float(learning_rate) * (float(pos_features.get(name, 0.0)) - float(pred_features.get(name, 0.0)))),
                    6,
                )
    correct = 0
    evaluated = 0
    for record in trainable:
        candidates = list((record.get("ocr") or {}).get("ranked_candidates") or [])
        positive = _positive_index(record)
        field = dict(record.get("field") or {})
        scored = [(ranker_dot(weights, ranker_candidate_features(candidate, field)), index) for index, candidate in enumerate(candidates)]
        predicted = max(scored, key=lambda item: (item[0], -item[1]))[1]
        evaluated += 1
        if predicted == positive:
            correct += 1
    output_dir.mkdir(parents=True, exist_ok=True)
    model = {
        "pipeline": PIPELINE_NAME,
        "schema_version": SCHEMA_VERSION,
        "stage": "field_ranker_model",
        "model_type": "linear_candidate_ranker",
        "weights": weights,
        "training": {
            "dataset_path": str(dataset_path),
            "records_total": len(records),
            "records_trainable": len(trainable),
            "epochs": int(epochs),
            "learning_rate": float(learning_rate),
            "accuracy_on_trainable": 0.0 if evaluated == 0 else round(correct / float(evaluated), 4),
        },
        "gb10": {
            "intended_url": gb10_url,
            "intended_model": gb10_model,
            "role": "teacher_model_for_rationales_and_synthetic_expansion",
            "note": "This lightweight local ranker is the first correction layer; GB10 should expand and judge examples before larger fine-tunes.",
        },
    }
    model_path = output_dir / "helix97_field_ranker_model.json"
    model_path.write_text(json.dumps(model, indent=2) + "\n", encoding="utf-8")
    return model | {"model_path": str(model_path)}


def evaluate_field_ranker(dataset_path: Path, model_path: Path, *, output_dir: Path) -> Dict[str, Any]:
    records = _read_jsonl(dataset_path)
    model = _read_json(model_path)
    rows: List[Dict[str, Any]] = []
    correct = 0
    evaluated = 0
    for record in records:
        candidates = list((record.get("ocr") or {}).get("ranked_candidates") or [])
        positive = _positive_index(record)
        if positive < 0 or len(candidates) < 2:
            continue
        ranked = rerank_candidates(candidates, dict(record.get("field") or {}), model=model)
        predicted_norm = str((ranked[0] if ranked else {}).get("normalized_value") or "")
        expected_norm = str((record.get("training_target") or {}).get("correct_normalized_value") or "")
        passed = predicted_norm == expected_norm
        evaluated += 1
        correct += 1 if passed else 0
        rows.append(
            {
                "case_id": (record.get("case") or {}).get("id"),
                "field": (record.get("field") or {}).get("name"),
                "expected_normalized": expected_norm,
                "predicted_normalized": predicted_norm,
                "passed": passed,
                "top_value": (ranked[0] if ranked else {}).get("value"),
                "top_score": (ranked[0] if ranked else {}).get("learned_rank_score"),
            }
        )
    output_dir.mkdir(parents=True, exist_ok=True)
    report = {
        "pipeline": PIPELINE_NAME,
        "schema_version": SCHEMA_VERSION,
        "stage": "field_ranker_evaluation",
        "dataset_path": str(dataset_path),
        "model_path": str(model_path),
        "evaluated": evaluated,
        "correct": correct,
        "accuracy": 0.0 if evaluated == 0 else round(correct / float(evaluated), 4),
        "rows": rows,
    }
    report_path = output_dir / "helix97_field_ranker_eval.json"
    report_path.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    lines = [
        "# Helix97 Field Ranker Evaluation",
        "",
        f"- dataset: `{dataset_path}`",
        f"- model: `{model_path}`",
        f"- accuracy: `{report['accuracy']}` ({correct}/{evaluated})",
        "",
        "| Case | Field | Expected | Predicted | Pass |",
        "|---|---|---|---|---:|",
    ]
    for row in rows:
        lines.append(
            f"| {row.get('case_id')} | {row.get('field')} | {row.get('expected_normalized')} | {row.get('predicted_normalized')} | {row.get('passed')} |"
        )
    (output_dir / "helix97_field_ranker_eval.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    return report | {"report_path": str(report_path), "markdown_path": str(output_dir / "helix97_field_ranker_eval.md")}


def evaluate_promotion_gates(
    *,
    comparison_path: Path,
    output_dir: Path,
    raw_ocr_gate: Optional[Dict[str, Any]] = None,
    field_ranker_eval_path: Optional[Path] = None,
    clean_manifest_path: Optional[Path] = None,
    strict_matrix_summary_path: Optional[Path] = None,
    field_ranker_min_accuracy: float = DEFAULT_FIELD_RANKER_MIN_ACCURACY,
    clean_manifest_min_score: int = DEFAULT_CLEAN_MANIFEST_MIN_SCORE,
    clean_manifest_max_failures: int = DEFAULT_CLEAN_MANIFEST_MAX_FAILURES,
    strict_matrix_min_delta: int = DEFAULT_STRICT_MATRIX_MIN_DELTA,
) -> Dict[str, Any]:
    checks: List[Dict[str, Any]] = []
    raw_gate = raw_ocr_gate or raw_ocr_training_gate(
        comparison_path,
        output_path=output_dir / "helix97_raw_ocr_gate_for_promotion.json",
    )

    captured_ok = False
    if field_ranker_eval_path and field_ranker_eval_path.exists():
        eval_payload = _read_json(field_ranker_eval_path)
        evaluated = _safe_int(eval_payload.get("evaluated"))
        accuracy = _safe_float(eval_payload.get("accuracy"))
        passed = evaluated > 0 and accuracy >= field_ranker_min_accuracy
        checks.append(
            {
                "gate": "captured_failure_accuracy",
                "status": "pass" if passed else "fail",
                "required": {"min_accuracy": field_ranker_min_accuracy, "min_evaluated": 1},
                "actual": {"evaluated": evaluated, "accuracy": accuracy},
                "path": str(field_ranker_eval_path),
                "reason": (
                    "Captured failure evidence is strong enough for promotion."
                    if passed
                    else "Captured-failure evaluation is weak or unavailable for promotion."
                ),
            }
        )
    else:
        checks.append(
            {
                "gate": "captured_failure_accuracy",
                "status": "missing",
                "required": {"min_accuracy": field_ranker_min_accuracy, "min_evaluated": 1},
                "actual": {"evaluated": 0, "accuracy": 0.0},
                "path": None,
                "reason": "Captured failure review evidence is missing.",
            }
        )
    captured_ok = checks[-1]["status"] == "pass"

    clean_manifest_ok = False
    if clean_manifest_path and clean_manifest_path.exists():
        manifest = _read_json(clean_manifest_path)
        score_avg = _safe_int(manifest.get("score_avg"))
        failure_count = _safe_int(manifest.get("failure_count"), default=-1)
        passed = score_avg >= clean_manifest_min_score and failure_count <= clean_manifest_max_failures
        checks.append(
            {
                "gate": "clean_manifest_gate",
                "status": "pass" if passed else "fail",
                "required": {
                    "min_score": clean_manifest_min_score,
                    "max_failures": clean_manifest_max_failures,
                },
                "actual": {"score_avg": score_avg, "failure_count": failure_count},
                "path": str(clean_manifest_path),
                "reason": (
                    "Clean real-document evidence satisfies promotion floor."
                    if passed
                    else "Clean manifest is below promotion threshold."
                ),
            }
        )
        clean_manifest_ok = passed
    else:
        checks.append(
            {
                "gate": "clean_manifest_gate",
                "status": "missing",
                "required": {
                    "min_score": clean_manifest_min_score,
                    "max_failures": clean_manifest_max_failures,
                },
                "actual": {"score_avg": None, "failure_count": None},
                "path": None,
                "reason": "Clean manifest evidence is required for promotion.",
            }
        )

    strict_ok = False
    if strict_matrix_summary_path and strict_matrix_summary_path.exists():
        strict_payload = _read_json(strict_matrix_summary_path)
        deltas: List[int] = []
        for variant in list(strict_payload.get("variants") or []):
            delta = _safe_int((variant.get("summary") or {}).get("ocr97_vs_best_baseline_delta"), default=-10**9)
            deltas.append(delta)
        if deltas:
            worst_delta = min(deltas)
            passed = worst_delta >= strict_matrix_min_delta
            checks.append(
                {
                    "gate": "strict_matrix_gate",
                    "status": "pass" if passed else "fail",
                    "required": {"min_delta": strict_matrix_min_delta},
                    "actual": {"worst_delta": worst_delta, "variant_count": len(deltas)},
                    "path": str(strict_matrix_summary_path),
                    "reason": (
                        "Strict matrix deltas are non-regressive."
                        if passed
                        else "Strict matrix shows one or more degrading variants."
                    ),
                }
            )
            strict_ok = passed
        else:
            checks.append(
                {
                    "gate": "strict_matrix_gate",
                    "status": "missing",
                    "required": {"min_delta": strict_matrix_min_delta},
                    "actual": {"worst_delta": None, "variant_count": 0},
                    "path": str(strict_matrix_summary_path),
                    "reason": "Strict matrix summary has no parsed deltas.",
                }
            )
    else:
        checks.append(
            {
                "gate": "strict_matrix_gate",
                "status": "missing",
                "required": {"min_delta": strict_matrix_min_delta},
                "actual": {"worst_delta": None, "variant_count": 0},
                "path": None,
                "reason": "Strict matrix evidence is required for promotion.",
            }
        )

    can_promote = raw_gate.get("decision") == "consider_after_layout_gate" and captured_ok and clean_manifest_ok and strict_ok
    if can_promote:
        decision = "allow_raw_ocr_promotion"
        status = "pass"
    elif raw_gate.get("decision") == "defer_raw_ocr_training":
        decision = "hold_raw_ocr_promotion"
        status = "hold"
    else:
        decision = "block_raw_ocr_promotion"
        status = "fail"

    result = {
        "pipeline": PIPELINE_NAME,
        "schema_version": SCHEMA_VERSION,
        "stage": "raw_ocr_promotion_gates",
        "comparison_path": str(comparison_path),
        "status": status,
        "decision": decision,
        "decision_reason": raw_gate.get("reason"),
        "raw_ocr_training_gate": raw_gate,
        "checks": checks,
        "requirements": {
            "field_ranker_eval": bool(field_ranker_eval_path and field_ranker_eval_path.exists()),
            "clean_manifest": bool(clean_manifest_path and clean_manifest_path.exists()),
            "strict_matrix_summary": bool(strict_matrix_summary_path and strict_matrix_summary_path.exists()),
            "thresholds": {
                "field_ranker_min_accuracy": field_ranker_min_accuracy,
                "clean_manifest_min_score": clean_manifest_min_score,
                "clean_manifest_max_failures": clean_manifest_max_failures,
                "strict_matrix_min_delta": strict_matrix_min_delta,
            },
        },
    }
    (output_dir / "helix97_promotion_gate.json").write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    return result


def export_layout_region_examples(dataset_path: Path, *, output_dir: Path) -> Dict[str, Any]:
    records = _read_jsonl(dataset_path)
    examples: List[Dict[str, Any]] = []
    for record in records:
        evidence = dict((record.get("ocr") or {}).get("source_evidence") or {})
        source_line = str(evidence.get("source_line") or "")
        if not source_line:
            continue
        examples.append(
            {
                "schema_version": SCHEMA_VERSION,
                "pipeline": PIPELINE_NAME,
                "stage": "layout_region_weak_label",
                "case_id": (record.get("case") or {}).get("id"),
                "field_name": (record.get("field") or {}).get("name"),
                "field_type": (record.get("field") or {}).get("type"),
                "input_path": (record.get("source") or {}).get("input_path"),
                "region_text": source_line,
                "line_index": int(evidence.get("line_index") or 0),
                "weak_label": {
                    "region_role": "field_value_line",
                    "expected_value": (record.get("training_target") or {}).get("correct_value"),
                    "expected_normalized_value": (record.get("training_target") or {}).get("correct_normalized_value"),
                },
                "next_model_family": "layout-region classifier before raw OCR recognizer training",
            }
        )
    output_dir.mkdir(parents=True, exist_ok=True)
    layout_path = output_dir / "helix97_layout_regions.jsonl"
    count = _write_jsonl(layout_path, examples)
    summary = {
        "pipeline": PIPELINE_NAME,
        "schema_version": SCHEMA_VERSION,
        "stage": "layout_region_export",
        "source_dataset": str(dataset_path),
        "example_count": count,
        "layout_dataset_path": str(layout_path),
    }
    (output_dir / "helix97_layout_manifest.json").write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    return summary


def gb10_status(gb10_url: str = DEFAULT_GB10_URL, *, timeout: float = 2.5) -> Dict[str, Any]:
    try:
        response = requests.get(gb10_url.rstrip("/") + "/api/tags", timeout=timeout)
        return {
            "url": gb10_url,
            "reachable": response.ok,
            "status_code": response.status_code,
            "models": [str(item.get("name") or "") for item in (response.json().get("models") or []) if isinstance(item, dict)] if response.ok else [],
        }
    except Exception as exc:
        return {"url": gb10_url, "reachable": False, "error": f"{type(exc).__name__}:{exc}"}


def _gb10_teacher_prompt(record: Mapping[str, Any]) -> str:
    field = dict(record.get("field") or {})
    target = dict(record.get("training_target") or {})
    ocr = dict(record.get("ocr") or {})
    candidates = list(ocr.get("ranked_candidates") or [])[:6]
    compact = {
        "case": record.get("case"),
        "field": field,
        "correct_normalized_value": target.get("correct_normalized_value"),
        "selected_candidate": ocr.get("selected_candidate"),
        "ranked_candidates": candidates,
    }
    return (
        "You are Helix97, a local OCR97 teacher model running on GB10. "
        "Given this OCR field-selection failure, return compact JSON with keys "
        "correction_reason_summary, hard_negative_patterns, synthetic_variant_prompts, and layout_region_hint. "
        "Do not include hidden reasoning. Use concrete evidence from the candidates.\n\n"
        + json.dumps(compact, ensure_ascii=False)
    )


def augment_with_gb10_teacher(
    dataset_path: Path,
    *,
    output_dir: Path,
    gb10_url: str = DEFAULT_GB10_URL,
    gb10_model: str = DEFAULT_GB10_MODEL,
    limit: int = 25,
    timeout: float = 120.0,
) -> Dict[str, Any]:
    records = _read_jsonl(dataset_path)
    selected = records[: max(0, int(limit))]
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / "helix97_gb10_teacher_augmented.jsonl"
    augmented: List[Dict[str, Any]] = []
    for record in selected:
        prompt = _gb10_teacher_prompt(record)
        teacher: Dict[str, Any]
        try:
            response = requests.post(
                gb10_url.rstrip("/") + "/api/generate",
                json={"model": gb10_model, "prompt": prompt, "stream": False, "format": "json"},
                timeout=timeout,
            )
            response.raise_for_status()
            text = str(response.json().get("response") or "")
            try:
                teacher = json.loads(text)
            except Exception:
                teacher = {"raw_response": text}
            status = "augmented"
            error = ""
        except Exception as exc:
            teacher = {}
            status = "gb10_unavailable"
            error = f"{type(exc).__name__}:{exc}"
        augmented.append(
            {
                **record,
                "gb10": {
                    **dict(record.get("gb10") or {}),
                    "intended_url": gb10_url,
                    "intended_model": gb10_model,
                    "status": status,
                    "error": error,
                },
                "teacher": teacher,
            }
        )
    count = _write_jsonl(output_path, augmented)
    summary = {
        "pipeline": PIPELINE_NAME,
        "schema_version": SCHEMA_VERSION,
        "stage": "gb10_teacher_augmentation",
        "source_dataset": str(dataset_path),
        "output_path": str(output_path),
        "requested": len(selected),
        "written": count,
        "augmented": sum(1 for row in augmented if (row.get("gb10") or {}).get("status") == "augmented"),
        "gb10_unavailable": sum(1 for row in augmented if (row.get("gb10") or {}).get("status") == "gb10_unavailable"),
        "gb10_url": gb10_url,
        "gb10_model": gb10_model,
    }
    (output_dir / "helix97_gb10_teacher_manifest.json").write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    return summary


def write_gb10_training_plan(
    *,
    output_dir: Path,
    field_dataset_path: Path,
    field_model_path: Path,
    layout_dataset_path: Path,
    gb10_url: str = DEFAULT_GB10_URL,
    gb10_model: str = DEFAULT_GB10_MODEL,
    raw_ocr_promotion_gate: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    raw_ocr_status = "deferred"
    if raw_ocr_promotion_gate:
        decision = str(raw_ocr_promotion_gate.get("decision") or "")
        if decision == "allow_raw_ocr_promotion":
            raw_ocr_status = "allowed"
        elif decision == "block_raw_ocr_promotion":
            raw_ocr_status = "blocked_by_gate"
        elif decision == "hold_raw_ocr_promotion":
            raw_ocr_status = "on_hold"
    status = gb10_status(gb10_url)
    plan = {
        "pipeline": PIPELINE_NAME,
        "schema_version": SCHEMA_VERSION,
        "stage": "gb10_training_plan",
        "gb10": {
            "url": gb10_url,
            "preferred_teacher_model": gb10_model,
            "status": status,
        },
        "phases": [
            {
                "id": "field_ranker",
                "status": "implemented",
                "dataset": str(field_dataset_path),
                "model": str(field_model_path),
                "purpose": "Learn to select the correct field candidate from OCR97 ranked candidates.",
            },
            {
                "id": "gb10_teacher_expansion",
                "status": "ready_for_operator",
                "dataset": str(field_dataset_path),
                "purpose": "Use GB10/Ollama as a local teacher to add correction rationales, hard negatives, and synthetic variants.",
            },
            {
                "id": "layout_region_training",
                "status": "scaffolded",
                "dataset": str(layout_dataset_path),
                "purpose": "Train a local model to identify field regions before selecting values.",
            },
            {
                "id": "raw_ocr_recognizer",
                "status": raw_ocr_status,
                "promotion_gate": raw_ocr_promotion_gate.get("decision") if raw_ocr_promotion_gate else "not_evaluated",
                "purpose": "Only start raw OCR recognition model training if extraction/ranking/layout layers still lose to baselines on real-document corpus gates.",
            },
        ],
    }
    output_dir.mkdir(parents=True, exist_ok=True)
    plan_path = output_dir / "helix97_gb10_training_plan.json"
    plan_path.write_text(json.dumps(plan, indent=2) + "\n", encoding="utf-8")
    md = [
        "# Helix97 GB10 Training Plan",
        "",
        "Helix97 is OCR97's correction-data and local-adaptation pipeline.",
        "",
        f"- GB10/Ollama URL: `{gb10_url}`",
        f"- Preferred teacher model: `{gb10_model}`",
        f"- Reachable now: `{status.get('reachable')}`",
        "",
        "## Phases",
        "",
    ]
    for phase in plan["phases"]:
        md.append(f"- `{phase['id']}`: {phase['status']} - {phase['purpose']}")
    md.extend(
        [
            "",
            "## Policy",
            "",
            "Raw OCR recognizer training stays deferred until field ranking, correction, and layout-region training have matured and OCR97 still loses on real-document gates.",
        ]
    )
    (output_dir / "helix97_gb10_training_plan.md").write_text("\n".join(md) + "\n", encoding="utf-8")
    return plan | {"plan_path": str(plan_path), "markdown_path": str(output_dir / "helix97_gb10_training_plan.md")}


def raw_ocr_training_gate(comparison_path: Path, *, output_path: Path, min_delta_to_defer: int = 0) -> Dict[str, Any]:
    payload = _read_json(comparison_path)
    summary = dict(payload.get("summary") or {})
    delta = summary.get("ocr97_vs_best_baseline_delta")
    if delta is None and isinstance(payload.get("rows"), list):
        row_deltas = [
            int(row.get("ocr97_vs_best_baseline_delta"))
            for row in payload.get("rows", [])
            if isinstance(row, dict) and row.get("ocr97_vs_best_baseline_delta") is not None
        ]
        if row_deltas:
            delta = min(row_deltas)
    try:
        delta_int = int(delta)
    except Exception:
        delta_int = -999
    if delta_int >= int(min_delta_to_defer):
        decision = "defer_raw_ocr_training"
        reason = "OCR97 is not losing this benchmark after extraction/ranking/layout layers; raw recognizer training is premature."
    else:
        decision = "consider_after_layout_gate"
        reason = "OCR97 is still behind the best baseline; finish layout-region training first, then reassess raw recognizer training."
    result = {
        "pipeline": PIPELINE_NAME,
        "schema_version": SCHEMA_VERSION,
        "stage": "raw_ocr_training_gate",
        "comparison_path": str(comparison_path),
        "delta_vs_best_baseline": delta,
        "decision": decision,
        "reason": reason,
    }
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    return result


def run_pipeline(
    comparison_path: Path,
    *,
    output_dir: Path,
    engine_name: str = "ocr97",
    gb10_url: str = DEFAULT_GB10_URL,
    gb10_model: str = DEFAULT_GB10_MODEL,
    clean_manifest_path: Optional[Path] = None,
    strict_matrix_summary_path: Optional[Path] = None,
    field_ranker_min_accuracy: float = DEFAULT_FIELD_RANKER_MIN_ACCURACY,
    clean_manifest_min_score: int = DEFAULT_CLEAN_MANIFEST_MIN_SCORE,
    clean_manifest_max_failures: int = DEFAULT_CLEAN_MANIFEST_MAX_FAILURES,
    strict_matrix_min_delta: int = DEFAULT_STRICT_MATRIX_MIN_DELTA,
) -> Dict[str, Any]:
    failures = collect_failure_records(
        comparison_path,
        output_dir=output_dir,
        engine_name=engine_name,
        gb10_url=gb10_url,
        gb10_model=gb10_model,
    )
    dataset_path = Path(failures["dataset_path"])
    model = train_field_ranker(dataset_path, output_dir=output_dir, gb10_url=gb10_url, gb10_model=gb10_model)
    evaluation = evaluate_field_ranker(dataset_path, Path(model["model_path"]), output_dir=output_dir)
    layout = export_layout_region_examples(dataset_path, output_dir=output_dir)
    gate = raw_ocr_training_gate(comparison_path, output_path=output_dir / "helix97_raw_ocr_gate.json")
    promotion_gate = evaluate_promotion_gates(
        comparison_path=comparison_path,
        output_dir=output_dir,
        raw_ocr_gate=gate,
        field_ranker_eval_path=Path(evaluation["report_path"]),
        clean_manifest_path=_coerce_evidence_path(clean_manifest_path),
        strict_matrix_summary_path=_coerce_evidence_path(strict_matrix_summary_path),
        field_ranker_min_accuracy=field_ranker_min_accuracy,
        clean_manifest_min_score=clean_manifest_min_score,
        clean_manifest_max_failures=clean_manifest_max_failures,
        strict_matrix_min_delta=strict_matrix_min_delta,
    )
    plan = write_gb10_training_plan(
        output_dir=output_dir,
        field_dataset_path=dataset_path,
        field_model_path=Path(model["model_path"]),
        layout_dataset_path=Path(layout["layout_dataset_path"]),
        gb10_url=gb10_url,
        gb10_model=gb10_model,
        raw_ocr_promotion_gate=promotion_gate,
    )
    summary = {
        "pipeline": PIPELINE_NAME,
        "schema_version": SCHEMA_VERSION,
        "output_dir": str(output_dir),
        "failure_capture": failures,
        "field_ranker": model,
        "field_ranker_eval": evaluation,
        "layout_regions": layout,
        "raw_ocr_gate": gate,
        "promotion_gate": promotion_gate,
        "gb10_training_plan": plan,
    }
    (output_dir / "helix97_run_summary.json").write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    return summary


def main(argv: Optional[List[str]] = None) -> int:
    parser = argparse.ArgumentParser(description="Helix97: OCR97 correction dataset, local ranker, layout scaffold, and GB10 training handoff.")
    sub = parser.add_subparsers(dest="command", required=True)

    run = sub.add_parser("run")
    run.add_argument("--comparison", required=True)
    run.add_argument("--output-dir", required=True)
    run.add_argument("--engine", default="ocr97")
    run.add_argument("--gb10-url", default=os.getenv("HELIX97_GB10_URL", DEFAULT_GB10_URL))
    run.add_argument("--gb10-model", default=os.getenv("HELIX97_GB10_MODEL", DEFAULT_GB10_MODEL))
    run.add_argument("--clean-manifest", default="")
    run.add_argument("--strict-matrix-summary", default="")
    run.add_argument("--field-ranker-min-accuracy", type=float, default=DEFAULT_FIELD_RANKER_MIN_ACCURACY)
    run.add_argument("--clean-manifest-min-score", type=int, default=DEFAULT_CLEAN_MANIFEST_MIN_SCORE)
    run.add_argument("--clean-manifest-max-failures", type=int, default=DEFAULT_CLEAN_MANIFEST_MAX_FAILURES)
    run.add_argument("--strict-matrix-min-delta", type=int, default=DEFAULT_STRICT_MATRIX_MIN_DELTA)

    collect = sub.add_parser("collect-failures")
    collect.add_argument("--comparison", required=True)
    collect.add_argument("--output-dir", required=True)
    collect.add_argument("--engine", default="ocr97")
    collect.add_argument("--gb10-url", default=os.getenv("HELIX97_GB10_URL", DEFAULT_GB10_URL))
    collect.add_argument("--gb10-model", default=os.getenv("HELIX97_GB10_MODEL", DEFAULT_GB10_MODEL))

    collect_many = sub.add_parser("collect-failures-many")
    collect_many.add_argument("--comparisons", nargs="+", required=True)
    collect_many.add_argument("--output-dir", required=True)
    collect_many.add_argument("--engine", default="ocr97")
    collect_many.add_argument("--gb10-url", default=os.getenv("HELIX97_GB10_URL", DEFAULT_GB10_URL))
    collect_many.add_argument("--gb10-model", default=os.getenv("HELIX97_GB10_MODEL", DEFAULT_GB10_MODEL))

    train = sub.add_parser("train-field-ranker")
    train.add_argument("--dataset", required=True)
    train.add_argument("--output-dir", required=True)
    train.add_argument("--epochs", type=int, default=8)
    train.add_argument("--learning-rate", type=float, default=0.25)

    evaluate = sub.add_parser("evaluate-field-ranker")
    evaluate.add_argument("--dataset", required=True)
    evaluate.add_argument("--model", required=True)
    evaluate.add_argument("--output-dir", required=True)

    layout = sub.add_parser("export-layout")
    layout.add_argument("--dataset", required=True)
    layout.add_argument("--output-dir", required=True)

    gate = sub.add_parser("raw-ocr-gate")
    gate.add_argument("--comparison", required=True)
    gate.add_argument("--output", required=True)

    augment = sub.add_parser("gb10-augment")
    augment.add_argument("--dataset", required=True)
    augment.add_argument("--output-dir", required=True)
    augment.add_argument("--gb10-url", default=os.getenv("HELIX97_GB10_URL", DEFAULT_GB10_URL))
    augment.add_argument("--gb10-model", default=os.getenv("HELIX97_GB10_MODEL", DEFAULT_GB10_MODEL))
    augment.add_argument("--limit", type=int, default=25)

    args = parser.parse_args(argv)
    if args.command == "run":
        result = run_pipeline(
            Path(args.comparison).expanduser(),
            output_dir=Path(args.output_dir).expanduser(),
            engine_name=str(args.engine),
            gb10_url=str(args.gb10_url),
            gb10_model=str(args.gb10_model),
            clean_manifest_path=Path(args.clean_manifest).expanduser() if str(getattr(args, "clean_manifest", "")) else None,
            strict_matrix_summary_path=Path(args.strict_matrix_summary).expanduser() if str(getattr(args, "strict_matrix_summary", "")) else None,
            field_ranker_min_accuracy=float(args.field_ranker_min_accuracy),
            clean_manifest_min_score=int(args.clean_manifest_min_score),
            clean_manifest_max_failures=int(args.clean_manifest_max_failures),
            strict_matrix_min_delta=int(args.strict_matrix_min_delta),
        )
    elif args.command == "collect-failures":
        result = collect_failure_records(
            Path(args.comparison).expanduser(),
            output_dir=Path(args.output_dir).expanduser(),
            engine_name=str(args.engine),
            gb10_url=str(args.gb10_url),
            gb10_model=str(args.gb10_model),
        )
    elif args.command == "collect-failures-many":
        result = collect_failure_records_many(
            [Path(item).expanduser() for item in list(args.comparisons or [])],
            output_dir=Path(args.output_dir).expanduser(),
            engine_name=str(args.engine),
            gb10_url=str(args.gb10_url),
            gb10_model=str(args.gb10_model),
        )
    elif args.command == "train-field-ranker":
        result = train_field_ranker(
            Path(args.dataset).expanduser(),
            output_dir=Path(args.output_dir).expanduser(),
            epochs=int(args.epochs),
            learning_rate=float(args.learning_rate),
        )
    elif args.command == "evaluate-field-ranker":
        result = evaluate_field_ranker(
            Path(args.dataset).expanduser(),
            Path(args.model).expanduser(),
            output_dir=Path(args.output_dir).expanduser(),
        )
    elif args.command == "export-layout":
        result = export_layout_region_examples(Path(args.dataset).expanduser(), output_dir=Path(args.output_dir).expanduser())
    elif args.command == "raw-ocr-gate":
        result = raw_ocr_training_gate(Path(args.comparison).expanduser(), output_path=Path(args.output).expanduser())
    elif args.command == "gb10-augment":
        result = augment_with_gb10_teacher(
            Path(args.dataset).expanduser(),
            output_dir=Path(args.output_dir).expanduser(),
            gb10_url=str(args.gb10_url),
            gb10_model=str(args.gb10_model),
            limit=int(args.limit),
        )
    else:
        raise ValueError(f"unknown_command:{args.command}")
    print(json.dumps(result, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
