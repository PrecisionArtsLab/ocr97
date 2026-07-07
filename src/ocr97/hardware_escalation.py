from __future__ import annotations

from typing import Any, Dict, Mapping, Optional


LOW_CONFIDENCE_SCORE = 85


def _safe_int(value: Any, default: int = 0) -> int:
    try:
        return int(value)
    except Exception:
        return int(default)


def _bool(value: Any) -> bool:
    if isinstance(value, bool):
        return value
    if value is None:
        return False
    return str(value).strip().lower() in {"1", "true", "yes", "on", "ready", "available"}


def decide_hardware_escalation(
    ocr_result: Mapping[str, Any],
    hardware: Optional[Mapping[str, Any]] = None,
    *,
    low_confidence_score: int = LOW_CONFIDENCE_SCORE,
) -> Dict[str, Any]:
    """Classify whether a document result should escalate to the hard OCR lane."""

    hardware = dict(hardware or {})
    gpu_ready = _bool(hardware.get("gpu_ready") or hardware.get("cuda_ready") or hardware.get("vlm_lane_ready"))
    score = _safe_int(ocr_result.get("score") or ocr_result.get("score_avg") or ocr_result.get("confidence_score"), 0)
    low_confidence = _bool(ocr_result.get("low_confidence")) or score < int(low_confidence_score)
    missed_fields = ocr_result.get("missed_fields") or ocr_result.get("field_misses") or []
    if isinstance(missed_fields, str):
        missed_fields = [missed_fields]
    table_gap = _bool(ocr_result.get("table_row_gap")) or _safe_int(ocr_result.get("expected_table_rows"), 0) > _safe_int(
        ocr_result.get("actual_table_rows"), 0
    )
    degraded = _bool(ocr_result.get("degraded")) or str(ocr_result.get("variant") or "").lower() in {
        "noisy_scan",
        "rotated",
        "hard",
        "degraded",
    }

    reasons = []
    if low_confidence:
        reasons.append("low_confidence")
    if missed_fields:
        reasons.append("field_miss")
    if table_gap:
        reasons.append("table_row_gap")
    if degraded:
        reasons.append("degraded_input")

    should_escalate = bool(reasons)
    if should_escalate and gpu_ready:
        action = "route_to_3090_hard_document_lane"
        category = "gpu_escalation"
        recoverable = True
    elif should_escalate:
        action = "record_gpu_lane_gap_and_use_best_cpu_preprocessing"
        category = "gpu_lane_unavailable"
        recoverable = False
    else:
        action = "keep_cpu_preprocessed_lane"
        category = "no_escalation_needed"
        recoverable = True

    return {
        "should_escalate": should_escalate,
        "action": action,
        "category": category,
        "recoverable": recoverable,
        "score": score,
        "threshold": int(low_confidence_score),
        "reasons": reasons,
        "gpu_ready": gpu_ready,
        "evidence": {
            "missed_fields": list(missed_fields or [])[:12],
            "table_gap": table_gap,
            "degraded": degraded,
            "hardware": {
                "gpu_name": hardware.get("gpu_name") or hardware.get("name") or "",
                "vlm_lane_ready": _bool(hardware.get("vlm_lane_ready")),
            },
        },
    }
