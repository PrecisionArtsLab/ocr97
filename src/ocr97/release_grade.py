from __future__ import annotations

import argparse
import json
from pathlib import Path
from statistics import median
from typing import Any, Dict, Iterable, List, Mapping, Optional


DEFAULT_THRESHOLDS = {
    "target_grade": 97,
    "min_cases": 100,
    "min_categories": 10,
    "min_best_avg": 95,
    "min_worst_avg": 92,
    "max_below_75": 0,
    "max_latency_avg_ms": 15000,
    "max_latency_p95_ms": 30000,
}

PRODUCTION_ROUTE_ENGINES = {"auto", "native_pdf_text", "local_image_preprocessed_best", "local_image_best"}


def _load_json(path: Path) -> Dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _percentile(values: Iterable[float], pct: float) -> float:
    ordered = sorted(float(value) for value in values)
    if not ordered:
        return 0.0
    if len(ordered) == 1:
        return ordered[0]
    rank = (len(ordered) - 1) * pct
    lower = int(rank)
    upper = min(lower + 1, len(ordered) - 1)
    weight = rank - lower
    return ordered[lower] * (1 - weight) + ordered[upper] * weight


def _step_rows(summary: Mapping[str, Any]) -> List[Dict[str, Any]]:
    return [dict(step) for step in list(summary.get("steps") or []) if isinstance(step, dict)]


def _score_gate(ok: bool, points: int) -> int:
    return points if ok else 0


def _view_metrics(
    step_summaries: List[Dict[str, Any]],
    *,
    case_count: int,
    category_count: int,
    pending_runs: List[str],
    thresholds: Mapping[str, Any],
) -> Dict[str, Any]:
    latencies = [
        float(row.get("latency_avg_ms") or 0.0)
        for row in step_summaries
        if float(row.get("latency_avg_ms") or 0.0) > 0
    ]
    scores = [int(row.get("score_avg") or 0) for row in step_summaries]
    below_75 = sum(int(row.get("below_75_cases") or 0) for row in step_summaries)
    best_avg = max(scores) if scores else 0
    worst_avg = min(scores) if scores else 0
    latency_avg = round(sum(latencies) / float(len(latencies)), 2) if latencies else 0.0
    latency_p95 = round(_percentile(latencies, 0.95), 2) if latencies else 0.0

    gates = [
        {
            "name": "release_corpus_size",
            "points": 15,
            "ok": case_count >= int(thresholds["min_cases"]),
            "actual": case_count,
            "target": thresholds["min_cases"],
        },
        {
            "name": "document_breadth",
            "points": 10,
            "ok": category_count >= int(thresholds["min_categories"]),
            "actual": category_count,
            "target": thresholds["min_categories"],
        },
        {
            "name": "best_pipeline_quality",
            "points": 25,
            "ok": best_avg >= int(thresholds["min_best_avg"]),
            "actual": best_avg,
            "target": thresholds["min_best_avg"],
        },
        {
            "name": "worst_pipeline_floor",
            "points": 20,
            "ok": worst_avg >= int(thresholds["min_worst_avg"]),
            "actual": worst_avg,
            "target": thresholds["min_worst_avg"],
        },
        {
            "name": "no_bad_cases",
            "points": 10,
            "ok": below_75 <= int(thresholds["max_below_75"]),
            "actual": below_75,
            "target": thresholds["max_below_75"],
        },
        {
            "name": "latency_average",
            "points": 10,
            "ok": latency_avg <= float(thresholds["max_latency_avg_ms"]) if latencies else False,
            "actual": latency_avg,
            "target": thresholds["max_latency_avg_ms"],
        },
        {
            "name": "latency_p95",
            "points": 5,
            "ok": latency_p95 <= float(thresholds["max_latency_p95_ms"]) if latencies else False,
            "actual": latency_p95,
            "target": thresholds["max_latency_p95_ms"],
        },
        {
            "name": "queue_complete",
            "points": 5,
            "ok": not pending_runs,
            "actual": pending_runs,
            "target": "no pending runs",
        },
    ]
    grade = sum(_score_gate(bool(gate["ok"]), int(gate["points"])) for gate in gates)
    return {
        "grade": grade,
        "target_grade": thresholds["target_grade"],
        "passes_97_gate": grade >= int(thresholds["target_grade"]),
        "best_score_avg": best_avg,
        "worst_score_avg": worst_avg,
        "below_75_cases": below_75,
        "latency_avg_ms": latency_avg,
        "latency_p95_ms": latency_p95,
        "latency_median_ms": round(median(latencies), 2) if latencies else 0.0,
        "step_count": len(step_summaries),
        "case_count": case_count,
        "category_count": category_count,
        "gates": gates,
    }


def _engine_for_step(step: Mapping[str, Any]) -> str:
    summary = step.get("summary") or {}
    mode = str(summary.get("mode") or step.get("engine") or "")
    if "local_image_preprocessed_best" in mode:
        return "local_image_preprocessed_best"
    if "local_image_best" in mode:
        return "local_image_best"
    if "native_pdf_text" in mode:
        return "native_pdf_text"
    return str(step.get("engine") or "").strip()


def grade_release(summary_path: Path, *, manifest_path: Optional[Path] = None, queue_path: Optional[Path] = None) -> Dict[str, Any]:
    summary = _load_json(summary_path)
    thresholds = dict(DEFAULT_THRESHOLDS)
    steps = _step_rows(summary)
    step_summaries = [dict(step.get("summary") or {}) for step in steps if isinstance(step.get("summary"), dict)]
    case_count = int(summary.get("manifest_case_count") or 0)

    categories = set()
    if manifest_path and manifest_path.exists():
        manifest = _load_json(manifest_path)
        categories = {str(case.get("category") or "") for case in list(manifest.get("cases") or []) if isinstance(case, dict)}

    pending_runs: List[str] = []
    if queue_path and queue_path.exists():
        queue = _load_json(queue_path)
        for run in list(queue.get("runs") or []):
            if isinstance(run, dict) and str(run.get("status") or "") != "complete":
                pending_runs.append(str(run.get("id") or "unknown"))

    category_count = len(categories)
    all_lanes = _view_metrics(
        step_summaries,
        case_count=case_count,
        category_count=category_count,
        pending_runs=pending_runs,
        thresholds=thresholds,
    )
    auto_route_steps = [
        step
        for step in steps
        if str(step.get("benchmark_kind") or "") == "end_to_end_auto_route"
        and isinstance(step.get("summary"), dict)
    ]
    production_source_steps = auto_route_steps or [
        step
        for step in steps
        if isinstance(step.get("summary"), dict) and _engine_for_step(step) in PRODUCTION_ROUTE_ENGINES
    ]
    production_step_summaries = [dict(step.get("summary") or {}) for step in production_source_steps]
    fallback_step_summaries = [
        dict(step.get("summary") or {})
        for step in steps
        if isinstance(step.get("summary"), dict) and _engine_for_step(step) not in PRODUCTION_ROUTE_ENGINES
    ]
    production = _view_metrics(
        production_step_summaries,
        case_count=case_count,
        category_count=category_count,
        pending_runs=pending_runs,
        thresholds=thresholds,
    )
    fallback = _view_metrics(
        fallback_step_summaries,
        case_count=case_count,
        category_count=category_count,
        pending_runs=pending_runs,
        thresholds=thresholds,
    )
    corpus_type = str((manifest if manifest_path and manifest_path.exists() else {}).get("corpus_type") or "unspecified")
    forced_diagnostic_count = sum(
        1 for step in steps if str(step.get("benchmark_kind") or "") == "forced_engine_diagnostic"
    )
    _STRESS_NOTE = (
        "This view includes forced-engine compatibility diagnostics that deliberately bypass auto routing. "
        "Keep their failures as engine-specific evidence, but use end_to_end_auto_route steps for production claims."
    )
    all_lanes["corpus_type"] = corpus_type
    all_lanes["view_kind"] = "combined_with_forced_engine_diagnostics"
    all_lanes["forced_diagnostic_step_count"] = forced_diagnostic_count
    all_lanes["is_artificial_scenario"] = False
    all_lanes["corpus_note"] = _STRESS_NOTE if forced_diagnostic_count else ""
    fallback["corpus_type"] = corpus_type
    fallback["view_kind"] = "forced_engine_diagnostic"
    fallback["forced_diagnostic_step_count"] = forced_diagnostic_count
    fallback["is_artificial_scenario"] = False
    fallback["corpus_note"] = _STRESS_NOTE if forced_diagnostic_count else ""
    production["view_kind"] = "end_to_end_auto_route" if auto_route_steps else "legacy_production_proxy"
    return {
        "grade": production["grade"],
        "target_grade": thresholds["target_grade"],
        "passes_97_gate": production["passes_97_gate"],
        "summary": str(summary_path),
        "manifest": str(manifest_path or ""),
        "queue": str(queue_path or ""),
        "case_count": case_count,
        "category_count": category_count,
        "best_score_avg": production["best_score_avg"],
        "worst_score_avg": production["worst_score_avg"],
        "below_75_cases": production["below_75_cases"],
        "latency_avg_ms": production["latency_avg_ms"],
        "latency_p95_ms": production["latency_p95_ms"],
        "latency_median_ms": production["latency_median_ms"],
        "pending_runs": pending_runs,
        "gates": production["gates"],
        "views": {
            "all_lanes_stress": all_lanes,
            "production_router": production,
            "fallback_lane_stress": fallback,
        },
    }


def write_report(result: Mapping[str, Any], report_path: Path) -> None:
    views = dict(result.get("views") or {})
    production = dict(views.get("production_router") or {})
    fallback = dict(views.get("fallback_lane_stress") or {})
    lines = [
        "# OCR97 97-Grade Release Gate Report",
        "",
        f"- Grade: `{result.get('grade')}/100`",
        f"- Grade view: `production_router`",
        f"- Target: `{result.get('target_grade')}/100`",
        f"- Passes 97 gate: `{str(bool(result.get('passes_97_gate'))).lower()}`",
        f"- Cases: `{result.get('case_count')}`",
        f"- Categories: `{result.get('category_count')}`",
        f"- Best avg: `{result.get('best_score_avg')}`",
        f"- Worst avg: `{result.get('worst_score_avg')}`",
        f"- Below 75 cases: `{result.get('below_75_cases')}`",
        f"- Avg latency: `{result.get('latency_avg_ms')} ms`",
        f"- P95 latency: `{result.get('latency_p95_ms')} ms`",
        "",
        "## Grade Views",
        "",
        "| View | Grade | Step count | Best avg | Worst avg | Below 75 | Avg latency ms | P95 latency ms | Meaning |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---|",
        (
            f"| production_router | {result.get('grade')} | "
            f"{(views.get('production_router') or {}).get('step_count', '')} | "
            f"{result.get('best_score_avg')} | {result.get('worst_score_avg')} | "
            f"{result.get('below_75_cases')} | {result.get('latency_avg_ms')} | "
            f"{result.get('latency_p95_ms')} | **Primary grade.** Production-route evidence only: native PDF text plus local image routers. |"
        ),
        (
            f"| all_lanes_stress | {(views.get('all_lanes_stress') or {}).get('grade', '')} | "
            f"{(views.get('all_lanes_stress') or {}).get('step_count', '')} | "
            f"{(views.get('all_lanes_stress') or {}).get('best_score_avg', '')} | "
            f"{(views.get('all_lanes_stress') or {}).get('worst_score_avg', '')} | "
            f"{(views.get('all_lanes_stress') or {}).get('below_75_cases', '')} | "
            f"{(views.get('all_lanes_stress') or {}).get('latency_avg_ms', '')} | "
            f"{(views.get('all_lanes_stress') or {}).get('latency_p95_ms', '')} | "
            + ("**ARTIFICIAL SCENARIO** — corpus is native PDFs; Tesseract never processes these in production. |"
               if (views.get("all_lanes_stress") or {}).get("is_artificial_scenario") else
               "Diagnostic: combined view including forced-engine compatibility lanes. |")
        ),
        (
            f"| fallback_lane_stress | {fallback.get('grade', '')} | "
            f"{fallback.get('step_count', '')} | {fallback.get('best_score_avg', '')} | "
            f"{fallback.get('worst_score_avg', '')} | {fallback.get('below_75_cases', '')} | "
            f"{fallback.get('latency_avg_ms', '')} | {fallback.get('latency_p95_ms', '')} | "
            + ("**ARTIFICIAL SCENARIO** — corpus is native PDFs; Tesseract never processes these in production. See scanned_fallback_benchmark for real fallback grading. |"
               if fallback.get("is_artificial_scenario") else
               "Forced-engine compatibility lanes; useful for diagnosis and excluded from production claims. |")
        ),
        "",
        "The top-level grade uses the production-router view: native PDF text plus local image preprocessing routes. "
        "The all-lanes stress view is retained for diagnostic purposes but does not determine release gating.",
        *(["", f"> **Diagnostic view note:** {(views.get('all_lanes_stress') or {}).get('corpus_note', '')}"]
          if (views.get("all_lanes_stress") or {}).get("corpus_note") else []),
        "",
        "## Gates",
        "",
        "| Gate | Points | Status | Actual | Target |",
        "|---|---:|---|---:|---:|",
    ]
    for gate in list(result.get("gates") or []):
        status = "pass" if gate.get("ok") else "fail"
        actual = gate.get("actual")
        if isinstance(actual, list):
            actual = ", ".join(str(item) for item in actual) or "none"
        lines.append(f"| {gate.get('name')} | {gate.get('points')} | {status} | {actual} | {gate.get('target')} |")
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main(argv: Optional[List[str]] = None) -> int:
    parser = argparse.ArgumentParser(description="Grade an OCR97 benchmark summary against the 97-grade release gate.")
    parser.add_argument("--summary", required=True, help="mixed_corpus_summary.json path.")
    parser.add_argument("--manifest", default="", help="Release corpus manifest path.")
    parser.add_argument("--queue", default="", help="Queue JSON path.")
    parser.add_argument("--report", default="", help="Optional markdown report output path.")
    args = parser.parse_args(argv)

    result = grade_release(
        Path(args.summary).expanduser(),
        manifest_path=Path(args.manifest).expanduser() if args.manifest else None,
        queue_path=Path(args.queue).expanduser() if args.queue else None,
    )
    if args.report:
        write_report(result, Path(args.report).expanduser())
    print(json.dumps(result, indent=2))
    return 0 if bool(result["passes_97_gate"]) else 1


if __name__ == "__main__":
    raise SystemExit(main())
