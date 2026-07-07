from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Optional


def _load_json(path: Path) -> Dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _iter_result_files(run_dir: Path) -> Iterable[Path]:
    for path in sorted(run_dir.rglob("*.json")):
        name = path.name.lower()
        if name in {"progress.json", "mixed_corpus_summary.json"}:
            continue
        if name.endswith("_report.json") or name.endswith("_audit.json"):
            continue
        try:
            payload = _load_json(path)
        except Exception:
            continue
        if isinstance(payload.get("results"), list):
            yield path


def _field_rows(score: Mapping[str, Any]) -> List[Dict[str, Any]]:
    return [dict(row) for row in list(score.get("fields") or []) if isinstance(row, dict)]


def _case_findings(row: Mapping[str, Any]) -> List[str]:
    findings: List[str] = []
    score = dict(row.get("score") or {})
    score_value = int(score.get("score") or 0)
    fields = _field_rows(score)
    missed_fields = [str(field.get("name") or "") for field in fields if not field.get("matched")]
    expected_rows = score.get("expected_table_rows")
    actual_rows = int(score.get("actual_table_rows") or 0)

    if score_value < 75:
        findings.append("below_75")
    if missed_fields:
        findings.append("field_miss:" + ",".join(missed_fields))
    if expected_rows is not None and actual_rows < int(expected_rows or 0):
        findings.append(f"table_row_gap:{actual_rows}/{expected_rows}")
    if fields and not bool(row.get("field_consensus_used")):
        findings.append("field_consensus_not_used")
    if float(row.get("latency_ms") or 0.0) > 30000:
        findings.append("latency_gt_30s")
    if int(row.get("status_code") or 0) >= 400:
        findings.append(f"status_{row.get('status_code')}")
    return findings


def audit_run_dir(run_dir: Path) -> Dict[str, Any]:
    result_files = list(_iter_result_files(run_dir))
    engine_rows: List[Dict[str, Any]] = []
    weak_cases: Dict[str, Dict[str, Any]] = {}
    reason_counts: Dict[str, int] = {}

    for result_file in result_files:
        payload = _load_json(result_file)
        results = [dict(row) for row in list(payload.get("results") or []) if isinstance(row, dict)]
        scores = [int((row.get("score") or {}).get("score") or 0) for row in results]
        latencies = [float(row.get("latency_ms") or 0.0) for row in results]
        engine_rows.append(
            {
                "file": str(result_file),
                "mode": str(payload.get("mode") or ""),
                "case_count": len(results),
                "score_avg": int(payload.get("score_avg") or 0),
                "below_75": sum(1 for score in scores if score < 75),
                "perfect": sum(1 for score in scores if score >= 100),
                "latency_avg_ms": round(sum(latencies) / float(len(latencies)), 2) if latencies else 0.0,
                "latency_max_ms": round(max(latencies), 2) if latencies else 0.0,
            }
        )
        for row in results:
            case_id = str(row.get("id") or "").strip()
            if not case_id:
                continue
            findings = _case_findings(row)
            if not findings:
                continue
            for finding in findings:
                key = finding.split(":", 1)[0]
                reason_counts[key] = reason_counts.get(key, 0) + 1
            score_value = int((row.get("score") or {}).get("score") or 0)
            existing = weak_cases.get(case_id)
            if existing is None or score_value < int(existing.get("lowest_score") or 101):
                weak_cases[case_id] = {
                    "case_id": case_id,
                    "lowest_score": score_value,
                    "engine": str(row.get("engine") or row.get("selected_engine") or ""),
                    "variant": str(row.get("variant") or ""),
                    "findings": findings,
                    "artifact_path": str(row.get("artifact_path") or ""),
                }

    weakest = sorted(weak_cases.values(), key=lambda row: (int(row.get("lowest_score") or 0), str(row.get("case_id") or "")))
    return {
        "run_dir": str(run_dir),
        "result_file_count": len(result_files),
        "engine_rows": engine_rows,
        "weak_case_count": len(weakest),
        "weakest_cases": weakest[:25],
        "reason_counts": dict(sorted(reason_counts.items())),
        "recommendations": _recommendations(reason_counts),
    }


def _recommendations(reason_counts: Mapping[str, int]) -> List[str]:
    recs: List[str] = []
    if int(reason_counts.get("field_miss") or 0) > 0:
        recs.append("Prioritize field consensus and crop-level rereads for fields that miss across otherwise readable pages.")
    if int(reason_counts.get("table_row_gap") or 0) > 0:
        recs.append("Prioritize table structure recovery with row/column detection before more whole-page OCR attempts.")
    if int(reason_counts.get("below_75") or 0) > 0:
        recs.append("Route below-75 image cases through the 3090 hard-document lane and compare against local preprocessing.")
    if int(reason_counts.get("latency_gt_30s") or 0) > 0:
        recs.append("Keep 3090 VLM work crop-gated; whole-page heavy inference is too expensive for routine use.")
    if not recs:
        recs.append("No dominant failure class found; expand the real-document corpus before raising the public grade.")
    return recs


def write_report(audit: Mapping[str, Any], output_path: Path) -> None:
    lines = [
        "# OCR97 3090 Capability Audit",
        "",
        f"- Run dir: `{audit.get('run_dir')}`",
        f"- Result files: `{audit.get('result_file_count')}`",
        f"- Weak cases: `{audit.get('weak_case_count')}`",
        "",
        "## Engine Summary",
        "",
        "| Mode | Cases | Avg | Perfect | Below 75 | Avg ms | Max ms |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ]
    for row in list(audit.get("engine_rows") or []):
        lines.append(
            f"| {row.get('mode')} | {row.get('case_count')} | {row.get('score_avg')} | "
            f"{row.get('perfect')} | {row.get('below_75')} | {row.get('latency_avg_ms')} | {row.get('latency_max_ms')} |"
        )
    lines.extend(["", "## Failure Classes", "", "| Reason | Count |", "|---|---:|"])
    for reason, count in dict(audit.get("reason_counts") or {}).items():
        lines.append(f"| {reason} | {count} |")
    lines.extend(["", "## Weakest Cases", "", "| Case | Lowest | Engine | Variant | Findings |", "|---|---:|---|---|---|"])
    for row in list(audit.get("weakest_cases") or []):
        findings = ", ".join(str(item) for item in list(row.get("findings") or []))
        lines.append(f"| {row.get('case_id')} | {row.get('lowest_score')} | {row.get('engine')} | {row.get('variant')} | {findings} |")
    lines.extend(["", "## Recommendations", ""])
    for item in list(audit.get("recommendations") or []):
        lines.append(f"- {item}")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main(argv: Optional[List[str]] = None) -> int:
    parser = argparse.ArgumentParser(description="Audit OCR97 benchmark artifacts for 3090 capability gaps.")
    parser.add_argument("--run-dir", required=True, help="Directory containing mixed corpus run artifacts.")
    parser.add_argument("--output", default="", help="Markdown report output path.")
    parser.add_argument("--json-output", default="", help="JSON report output path.")
    args = parser.parse_args(argv)

    audit = audit_run_dir(Path(args.run_dir).expanduser())
    if args.output:
        write_report(audit, Path(args.output).expanduser())
    if args.json_output:
        out = Path(args.json_output).expanduser()
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(json.dumps(audit, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(audit, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
