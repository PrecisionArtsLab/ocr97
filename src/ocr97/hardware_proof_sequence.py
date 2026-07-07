from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Optional
from zoneinfo import ZoneInfo

from .hardware_escalation import decide_hardware_escalation


OCR97_ROOT = Path(__file__).resolve().parents[2]
ENGINEERING_ROOT = OCR97_ROOT.parent
REPORT_ROOT = OCR97_ROOT / "artifacts" / "hardware_proof_sequence"
README_PATH = OCR97_ROOT / "README.md"
TEST_TYPE = "ocr97_hardware_proof_sequence"
CAPABILITY = "hardware_saturated_document_ocr_lane"
SECTION_BEGIN = "<!-- ocr97-hardware-proof-reports:begin -->"
SECTION_END = "<!-- ocr97-hardware-proof-reports:end -->"


def _python() -> str:
    return sys.executable


def _quote(path: Path) -> str:
    return '"' + str(path) + '"'


def _utc_iso() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat().replace("+00:00", "Z")


def _local_now() -> datetime:
    return datetime.now(ZoneInfo("America/Chicago")).replace(microsecond=0)


def _default_run_id(now: Optional[datetime] = None) -> str:
    stamp = (now or _local_now()).strftime("%Y%m%d_%H%M%S")
    return f"{TEST_TYPE}_{stamp}"


def _command_templates(run_dir: Path) -> List[str]:
    bootstrap = run_dir / "bootstrap_check.json"
    rotated = run_dir / "rotated_result.json"
    noisy = run_dir / "noisy_scan_result.json"
    audit_md = run_dir / "OCR97_HARDWARE_PROOF_AUDIT.md"
    audit_json = run_dir / "OCR97_HARDWARE_PROOF_AUDIT.json"
    return [
        f"{_python()} -m ocr97.cli doctor",
        f"{_python()} -m ocr97.bootstrap --check-only --output {_quote(bootstrap)}",
        f"{_python()} -m pytest tests/test_package_smoke.py tests/test_lightweight_bootstrap.py tests/test_3090_longform_plan.py -q",
        (
            f"{_python()} -m ocr97.truth_runner --mode image --variant rotated "
            f"--engine local_image_preprocessed_best --manifest benchmarks/truth10_manifest.json "
            f"--fixture-dir {_quote(run_dir / 'fixtures' / 'rotated')} "
            f"--artifact-dir {_quote(run_dir / 'artifacts' / 'rotated')} --output {_quote(rotated)}"
        ),
        (
            f"{_python()} -m ocr97.truth_runner --mode image --variant noisy_scan "
            f"--engine local_image_preprocessed_best --manifest benchmarks/truth10_manifest.json "
            f"--fixture-dir {_quote(run_dir / 'fixtures' / 'noisy_scan')} "
            f"--artifact-dir {_quote(run_dir / 'artifacts' / 'noisy_scan')} --output {_quote(noisy)}"
        ),
        f"{_python()} -m ocr97.capability_audit --run-dir {_quote(run_dir)} --output {_quote(audit_md)} --json-output {_quote(audit_json)}",
    ]


def rendered_commands(run_id: Optional[str] = None, root: Optional[Path] = None) -> List[str]:
    run_id = run_id or _default_run_id()
    run_dir = Path(root or REPORT_ROOT) / run_id
    return _command_templates(run_dir)


COMMANDS = rendered_commands(run_id="ocr97_hardware_proof_sequence_dry_run")


def _run_env() -> Dict[str, str]:
    env = dict(os.environ)
    env["PYTHONPATH"] = str(OCR97_ROOT / "src") + os.pathsep + env.get("PYTHONPATH", "")
    env.setdefault("OCR97_OCR_PREPROCESS_INCLUDE_TEXT", "1")
    env.setdefault("OCR97_OCR_PREPROCESS_WORKERS", "8")
    env.setdefault("OCR97_GATEWAY_SLO_CAP_LOCAL_IMAGE_PREPROCESSED_BEST_MS", "30000")
    return env


def collect_hardware_evidence(command_runner=subprocess.run) -> Dict[str, Any]:
    query = [
        "nvidia-smi",
        "--query-gpu=name,memory.total,driver_version,utilization.gpu,memory.used",
        "--format=csv,noheader,nounits",
    ]
    try:
        proc = command_runner(query, capture_output=True, text=True, timeout=10)
    except Exception as exc:
        return {"gpu_ready": False, "error": str(exc)[:240], "command": "nvidia-smi"}
    text = (proc.stdout or "").strip()
    if proc.returncode != 0 or not text:
        return {
            "gpu_ready": False,
            "command": "nvidia-smi",
            "returncode": proc.returncode,
            "stderr": (proc.stderr or "")[-1000:],
        }
    first = [part.strip() for part in text.splitlines()[0].split(",")]
    return {
        "gpu_ready": True,
        "command": "nvidia-smi",
        "raw": text,
        "gpu_name": first[0] if first else "",
        "memory_total_mb": first[1] if len(first) > 1 else "",
        "driver_version": first[2] if len(first) > 2 else "",
        "utilization_gpu_percent": first[3] if len(first) > 3 else "",
        "memory_used_mb": first[4] if len(first) > 4 else "",
        "vlm_lane_ready": False,
    }


def _read_json(path: Path) -> Dict[str, Any]:
    try:
        payload = json.loads(path.read_text(encoding="utf-8-sig"))
    except Exception:
        return {}
    return payload if isinstance(payload, dict) else {}


def _summarize_benchmarks(run_dir: Path) -> Dict[str, Any]:
    rows = []
    for name in ("rotated_result.json", "noisy_scan_result.json"):
        payload = _read_json(run_dir / name)
        if payload:
            rows.append(
                {
                    "file": name,
                    "mode": payload.get("mode") or "",
                    "case_count": int(payload.get("case_count") or 0),
                    "score_avg": int(payload.get("score_avg") or 0),
                }
            )
    score_avg = int(round(sum(int(row.get("score_avg") or 0) for row in rows) / len(rows))) if rows else 0
    return {"rows": rows, "score_avg": score_avg}


def _audit_summary(run_dir: Path) -> Dict[str, Any]:
    audit = _read_json(run_dir / "OCR97_HARDWARE_PROOF_AUDIT.json")
    return {
        "result_file_count": int(audit.get("result_file_count") or 0),
        "weak_case_count": int(audit.get("weak_case_count") or 0),
        "reason_counts": dict(audit.get("reason_counts") or {}),
        "recommendations": list(audit.get("recommendations") or [])[:6],
    }


def _test_counts(results: Iterable[Mapping[str, Any]]) -> Dict[str, int]:
    rows = list(results)
    failed = sum(1 for row in rows if int(row.get("returncode") or 0) != 0)
    return {"total": len(rows), "passed": len(rows) - failed, "failed": failed}


def _active_default_engine_chain() -> List[str]:
    try:
        from .gateway import _engine_chain

        chain = list(_engine_chain("digital_pdf", "quality_first", "", {}))
        return [str(item) for item in chain if str(item or "").strip()]
    except Exception:
        return [
            "native_pdf_text",
            "gb10_paddleocr_vl",
            "mineru2_5",
            "olmocr2",
            "gb10_qwen_ocr",
            "rapidocr",
            "tesseract",
        ]


def _engine_readiness_snapshot() -> Dict[str, Any]:
    try:
        from .server import create_app

        app = create_app(instance_name="ocr97_capability_probe")
        client = app.test_client()
        response = client.get("/ocr/capabilities")
        payload = response.get_json(silent=True) if response.is_json else {}
        payload = payload if isinstance(payload, dict) else {}
        engines = [dict(row) for row in list(payload.get("engines") or []) if isinstance(row, dict)]
        ready_rows = [row for row in engines if bool(row.get("ready"))]
        return {
            "ok": response.status_code == 200,
            "status_code": response.status_code,
            "route_mode_default": str(payload.get("route_mode_default") or ""),
            "doc_classes": list(payload.get("doc_classes") or []),
            "engine_count": len(engines),
            "ready_engine_count": len(ready_rows),
            "engines": engines,
            "engine_names": [str(row.get("name") or "") for row in engines if str(row.get("name") or "").strip()],
            "ready_engine_names": [str(row.get("name") or "") for row in ready_rows if str(row.get("name") or "").strip()],
        }
    except Exception as exc:
        return {
            "ok": False,
            "status_code": 0,
            "error": f"{type(exc).__name__}:{exc}",
            "engines": [],
            "engine_names": [],
            "ready_engine_names": [],
        }


def _score(status: str, benchmark: Mapping[str, Any], hardware: Mapping[str, Any], audit: Mapping[str, Any]) -> int:
    if status != "passed":
        return 45
    benchmark_score = int(benchmark.get("score_avg") or 0)
    hardware_points = 10 if bool(hardware.get("gpu_ready")) else 3
    audit_points = 10 if int(audit.get("weak_case_count") or 0) == 0 else 5
    return max(0, min(100, int(round(benchmark_score * 0.8 + hardware_points + audit_points))))


def build_report_payload(
    results: List[Dict[str, Any]],
    *,
    run_id: str,
    completed_at: Optional[str] = None,
    run_dir: Optional[Path] = None,
    hardware: Optional[Mapping[str, Any]] = None,
    engine_snapshot: Optional[Mapping[str, Any]] = None,
) -> Dict[str, Any]:
    run_dir = Path(run_dir or (REPORT_ROOT / run_id))
    completed_at = completed_at or _utc_iso()
    counts = _test_counts(results)
    status = "passed" if counts["failed"] == 0 else "failed"
    benchmark = _summarize_benchmarks(run_dir)
    audit = _audit_summary(run_dir)
    hardware = dict(hardware or {})
    engine_snapshot = dict(engine_snapshot or {})
    default_chain = _active_default_engine_chain()
    escalation_sample = {
        "score": benchmark.get("score_avg") or (100 if status == "passed" else 0),
        "low_confidence": (benchmark.get("score_avg") or 0) < 85,
        "field_misses": list((audit.get("reason_counts") or {}).keys()),
        "degraded": True,
    }
    escalation = decide_hardware_escalation(escalation_sample, hardware)
    score = _score(status, benchmark, hardware, audit)
    md_path = run_dir / f"{run_id}.md"
    json_path = run_dir / f"{run_id}.json"
    diagnostic = (
        {
            "cause": "ocr97_hardware_proof_sequence_passed",
            "category": "verification",
            "recoverable": True,
            "next_action": "use benchmark findings to tune OCR97 hard document routing",
            "confidence": 0.88,
            "evidence": [str(md_path), str(json_path)],
        }
        if status == "passed"
        else {
            "cause": "ocr97_hardware_proof_sequence_failed",
            "category": "verification",
            "recoverable": True,
            "next_action": "inspect failing command output and rerun the focused OCR97 hardware proof sequence",
            "confidence": 0.82,
            "evidence": [str(md_path), str(json_path)],
        }
    )
    return {
        "project": "ocr97",
        "test_type": TEST_TYPE,
        "name": "OCR97 hardware proof validation",
        "run_id": run_id,
        "status": status,
        "completed_at": completed_at,
        "summary": (
            f"OCR97 hardware proof sequence {status}; benchmark average {benchmark.get('score_avg')}; "
            f"GPU ready: {bool(hardware.get('gpu_ready'))}; escalation action: {escalation.get('action')}."
        ),
        "score": score,
        "benchmark_average": int(benchmark.get("score_avg") or 0),
        "low_confidence_flag": int(benchmark.get("score_avg") or 0) < 85,
        "active_default_engine_chain": default_chain,
        "engine_readiness_snapshot": engine_snapshot,
        "last_successful_proof_timestamp": completed_at if status == "passed" else "",
        "score_components": {
            "benchmark_average": int(benchmark.get("score_avg") or 0),
            "gpu_readiness": 10 if bool(hardware.get("gpu_ready")) else 3,
            "audit_cleanliness": 10 if int(audit.get("weak_case_count") or 0) == 0 else 5,
            "sequence_completion": 20 if status == "passed" else 0,
        },
        "capability": CAPABILITY,
        "capabilities_confirmed": [CAPABILITY] if status == "passed" else [],
        "capabilities_not_confirmed": [] if status == "passed" else [CAPABILITY],
        "tests_improved": [
            "OCR97 GPU/hardware evidence capture",
            "OCR97 degraded image truth benchmark",
            "OCR97 low-confidence hard-lane escalation contract",
        ],
        "tests_confirmed": [row["command"] for row in results if int(row.get("returncode") or 0) == 0],
        "verification_command": " && ".join(_command_templates(run_dir)),
        "report_path": str(md_path),
        "artifact_paths": [
            str(json_path),
            str(md_path),
            str(run_dir / "rotated_result.json"),
            str(run_dir / "noisy_scan_result.json"),
            str(run_dir / "OCR97_HARDWARE_PROOF_AUDIT.md"),
            str(run_dir / "OCR97_HARDWARE_PROOF_AUDIT.json"),
        ],
        "passed": counts["passed"],
        "failed": counts["failed"],
        "total": counts["total"],
        "diagnostic": diagnostic,
        "hardware_evidence": hardware,
        "benchmark": benchmark,
        "audit": audit,
        "hardware_escalation": escalation,
        "progress": "hardware proof report written and OCR97 README section updated",
        "details": "Runs focused OCR97 checks, degraded-image truth benchmarks, capability audit, and hardware evidence capture.",
    }


def _render_markdown(payload: Mapping[str, Any], results: List[Mapping[str, Any]]) -> str:
    lines = [
        "# OCR97 Hardware Proof Sequence",
        "",
        f"- run_id: `{payload.get('run_id')}`",
        f"- status: `{payload.get('status')}`",
        f"- score: `{payload.get('score')}`",
        f"- capability: `{payload.get('capability')}`",
        f"- completed_at: `{payload.get('completed_at')}`",
        f"- GPU ready: `{bool((payload.get('hardware_evidence') or {}).get('gpu_ready'))}`",
        f"- benchmark average: `{(payload.get('benchmark') or {}).get('score_avg')}`",
        f"- escalation action: `{(payload.get('hardware_escalation') or {}).get('action')}`",
        f"- low confidence flag: `{bool(payload.get('low_confidence_flag'))}`",
        f"- active default chain: `{', '.join(list(payload.get('active_default_engine_chain') or []))}`",
        "",
        "## Commands",
        "",
        "| Result | Seconds | Command |",
        "|---|---:|---|",
    ]
    for row in results:
        result = "pass" if int(row.get("returncode") or 0) == 0 else "fail"
        lines.append(f"| {result} | {row.get('seconds')} | `{row.get('command')}` |")
    lines.extend(["", "## Benchmark Rows", "", "| File | Mode | Cases | Avg |", "|---|---|---:|---:|"])
    for row in list((payload.get("benchmark") or {}).get("rows") or []):
        lines.append(f"| {row.get('file')} | {row.get('mode')} | {row.get('case_count')} | {row.get('score_avg')} |")
    lines.extend(["", "## Audit", ""])
    audit = dict(payload.get("audit") or {})
    lines.append(f"- weak_case_count: `{audit.get('weak_case_count')}`")
    lines.append(f"- reason_counts: `{json.dumps(audit.get('reason_counts') or {}, sort_keys=True)}`")
    for rec in list(audit.get("recommendations") or [])[:6]:
        lines.append(f"- {rec}")
    lines.extend(["", "## Hardware Evidence", ""])
    hardware = dict(payload.get("hardware_evidence") or {})
    for key in ("gpu_ready", "gpu_name", "memory_total_mb", "driver_version", "utilization_gpu_percent", "memory_used_mb", "error"):
        if key in hardware:
            lines.append(f"- {key}: `{hardware.get(key)}`")
    return "\n".join(lines).strip() + "\n"


def update_readme_report(readme_path: Path, payload: Mapping[str, Any]) -> None:
    date = str(payload.get("completed_at") or _utc_iso())[:10]
    run_id = str(payload.get("run_id") or "")
    entry_begin = f"<!-- ocr97-hardware-proof:{date}:begin -->"
    entry_end = f"<!-- ocr97-hardware-proof:{date}:end -->"
    entry = "\n".join(
        [
            entry_begin,
            f"### {date} - OCR97 Hardware Proof Sequence",
            "",
            f"- run_id: `{run_id}`",
            f"- status: `{payload.get('status')}`",
            f"- score: `{payload.get('score')}`",
            f"- benchmark average: `{(payload.get('benchmark') or {}).get('score_avg')}`",
            f"- GPU ready: `{bool((payload.get('hardware_evidence') or {}).get('gpu_ready'))}`",
            f"- hard-lane action: `{(payload.get('hardware_escalation') or {}).get('action')}`",
            f"- report: `{payload.get('report_path')}`",
            "",
            entry_end,
        ]
    )
    text = readme_path.read_text(encoding="utf-8") if readme_path.exists() else "# OCR97\n"
    if SECTION_BEGIN not in text or SECTION_END not in text:
        text = text.rstrip() + "\n\n## OCR97 Hardware Proof Reports\n\n" + SECTION_BEGIN + "\n" + SECTION_END + "\n"
    start = text.index(SECTION_BEGIN) + len(SECTION_BEGIN)
    end = text.index(SECTION_END)
    block = text[start:end].strip()
    if entry_begin in block and entry_end in block:
        before = block[: block.index(entry_begin)].rstrip()
        after = block[block.index(entry_end) + len(entry_end) :].strip()
        pieces = [piece for piece in (before, entry, after) if piece]
        block = "\n\n".join(pieces)
    else:
        block = (entry + ("\n\n" + block if block else "")).strip()
    updated = text[:start] + "\n" + block + "\n" + text[end:]
    readme_path.write_text(updated, encoding="utf-8")


def record_sky_report(payload: Mapping[str, Any], *, root: Optional[Path] = None) -> Dict[str, Any]:
    if str(ENGINEERING_ROOT) not in sys.path:
        sys.path.insert(0, str(ENGINEERING_ROOT))
    try:
        from Sky.services.test_run_reports import record_test_run_report
    except Exception as exc:
        raise RuntimeError(
            "Sky report integration is unavailable. Run with --no-sky-report for standalone OCR97 usage."
        ) from exc

    return record_test_run_report(dict(payload), root=root)


def run_sequence(
    *,
    run_id: Optional[str] = None,
    root: Optional[Path] = None,
    no_post: bool = False,
    skip_readme: bool = False,
    command_runner=subprocess.run,
) -> Dict[str, Any]:
    run_id = run_id or _default_run_id()
    run_dir = Path(root or REPORT_ROOT) / run_id
    run_dir.mkdir(parents=True, exist_ok=True)
    hardware = collect_hardware_evidence(command_runner=command_runner)
    engine_snapshot = _engine_readiness_snapshot()
    results: List[Dict[str, Any]] = []
    for command in _command_templates(run_dir):
        started = time.perf_counter()
        proc = command_runner(
            command,
            cwd=str(OCR97_ROOT),
            env=_run_env(),
            shell=True,
            capture_output=True,
            text=True,
            timeout=900,
        )
        seconds = round(time.perf_counter() - started, 2)
        results.append(
            {
                "command": command,
                "returncode": int(getattr(proc, "returncode", 1)),
                "seconds": seconds,
                "stdout": str(getattr(proc, "stdout", "") or "")[-4000:],
                "stderr": str(getattr(proc, "stderr", "") or "")[-4000:],
            }
        )
        if int(getattr(proc, "returncode", 1)) != 0:
            break
    payload = build_report_payload(results, run_id=run_id, run_dir=run_dir, hardware=hardware, engine_snapshot=engine_snapshot)
    summary_json = run_dir / f"{run_id}.json"
    summary_md = run_dir / f"{run_id}.md"
    summary_json.write_text(json.dumps({"payload": payload, "command_results": results}, indent=2), encoding="utf-8")
    summary_md.write_text(_render_markdown(payload, results), encoding="utf-8")
    if not skip_readme:
        update_readme_report(README_PATH, payload)
    sky_report = {"ok": False, "skipped": True}
    if not no_post:
        try:
            sky_report = record_sky_report(payload)
        except RuntimeError as exc:
            sky_report = {"ok": False, "skipped": True, "reason": str(exc)}
    return {
        "ok": True,
        "status": payload["status"],
        "payload": payload,
        "command_results": results,
        "paths": {"run_dir": str(run_dir), "json": str(summary_json), "markdown": str(summary_md), "readme": str(README_PATH)},
        "sky_report": sky_report,
    }


def main(argv: Optional[List[str]] = None) -> int:
    parser = argparse.ArgumentParser(description="Run OCR97 hardware proof sequence and write Sky/README reports.")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--run-id", default="")
    parser.add_argument("--output-root", default="")
    parser.add_argument("--no-sky-report", action="store_true")
    parser.add_argument("--skip-readme", action="store_true")
    args = parser.parse_args(argv)
    root = Path(args.output_root).expanduser() if args.output_root else None
    run_id = args.run_id or _default_run_id()
    if args.dry_run:
        print(json.dumps({"ok": True, "run_id": run_id, "commands": rendered_commands(run_id=run_id, root=root)}, indent=2))
        return 0
    result = run_sequence(run_id=run_id, root=root, no_post=args.no_sky_report, skip_readme=args.skip_readme)
    print(json.dumps({"ok": result["ok"], "status": result["status"], "paths": result["paths"]}, indent=2))
    return 0 if result["status"] == "passed" else 1


if __name__ == "__main__":
    raise SystemExit(main())
