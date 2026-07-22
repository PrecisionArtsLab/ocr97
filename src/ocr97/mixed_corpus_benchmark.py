from __future__ import annotations

import argparse
import json
import os
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Optional

from .truth_benchmark import load_manifest, score_manifest
from .truth_runner import run_gateway_image_truth_benchmark, run_gateway_truth_benchmark

_OLLAMA_URL = os.getenv("OLLAMA_BASE_URL", "http://127.0.0.1:11434")
_OLLAMA_MODEL = os.getenv("OLLAMA_MODEL", "gemma3:12b")


DEFAULT_FOCUS_IDS = [
    "vendor_invoice_services",
    "invoice_line_items",
    "bank_statement_monthly",
    "brokerage_activity",
    "insurance_eob",
    "shipping_manifest",
]


_KNOWN_VARIANT_SUFFIXES = (
    "baseline",
    "dense",
    "rotated",
    "noisy_scan",
    "noisy",
    "small_text",
    "table_first",
    "mild_degraded",
    "low_contrast",
)


def _split_case_variant(raw: str) -> tuple[str, str]:
    value = str(raw or "").strip().lower()
    for suffix in _KNOWN_VARIANT_SUFFIXES:
        marker = f"_{suffix}"
        if value.endswith(marker):
            return value[: -len(marker)], suffix
    return value, ""


def _case_focus_match(case: Mapping[str, Any], focus_id: str) -> bool:
    wanted = str(focus_id or "").strip().lower()
    if not wanted:
        return False

    wanted_base, wanted_variant = _split_case_variant(wanted)
    case_id = str(case.get("id") or "").strip().lower()
    case_source = str(case.get("source_case_id") or "").strip().lower()
    case_base, case_variant = _split_case_variant(case_id)
    source_base, source_variant = _split_case_variant(case_source) if case_source else ("", "")

    if wanted == case_id or (case_source and wanted == case_source):
        return True

    if wanted_variant:
        if wanted_base == case_base and wanted_variant == case_variant:
            return True
        if not source_base:
            return False
        return wanted_base == source_base and not source_variant and wanted_variant in {"rotated", "noisy_scan", "table_first"}

    return wanted == case_base or (case_source and wanted == source_base)


def _utc_iso() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat().replace("+00:00", "Z")


def _split_csv(raw: str) -> List[str]:
    return [item.strip() for item in str(raw or "").split(",") if item.strip()]


def _select_cases(
    manifest: Mapping[str, Any],
    *,
    ids: Optional[Iterable[str]] = None,
    limit: int = 0,
    strict: bool = False,
) -> Dict[str, Any]:
    cases = [dict(row) for row in list(manifest.get("cases") or []) if isinstance(row, dict)]
    wanted = [str(item).strip() for item in list(ids or []) if str(item).strip()]
    if wanted:
        ordered: List[Dict[str, Any]] = []
        unresolved: List[str] = []
        seen_ids: set[str] = set()
        for raw_id in wanted:
            matched = False
            for case in cases:
                case_id = str(case.get("id") or "")
                if case_id in seen_ids:
                    continue
                if _case_focus_match(case, raw_id):
                    ordered.append(case)
                    seen_ids.add(case_id)
                    matched = True
            if not matched:
                unresolved.append(str(raw_id))
        if strict and unresolved:
            raise ValueError(f"focus_ids_not_found:{','.join(unresolved)}")
        if not ordered:
            ordered = cases
    else:
        ordered = cases
    if int(limit or 0) > 0:
        ordered = ordered[: int(limit)]
    return {**dict(manifest), "cases": ordered}


def _summarize_result(result: Mapping[str, Any]) -> Dict[str, Any]:
    rows = [dict(row) for row in list(result.get("results") or []) if isinstance(row, dict)]
    latencies = [float(row.get("latency_ms") or 0.0) for row in rows]
    scores = [int((row.get("score") or {}).get("score") or 0) for row in rows]
    attempted_engines: set[str] = set()
    selected_attempt_indices: set[int] = set()
    fallback_reasons: set[str] = set()
    conf_values: list[float] = []
    degraded_count = 0
    for row in rows:
        for raw in list(row.get("attempted_engines") or []):
            value = str(raw or "").strip()
            if value:
                attempted_engines.add(value)
        reason = str(row.get("fallback_reason") or "").strip()
        if reason:
            fallback_reasons.add(reason)
        if bool(row.get("degraded_fallback")):
            degraded_count += 1
        try:
            selected_attempt_indices.add(int(row.get("selected_attempt_index")))
        except (TypeError, ValueError):
            pass
        try:
            conf_values.append(float(row.get("confidence")))
        except (TypeError, ValueError):
            pass
    return {
        "mode": str(result.get("mode") or ""),
        "case_count": len(rows),
        "score_avg": int(result.get("score_avg") or 0),
        "perfect_cases": sum(1 for score in scores if score >= 100),
        "below_75_cases": sum(1 for score in scores if score < 75),
        "latency_avg_ms": round(sum(latencies) / float(len(latencies)), 2) if latencies else 0.0,
        "latency_max_ms": round(max(latencies), 2) if latencies else 0.0,
        "artifact_dir": str(result.get("artifact_dir") or ""),
        "attempted_engines": sorted(attempted_engines),
        "selected_attempt_indices": sorted(selected_attempt_indices),
        "fallback_reasons": sorted(fallback_reasons),
        "degraded_fallback_cases": degraded_count,
        "confidence_min": round(min(conf_values), 4) if conf_values else None,
        "confidence_max": round(max(conf_values), 4) if conf_values else None,
        "confidence_avg": round(sum(conf_values) / float(len(conf_values)), 4) if conf_values else None,
    }


def _llm_grade(summary: Mapping[str, Any]) -> str:
    try:
        import requests as _req
        model = _OLLAMA_MODEL
        step_lines = []
        for step in summary.get("steps") or []:
            s = step.get("summary") or {}
            retried = sum(1 for r in (step.get("retry_log") or []) if r.get("accepted"))
            retry_note = f", rotation_retries_accepted={retried}" if retried else ""
            step_lines.append(
                f"  {step['name']}: avg={s.get('score_avg','?')}, "
                f"perfect={s.get('perfect_cases','?')}/{s.get('case_count','?')}, "
                f"below75={s.get('below_75_cases','?')}, "
                f"latency_avg={s.get('latency_avg_ms','?')}ms{retry_note}"
            )
        prompt = (
            f"You are grading an OCR benchmark. The system was tested on "
            f"{summary.get('manifest_case_count', '?')} synthetic business documents "
            f"({summary.get('manifest_self_score', 100)}/100 manifest self-score) "
            f"across {len(step_lines)} pipeline configurations.\n\n"
            "Step results:\n" + "\n".join(step_lines) + "\n\n"
            f"Overall: best_score_avg={summary.get('best_score_avg')}, "
            f"worst_score_avg={summary.get('worst_score_avg')}, "
            f"steps_passed={summary.get('passed')}, steps_failed={summary.get('failed')}\n\n"
            "Grade this OCR system 0–100 (100=perfect extraction, 0=complete failure). Reply with:\n"
            "GRADE: <integer>\n"
            "STRENGTHS:\n- <bullet>\n- <bullet>\n"
            "WEAKNESSES:\n- <bullet>\n- <bullet>\n"
            "TOP RECOMMENDATION: <one sentence>"
        )
        def _call(m: str) -> Optional[str]:
            r = _req.post(f"{_OLLAMA_URL}/api/generate", json={"model": m, "prompt": prompt, "stream": False}, timeout=120)
            if r.ok:
                return str(r.json().get("response") or "").strip() or None
            return None
        text = _call(model)
        if text is None:
            tags = _req.get(f"{_OLLAMA_URL}/api/tags", timeout=5).json()
            first = ((tags.get("models") or [{}])[0]).get("name", "")
            if first and first != model:
                model = first
                text = _call(model)
        return text or "_LLM unavailable at grade time_"
    except Exception as exc:
        return f"_LLM grading failed: {type(exc).__name__}: {exc}_"


def _write_llm_grade_report(output_dir: Path, summary: Mapping[str, Any], llm_text: str) -> Path:
    lines = [
        "# OCR97 Mixed Corpus — LLM Graded Report",
        "",
        f"- Run started: `{summary.get('started_at', '')}`",
        f"- Run finished: `{summary.get('finished_at', '')}`",
        f"- Manifest cases: {summary.get('manifest_case_count', '?')} "
        f"(self-score: {summary.get('manifest_self_score', '?')}/100)",
        "",
        "## LLM Assessment",
        "",
        llm_text,
        "",
        "## Step Results",
        "",
        "| Step | Avg | Perfect | Below 75 | Latency avg | Retries accepted |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for step in summary.get("steps") or []:
        s = step.get("summary") or {}
        retried = sum(1 for r in (step.get("retry_log") or []) if r.get("accepted"))
        lines.append(
            f"| {step['name']} | {s.get('score_avg','')} "
            f"| {s.get('perfect_cases','')}/{s.get('case_count','')} "
            f"| {s.get('below_75_cases','')} "
            f"| {s.get('latency_avg_ms','')}ms "
            f"| {retried if retried else '—'} |"
        )
    lines += ["", "---", f"_Re-grade: `python -m ocr97.mixed_corpus_benchmark --regrade {output_dir}`_"]
    report_path = output_dir / "LLM_GRADED_REPORT.md"
    report_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return report_path


_SKY_URL = os.getenv("SKY_BASE_URL", "http://127.0.0.1:5011")


def _sky_calendar_create(title: str, notes: str, status: str) -> Optional[str]:
    """Create a Sky calendar event. Returns the event id or None on failure."""
    try:
        import requests as _req
        from datetime import timedelta
        now = datetime.now(timezone.utc).replace(microsecond=0)
        end = now + timedelta(hours=2)
        payload = {
            "title": title,
            "start": now.isoformat().replace("+00:00", "Z"),
            "end": end.isoformat().replace("+00:00", "Z"),
            "status": status,
            "source": "ocr97",
            "notes": notes,
        }
        r = _req.post(f"{_SKY_URL}/calendar/events", json=payload, timeout=10)
        if r.ok:
            return str(r.json().get("id") or "")
        return None
    except Exception:
        return None


def _sky_calendar_patch(event_id: str, status: str, notes: str = "") -> None:
    """Update a Sky calendar event status."""
    try:
        import requests as _req
        payload: Dict[str, Any] = {"status": status}
        if notes:
            payload["notes"] = notes
        _req.patch(f"{_SKY_URL}/calendar/events/{event_id}", json=payload, timeout=10)
    except Exception:
        pass


class QueueRunner:
    """Run a sequence of MixedCorpusBenchmark configs, each starting after the previous finishes."""

    def __init__(self, queue_file: Path) -> None:
        self.queue_file = queue_file
        raw = json.loads(queue_file.read_text(encoding="utf-8-sig"))
        self.manifest_path = Path(raw.get("manifest", "benchmarks/mixed_corpus_manifest.json"))
        self.runs: List[Dict[str, Any]] = list(raw.get("runs") or [])
        self.results: List[Dict[str, Any]] = []

    def _save_state(self) -> None:
        state = {"manifest": str(self.manifest_path), "runs": self.runs, "updated_at": _utc_iso()}
        self.queue_file.write_text(json.dumps(state, indent=2) + "\n", encoding="utf-8")

    def _ensure_calendar_event(self, run_cfg: Dict[str, Any]) -> None:
        if run_cfg.get("calendar_event_id"):
            return
        status = str(run_cfg.get("status") or "pending")
        cal_status = "paused" if status == "paused" else "pending"
        label = str(run_cfg.get("label") or run_cfg.get("id") or "OCR97 benchmark run")
        notes = f"Queue run: {run_cfg.get('id', '')} | variants: broad={run_cfg.get('broad_variants', '')}, focus={run_cfg.get('focus_variants', '')}"
        event_id = _sky_calendar_create(f"OCR97: {label}", notes, cal_status)
        if event_id:
            run_cfg["calendar_event_id"] = event_id
            self._save_state()

    def run_all(self) -> List[Dict[str, Any]]:
        # Ensure all pending/paused runs have calendar events before starting
        for run_cfg in self.runs:
            if str(run_cfg.get("status") or "") not in {"complete", "failed"}:
                self._ensure_calendar_event(run_cfg)

        for run_cfg in self.runs:
            cur_status = str(run_cfg.get("status") or "")
            if cur_status in {"complete", "failed", "paused"}:
                continue
            event_id = str(run_cfg.get("calendar_event_id") or "")
            run_cfg["status"] = "running"
            run_cfg["started_at"] = _utc_iso()
            self._save_state()
            if event_id:
                _sky_calendar_patch(event_id, "in_work", "OCR97 benchmark run started.")
            manifest_path = Path(run_cfg.get("manifest") or str(self.manifest_path))
            try:
                runner = MixedCorpusBenchmark(
                    manifest_path=manifest_path,
                    output_dir=Path(run_cfg["run_dir"]),
                    broad_limit=int(run_cfg.get("broad_limit") or 0),
                    focus_limit=int(run_cfg.get("focus_limit") or 20),
                    focus_ids=_split_csv(run_cfg.get("focus_ids")) if str(run_cfg.get("focus_ids") or "").strip() else None,
                    broad_variants=_split_csv(run_cfg.get("broad_variants") or "clean,mild_degraded,low_contrast,rotated,noisy_scan"),
                    focus_variants=_split_csv(run_cfg.get("focus_variants") or "rotated,noisy_scan"),
                    include_heavy=bool(run_cfg.get("include_heavy", True)),
                    include_auto_route=bool(run_cfg.get("include_auto_route", False)),
                    auto_route_variants=_split_csv(run_cfg.get("auto_route_variants") or "clean,noisy_scan"),
                    skip_native_pdf=bool(run_cfg.get("skip_native_pdf", False)),
                )
                result = runner.run()
                run_cfg["status"] = "complete"
                run_cfg["summary_path"] = str(runner.summary_path)
                summary = result.get("summary") or {}
                notes = (
                    f"Completed. best_avg={summary.get('best_score_avg', '?')}, "
                    f"worst_avg={summary.get('worst_score_avg', '?')}, "
                    f"passed={summary.get('passed', '?')}, failed={summary.get('failed', '?')}"
                )
                if event_id:
                    _sky_calendar_patch(event_id, "understood", notes)
            except Exception as exc:
                result = {"error": f"{type(exc).__name__}: {exc}"}
                run_cfg["status"] = "failed"
                run_cfg["error"] = str(exc)
                if event_id:
                    _sky_calendar_patch(event_id, "failed", f"Error: {exc}")
            run_cfg["finished_at"] = _utc_iso()
            self.results.append(result)
            self._save_state()
        return self.results


class MixedCorpusBenchmark:
    def __init__(
        self,
        *,
        manifest_path: Path,
        output_dir: Path,
        broad_limit: int = 0,
        focus_limit: int = 6,
        focus_ids: Optional[List[str]] = None,
        broad_variants: Optional[List[str]] = None,
        focus_variants: Optional[List[str]] = None,
        include_heavy: bool = True,
        plan_only: bool = False,
        skip_native_pdf: bool = False,
        include_auto_route: bool = False,
        auto_route_variants: Optional[List[str]] = None,
    ) -> None:
        self.manifest_path = manifest_path
        self.output_dir = output_dir
        self.fixture_dir = output_dir / "fixtures"
        self.artifact_dir = output_dir / "artifacts"
        self.summary_path = output_dir / "mixed_corpus_summary.json"
        self.progress_path = output_dir / "progress.json"
        self.report_path = output_dir / "MIXED_CORPUS_REPORT.md"
        self.broad_limit = int(broad_limit or 0)
        self.focus_limit = int(focus_limit or 0)
        self.focus_ids_are_explicit = focus_ids is not None
        self.focus_ids = list(focus_ids or DEFAULT_FOCUS_IDS)
        self.focus_is_strict = bool(self.focus_ids_are_explicit)
        self.broad_variants = list(broad_variants or ["clean", "mild_degraded", "low_contrast"])
        self.focus_variants = list(focus_variants or ["rotated", "noisy_scan"])
        self.include_heavy = bool(include_heavy)
        self.plan_only = bool(plan_only)
        self.skip_native_pdf = bool(skip_native_pdf)
        self.include_auto_route = bool(include_auto_route)
        self.auto_route_variants = list(auto_route_variants or ["clean", "noisy_scan"])
        self.started_at = _utc_iso()
        self.steps: List[Dict[str, Any]] = []
        self.status = "created"

    def _write_state(self, *, active_step: str = "") -> None:
        self.output_dir.mkdir(parents=True, exist_ok=True)
        payload = {
            "status": self.status,
            "started_at": self.started_at,
            "updated_at": _utc_iso(),
            "active_step": active_step,
            "manifest": str(self.manifest_path),
            "output_dir": str(self.output_dir),
            "summary": str(self.summary_path),
            "report": str(self.report_path),
            "steps": self.steps,
        }
        self.progress_path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
        self._write_report(payload)

    def _write_report(self, state: Mapping[str, Any]) -> None:
        lines = [
            "# OCR97 Mixed Corpus Benchmark Report",
            "",
            f"- Status: `{state.get('status')}`",
            f"- Started: `{state.get('started_at')}`",
            f"- Updated: `{state.get('updated_at')}`",
            f"- Manifest: `{state.get('manifest')}`",
            f"- Progress: `{self.progress_path}`",
            "",
            "| Step | Kind | Status | Cases | Avg score | Perfect | Below 75 | Avg latency ms | Max latency ms | Seconds |",
            "|---|---|---:|---:|---:|---:|---:|---:|---:|---:|",
        ]
        for step in self.steps:
            summary = dict(step.get("summary") or {})
            lines.append(
                "| "
                + " | ".join(
                    [
                        str(step.get("name") or ""),
                        str(step.get("benchmark_kind") or ""),
                        str(step.get("status") or ""),
                        str(summary.get("case_count", step.get("case_count", ""))),
                        str(summary.get("score_avg", "")),
                        str(summary.get("perfect_cases", "")),
                        str(summary.get("below_75_cases", "")),
                        str(summary.get("latency_avg_ms", "")),
                        str(summary.get("latency_max_ms", "")),
                        str(step.get("seconds", "")),
                    ]
                )
                + " |"
            )
        lines.append("")
        self.report_path.write_text("\n".join(lines) + "\n", encoding="utf-8")

    def _plan(self, manifest: Mapping[str, Any]) -> List[Dict[str, Any]]:
        broad = _select_cases(manifest, limit=self.broad_limit)
        focus = _select_cases(manifest, ids=self.focus_ids, limit=self.focus_limit, strict=self.focus_is_strict)
        if self.include_heavy and not list(focus.get("cases") or []):
            focus = _select_cases(manifest, limit=self.focus_limit)
        plan: List[Dict[str, Any]] = []
        if not self.skip_native_pdf:
            plan.append(
                {
                    "name": "native_pdf_text_broad",
                    "mode": "pdf",
                    "engine": "native_pdf_text",
                    "variant": "",
                    "manifest": broad,
                    "benchmark_kind": "forced_engine_diagnostic",
                    "requested_lane_strict": True,
                }
            )
        if self.include_auto_route:
            for variant in self.auto_route_variants:
                plan.append(
                    {
                        "name": f"auto_route_broad_{variant}",
                        "mode": "image",
                        "engine": "auto",
                        "variant": variant,
                        "manifest": broad,
                        "benchmark_kind": "end_to_end_auto_route",
                        "requested_lane_strict": False,
                    }
                )
        for variant in self.broad_variants:
            plan.append(
                {
                    "name": f"tesseract_broad_{variant}",
                    "mode": "image",
                    "engine": "tesseract",
                    "variant": variant,
                    "manifest": broad,
                    "benchmark_kind": "forced_engine_diagnostic",
                    "requested_lane_strict": True,
                }
            )
        if self.include_heavy:
            for variant in self.focus_variants:
                plan.append(
                    {
                        "name": f"preprocessed_focus_{variant}",
                        "mode": "image",
                        "engine": "local_image_preprocessed_best",
                        "variant": variant,
                        "manifest": focus,
                        "benchmark_kind": "forced_engine_diagnostic",
                        "requested_lane_strict": True,
                    }
                )
        return plan

    def run(self) -> Dict[str, Any]:
        os.environ.setdefault("OCR97_OCR_GATEWAY_PREWARM_ENABLED", "0")
        os.environ.setdefault("OCR97_OCR_GATEWAY_PREWARM_ON_STARTUP", "0")
        os.environ.setdefault("OCR97_OCR_SMOKE_REQUIRED", "0")
        os.environ.setdefault("OCR97_OCR_PREPROCESS_INCLUDE_TEXT", "1")
        os.environ.setdefault("OCR97_OCR_PREPROCESS_WORKERS", "4")
        os.environ.setdefault("OCR97_OCR_SLO_P95_IMAGE_PREPROCESSOR_MS", "180000")
        os.environ.setdefault("OCR97_OCR_SLO_P95_SEMANTIC_MS", "1800000")

        manifest = load_manifest(self.manifest_path)
        manifest_score = score_manifest(manifest)
        plan = self._plan(manifest)
        self.status = "planned" if self.plan_only else "running"
        self.steps = [
            {
                "name": str(step["name"]),
                "status": "planned",
                "mode": str(step["mode"]),
                "engine": str(step["engine"]),
                "variant": str(step["variant"]),
                "case_count": len(list((step.get("manifest") or {}).get("cases") or [])),
                "benchmark_kind": str(step.get("benchmark_kind") or ""),
                "requested_lane_strict": bool(step.get("requested_lane_strict", True)),
            }
            for step in plan
        ]
        self._write_state(active_step="")
        if self.plan_only:
            return self._finish(manifest_score=manifest_score)

        for index, step in enumerate(plan):
            name = str(step["name"])
            self.steps[index]["status"] = "running"
            self._write_state(active_step=name)
            started = time.perf_counter()
            try:
                if step["mode"] == "pdf":
                    result = run_gateway_truth_benchmark(
                        step["manifest"],
                        fixture_dir=self.fixture_dir / name,
                        output_dir=self.artifact_dir / name,
                    )
                else:
                    result = run_gateway_image_truth_benchmark(
                        step["manifest"],
                        fixture_dir=self.fixture_dir / name,
                        output_dir=self.artifact_dir / name,
                        variant=str(step["variant"]),
                        engine=str(step["engine"]),
                        requested_lane_strict=bool(step.get("requested_lane_strict", True)),
                        benchmark_kind=str(step.get("benchmark_kind") or "forced_engine_diagnostic"),
                    )
                seconds = round(time.perf_counter() - started, 2)
                step_output = self.output_dir / f"{name}.json"
                step_output.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
                self.steps[index].update(
                    {
                        "status": "pass",
                        "seconds": seconds,
                        "output": str(step_output),
                        "summary": _summarize_result(result),
                    }
                )
            except Exception as exc:
                seconds = round(time.perf_counter() - started, 2)
                self.steps[index].update({"status": "fail", "seconds": seconds, "error": f"{type(exc).__name__}: {exc}"})
                self.status = "failed"
                self._write_state(active_step=name)
                return self._finish(manifest_score=manifest_score)
            self._write_state(active_step=name)

        self.status = "completed"
        self._write_state(active_step="")
        return self._finish(manifest_score=manifest_score)

    def _finish(self, *, manifest_score: Mapping[str, Any]) -> Dict[str, Any]:
        failed = sum(1 for step in self.steps if step.get("status") == "fail")
        passed = sum(1 for step in self.steps if step.get("status") == "pass")
        completed = [dict(step.get("summary") or {}) for step in self.steps if step.get("summary")]
        summary = {
            "status": self.status,
            "started_at": self.started_at,
            "finished_at": _utc_iso(),
            "manifest": str(self.manifest_path),
            "manifest_case_count": int(manifest_score.get("case_count") or 0),
            "manifest_self_score": int(manifest_score.get("score_avg") or 0),
            "output_dir": str(self.output_dir),
            "progress": str(self.progress_path),
            "report": str(self.report_path),
            "passed": passed,
            "failed": failed,
            "steps": self.steps,
            "best_score_avg": max([int(row.get("score_avg") or 0) for row in completed], default=0),
            "worst_score_avg": min([int(row.get("score_avg") or 0) for row in completed], default=0),
        }
        self.summary_path.write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
        self._write_state(active_step="")
        llm_text = "_LLM grading skipped for plan-only run_" if self.plan_only else _llm_grade(summary)
        _write_llm_grade_report(self.output_dir, summary, llm_text)
        return summary


def run_scanned_fallback_benchmark(
    manifest_path: Path,
    output_dir: Path,
    *,
    broad_variants: Optional[List[str]] = None,
    focus_variants: Optional[List[str]] = None,
    plan_only: bool = False,
    auto_route_variants: Optional[List[str]] = None,
) -> Dict[str, Any]:
    """Run synthetic scan variants through auto routing and forced diagnostics.

    Auto-route steps exercise the production decision chain. Strict Tesseract and
    preprocessing steps remain compatibility diagnostics. Use the real-routing
    benchmark for claims about photographed or physically scanned documents.
    """
    runner = MixedCorpusBenchmark(
        manifest_path=manifest_path,
        output_dir=output_dir,
        broad_variants=broad_variants or ["clean", "mild_degraded", "low_contrast", "rotated", "noisy_scan"],
        focus_variants=focus_variants or ["rotated", "noisy_scan"],
        include_heavy=True,
        plan_only=plan_only,
        skip_native_pdf=True,
        include_auto_route=True,
        auto_route_variants=auto_route_variants or ["clean", "noisy_scan"],
    )
    return runner.run()


def main(argv: Optional[List[str]] = None) -> int:
    parser = argparse.ArgumentParser(description="Run OCR97 against a mixed generated document corpus with monitorable progress files.")
    parser.add_argument("--manifest", default="benchmarks/mixed_corpus_manifest.json", help="Mixed corpus manifest path.")
    parser.add_argument("--output-dir", default="artifacts/mixed_corpus_benchmark", help="Run output directory.")
    parser.add_argument("--broad-limit", type=int, default=0, help="Limit broad corpus cases; 0 means all cases.")
    parser.add_argument("--focus-limit", type=int, default=6, help="Limit heavy focus cases.")
    parser.add_argument("--focus-ids", default="", help="Comma-separated case ids for heavy focus lane.")
    parser.add_argument("--broad-variants", default="clean,mild_degraded,low_contrast", help="Comma-separated broad Tesseract image variants.")
    parser.add_argument("--focus-variants", default="rotated,noisy_scan", help="Comma-separated heavy local preprocessing variants.")
    parser.add_argument("--skip-heavy", action="store_true", help="Skip local_image_preprocessed_best focus steps.")
    parser.add_argument("--skip-native-pdf", action="store_true", help="Skip native_pdf_text step (use for image-only/scanned-doc manifests).")
    parser.add_argument("--include-auto-route", action="store_true", help="Run end-to-end auto routing without a strict requested lane.")
    parser.add_argument("--auto-route-variants", default="clean,noisy_scan", help="Comma-separated image variants for end-to-end auto routing.")
    parser.add_argument("--plan-only", action="store_true", help="Write the planned run and exit without invoking OCR.")
    parser.add_argument("--regrade", metavar="OUTPUT_DIR", help="Re-run LLM grading on an existing summary without re-running OCR.")
    parser.add_argument("--queue-file", metavar="QUEUE_JSON", help="JSON queue file; runs all pending entries sequentially.")
    args = parser.parse_args(argv)

    if args.regrade:
        output_dir = Path(args.regrade).expanduser()
        summary_path = output_dir / "mixed_corpus_summary.json"
        if not summary_path.exists():
            print(f"No summary found at {summary_path}")
            return 1
        summary = json.loads(summary_path.read_text(encoding="utf-8"))
        llm_text = _llm_grade(summary)
        report = _write_llm_grade_report(output_dir, summary, llm_text)
        print(json.dumps({"graded": True, "report": str(report)}, indent=2))
        return 0

    if args.queue_file:
        queue = QueueRunner(Path(args.queue_file).expanduser())
        results = queue.run_all()
        print(json.dumps({"runs_completed": len(results), "queue_file": args.queue_file}, indent=2))
        total_failed = sum(int(r.get("failed") or 0) for r in results if isinstance(r, dict))
        return 1 if total_failed else 0

    runner = MixedCorpusBenchmark(
        manifest_path=Path(args.manifest).expanduser(),
        output_dir=Path(args.output_dir).expanduser(),
        broad_limit=args.broad_limit,
        focus_limit=args.focus_limit,
        focus_ids=_split_csv(args.focus_ids) if str(args.focus_ids or "").strip() else None,
        broad_variants=_split_csv(args.broad_variants),
        focus_variants=_split_csv(args.focus_variants),
        include_heavy=not bool(args.skip_heavy),
        plan_only=bool(args.plan_only),
        skip_native_pdf=bool(getattr(args, "skip_native_pdf", False)),
        include_auto_route=bool(getattr(args, "include_auto_route", False)),
        auto_route_variants=_split_csv(args.auto_route_variants),
    )
    summary = runner.run()
    print(
        json.dumps(
            {
                "status": summary["status"],
                "manifest_case_count": summary["manifest_case_count"],
                "passed": summary["passed"],
                "failed": summary["failed"],
                "progress": summary["progress"],
                "report": summary["report"],
                "graded_report": str(runner.output_dir / "LLM_GRADED_REPORT.md"),
                "summary": str(runner.summary_path),
            },
            indent=2,
        )
    )
    return 1 if int(summary.get("failed") or 0) else 0


if __name__ == "__main__":
    raise SystemExit(main())
