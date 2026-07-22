from __future__ import annotations

import argparse
import importlib.util
import json
import os
import subprocess
import sys
import tempfile
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping, Optional

import requests

from .sroie_runner import SROIE_DATASET, SROIE_SOURCE_PAGE, _fetch_rows


CAMPAIGN_ID = "ocr97_week_comparative_benchmark"
DEFAULT_PHASES = [
    {"id": "preflight", "label": "Corpus and engine preflight", "kind": "preflight"},
    {"id": "sroie_tesseract", "label": "Tesseract on 50 SROIE photos", "kind": "sroie", "engine": "tesseract"},
    {"id": "sroie_ocr97", "label": "OCR97 on 50 SROIE photos", "kind": "sroie", "engine": "local_image_preprocessed_best"},
    {"id": "sroie_paddleocr", "label": "PaddleOCR on 50 SROIE photos", "kind": "sroie", "engine": "paddleocr"},
    {"id": "sroie_surya", "label": "Surya on 50 SROIE photos", "kind": "sroie", "engine": "surya"},
    {"id": "sroie_doctr", "label": "docTR on 50 SROIE photos", "kind": "sroie", "engine": "doctr"},
    {"id": "layout_multilingual_final", "label": "Layout, table, multilingual sanity, and final grade", "kind": "final"},
]


def utc_now() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat().replace("+00:00", "Z")


def atomic_json(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, raw = tempfile.mkstemp(prefix=f".{path.name}.", suffix=".tmp", dir=str(path.parent))
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as handle:
            json.dump(payload, handle, indent=2, ensure_ascii=True)
            handle.write("\n")
        os.replace(raw, path)
    finally:
        if os.path.exists(raw):
            os.unlink(raw)


def safe_json(path: Path) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8-sig"))
    except Exception:
        return {}
    return value if isinstance(value, dict) else {}


def comparative_grade(engine_rows: list[Mapping[str, Any]], table_rows: list[Mapping[str, Any]]) -> dict[str, Any]:
    ocr97 = next((dict(row) for row in engine_rows if row.get("engine") == "local_image_preprocessed_best"), {})
    baselines = [dict(row) for row in engine_rows if row.get("engine") != "local_image_preprocessed_best" and int(row.get("case_count") or 0) > 0]
    metrics = dict(ocr97.get("text_metrics") or {})
    completion = max(0.0, min(1.0, (int(ocr97.get("case_count") or 0) - int(ocr97.get("failure_count") or 0)) / 50.0))
    cer = float(metrics.get("cer") or 1.0)
    wer = float(metrics.get("wer") or 1.0)
    field = float(ocr97.get("score_avg") or 0.0) / 100.0
    accuracy_points = round(40.0 * max(0.0, min(1.0, (field + (1.0 - min(cer, 1.0)) + (1.0 - min(wer, 1.0))) / 3.0)), 2)
    layout_value = float(metrics.get("layout_table_proxy_score") or 0.0)
    table_ocr97 = next((dict(row) for row in table_rows if row.get("engine") == "ocr97"), {})
    table_score = float(table_ocr97.get("score_avg") or 0.0)
    layout_points = round(20.0 * max(0.0, min(1.0, ((layout_value + table_score) / 2.0) / 100.0)), 2)
    ocr_latency = float(ocr97.get("latency_avg_ms") or 0.0)
    baseline_latencies = [float(row.get("latency_avg_ms") or 0.0) for row in baselines if float(row.get("latency_avg_ms") or 0.0) > 0]
    best_latency = min(baseline_latencies) if baseline_latencies else 0.0
    latency_ratio = (best_latency / ocr_latency) if best_latency and ocr_latency else 0.0
    # Full latency credit requires OCR97 to be at least 25% faster than the
    # fastest fully scored baseline: baseline / OCR97 >= 1 / 0.75.
    latency_points = round(15.0 * max(0.0, min(1.0, latency_ratio * 0.75)), 2)
    reliability_points = round(15.0 * completion, 2)
    evidence_complete = len([row for row in engine_rows if int(row.get("case_count") or 0) >= 45])
    evidence_points = round(10.0 * min(1.0, evidence_complete / 5.0), 2)
    total = round(accuracy_points + layout_points + latency_points + reliability_points + evidence_points, 2)
    return {
        "score": int(round(total)),
        "raw_score": total,
        "rubric": {
            "accuracy": {"points": accuracy_points, "max": 40},
            "layout_and_tables": {"points": layout_points, "max": 20},
            "latency_vs_fastest_baseline": {"points": latency_points, "max": 15},
            "reliability": {"points": reliability_points, "max": 15},
            "evidence_completeness": {"points": evidence_points, "max": 10},
        },
        "grade_scope": "English real-photo OCR; generated table/layout and multilingual sanity are supporting evidence",
        "latency_ratio_vs_fastest_baseline": round(latency_ratio, 4),
    }


class WeekCampaign:
    def __init__(self, root: Path, *, max_phase_attempts: int = 2, notify: bool = True, run_id: str = "") -> None:
        self.root = root
        self.state_path = root / "campaign_state.json"
        self.report_path = root / "FINAL_REPORT.md"
        self.report_json_path = root / "final_report.json"
        self.max_phase_attempts = max(1, int(max_phase_attempts))
        self.notify = notify
        self.state = safe_json(self.state_path)
        if not self.state:
            self.state = {
                "schema_version": 1,
                "campaign_id": CAMPAIGN_ID,
                "status": "pending",
                "started_at": utc_now(),
                "updated_at": utc_now(),
                "dataset": SROIE_DATASET,
                "source_page": SROIE_SOURCE_PAGE,
                "target_real_documents": 50,
                "phases": [{**phase, "status": "pending", "attempts": 0} for phase in DEFAULT_PHASES],
                "events": [],
                "notification": {"sent": False},
            }
            self.save()
        if run_id:
            self.state["active_run_id"] = run_id
            self.save()

    def save(self) -> None:
        self.state["updated_at"] = utc_now()
        atomic_json(self.state_path, self.state)

    def event(self, kind: str, message: str, **extra: Any) -> None:
        self.state.setdefault("events", []).append({"ts": utc_now(), "kind": kind, "message": message, **extra})
        self.state["events"] = self.state["events"][-300:]
        self.save()

    def next_phase(self) -> Optional[dict[str, Any]]:
        phases = self.state.get("phases") if isinstance(self.state.get("phases"), list) else []
        return next((phase for phase in phases if phase.get("status") not in {"complete", "blocked"}), None)

    def _run_command(self, command: list[str], log_path: Path, timeout: int = 19800) -> tuple[int, str]:
        log_path.parent.mkdir(parents=True, exist_ok=True)
        env = dict(os.environ)
        src = str(Path(__file__).resolve().parents[1])
        env["PYTHONPATH"] = src + (os.pathsep + env["PYTHONPATH"] if env.get("PYTHONPATH") else "")
        try:
            result = subprocess.run(command, cwd=str(Path(__file__).resolve().parents[2]), env=env, text=True, capture_output=True, timeout=timeout)
            output = (result.stdout or "") + ("\n" + result.stderr if result.stderr else "")
            log_path.write_text(output[-200000:], encoding="utf-8", errors="replace")
            return result.returncode, output
        except subprocess.TimeoutExpired as exc:
            output = f"timed_out_after_{timeout}s\n{exc.stdout or ''}\n{exc.stderr or ''}"
            log_path.write_text(output[-200000:], encoding="utf-8", errors="replace")
            return 124, output

    def _preflight(self) -> dict[str, Any]:
        packages = {name: bool(importlib.util.find_spec(module)) for name, module in {
            "tesseract_python": "pytesseract", "paddleocr": "paddleocr", "surya": "surya", "doctr": "doctr"
        }.items()}
        rows = _fetch_rows(split="test", offset=0, length=50, dataset=SROIE_DATASET)
        manifest = []
        for item in rows:
            row = dict(item.get("row") or {})
            manifest.append({
                "row_idx": item.get("row_idx"), "key": row.get("key"),
                "source": str((row.get("image") or {}).get("src") or "").split("?")[0],
                "word_line_count": len(list(row.get("words") or [])), "bbox_count": len(list(row.get("bboxes") or [])),
                "entity_fields": sorted(dict(row.get("entities") or {}).keys()),
            })
        atomic_json(self.root / "corpus_manifest.json", {"dataset": SROIE_DATASET, "count": len(manifest), "rows": manifest})
        return {"ok": len(manifest) >= 50 and packages["tesseract_python"], "corpus_count": len(manifest), "packages": packages}

    def _run_sroie(self, phase: Mapping[str, Any]) -> dict[str, Any]:
        engine = str(phase.get("engine") or "")
        output = self.root / "runs" / str(phase.get("id"))
        command = [sys.executable, "-m", "ocr97.overnight_benchmark", "--output-dir", str(output), "--length", "50", "--engines", engine, "--retries", "1"]
        code, tail = self._run_command(command, self.root / "logs" / f"{phase.get('id')}.log")
        report = safe_json(output / "report.json")
        row = next((dict(item) for item in report.get("engine_results") or [] if item.get("engine") == engine), {})
        return {"ok": code == 0 and int(row.get("case_count") or 0) >= 50, "exit_code": code, "report_path": str(output / "report.json"), "engine_result": row, "error_tail": tail[-1000:] if code else ""}

    def _table_manifest(self) -> Path:
        source = Path(__file__).resolve().parents[2] / "benchmarks" / "release_97_gate_manifest.json"
        payload = safe_json(source)
        cases = [dict(case) for case in payload.get("cases") or [] if case.get("release_variant") == "table_first"][:20]
        cases.extend([
            {"id": "multilingual_es", "label": "Spanish invoice sanity", "category": "multilingual", "required_tokens": ["Factura", "Total", "Fecha"], "expected_fields": [{"name": "total", "expected": "184.75", "type": "money"}], "sample_text": "Factura: ES-2048\nFecha: 2026-07-11\nDescripción: reparación técnica\nTotal: 184,75 EUR"},
            {"id": "multilingual_fr", "label": "French receipt sanity", "category": "multilingual", "required_tokens": ["Reçu", "Montant", "Date"], "expected_fields": [{"name": "total", "expected": "42.90", "type": "money"}], "sample_text": "Reçu: FR-778\nDate: 11/07/2026\nCafé et pâtisserie\nMontant: 42,90 EUR"},
            {"id": "multilingual_de", "label": "German order sanity", "category": "multilingual", "required_tokens": ["Bestellung", "Gesamt", "Datum"], "expected_fields": [{"name": "total", "expected": "73.20", "type": "money"}], "sample_text": "Bestellung: DE-991\nDatum: 11.07.2026\nGeräteprüfung\nGesamt: 73,20 EUR"},
        ])
        target = self.root / "layout_multilingual_manifest.json"
        atomic_json(target, {"name": "ocr97_week_layout_multilingual", "description": "Generated supporting evidence; not part of the 50-photo accuracy score.", "cases": cases})
        return target

    def _final_phase(self) -> dict[str, Any]:
        manifest = self._table_manifest()
        output = self.root / "runs" / "layout_multilingual_final"
        command = [sys.executable, "-m", "ocr97.baseline_compare", "--manifest", str(manifest), "--fixture-dir", str(output / "fixtures"), "--artifact-dir", str(output), "--variant", "mild_degraded", "--engines", "ocr97,tesseract,paddleocr,surya,doctr", "--ocr97-engine", "local_image_preprocessed_best"]
        code, tail = self._run_command(command, self.root / "logs" / "layout_multilingual_final.log")
        table = safe_json(output / "baseline_comparison.json")
        final = self.write_final(table)
        return {"ok": bool(final), "exit_code": code, "report_path": str(self.report_json_path), "comparison_path": str(output / "baseline_comparison.json"), "error_tail": tail[-1000:] if code else ""}

    def collect_engine_rows(self) -> list[dict[str, Any]]:
        rows: list[dict[str, Any]] = []
        for phase in self.state.get("phases") or []:
            result = phase.get("result") if isinstance(phase.get("result"), dict) else {}
            row = result.get("engine_result") if isinstance(result.get("engine_result"), dict) else {}
            if row:
                rows.append(dict(row))
        return rows

    def write_final(self, comparison: Mapping[str, Any]) -> dict[str, Any]:
        engine_rows = self.collect_engine_rows()
        table_rows = [dict(row) for row in comparison.get("engines") or []]
        grade = comparative_grade(engine_rows, table_rows)
        complete_engines = len([row for row in engine_rows if int(row.get("case_count") or 0) >= 45])
        status = "complete" if complete_engines == 5 and comparison else "partial"
        payload = {
            "schema_version": 1, "suite_id": CAMPAIGN_ID, "status": status, "completed_at": utc_now(),
            "real_document_count": 50, "real_dataset": SROIE_DATASET, "engine_results": engine_rows,
            "supporting_comparison": table_rows, "comparative_grade": grade,
            "blocked_phases": [phase.get("id") for phase in self.state.get("phases") or [] if phase.get("status") == "blocked"],
            "state_path": str(self.state_path), "report_path": str(self.report_path),
        }
        atomic_json(self.report_json_path, payload)
        lines = [
            "# OCR97 Seven-Night Comparative Benchmark", "", f"- Status: `{status}`", f"- Comparative grade: `{grade['score']}/100`",
            f"- Real corpus: `50` SROIE receipt photographs", "- Scope: English accuracy is primary; generated table/layout and multilingual cases are supporting evidence.", "",
            "## Real-Photo Results", "", "| Engine | Cases | Failures | Field score | CER | WER | Layout proxy | Avg ms | p95 ms |", "|---|---:|---:|---:|---:|---:|---:|---:|---:|",
        ]
        for row in engine_rows:
            metrics = dict(row.get("text_metrics") or {})
            lines.append(f"| `{row.get('engine')}` | {row.get('case_count')} | {row.get('failure_count')} | {row.get('score_avg')} | {metrics.get('cer')} | {metrics.get('wer')} | {metrics.get('layout_table_proxy_score')} | {row.get('latency_avg_ms')} | {row.get('latency_p95_ms')} |")
        lines.extend(["", "## Grade Rubric", ""])
        for name, bucket in grade["rubric"].items():
            lines.append(f"- `{name}`: {bucket['points']}/{bucket['max']}")
        lines.extend(["", "## Guardrails", "", "- Results resume from durable per-engine state; existing artifacts are not discarded.", "- An unavailable engine is reported as blocked after bounded attempts; no score is fabricated.", "- Generated evidence is labeled separately and does not replace the 50 real-photo corpus.", ""])
        self.report_path.write_text("\n".join(lines), encoding="utf-8")
        self.state["status"] = status
        self.state["completed_at"] = payload["completed_at"]
        self.save()
        self.send_notification(payload)
        return payload

    def send_notification(self, payload: Mapping[str, Any]) -> None:
        notice = self.state.get("notification") if isinstance(self.state.get("notification"), dict) else {}
        if notice.get("sent") or not self.notify:
            return
        self._load_sky_env()
        topic = os.getenv("ARGUS_NTFY_TOPIC") or os.getenv("SKY_NTFY_TOPIC")
        base = (os.getenv("ARGUS_NTFY_URL") or os.getenv("SKY_NTFY_URL") or "https://ntfy.sh").rstrip("/")
        if not topic:
            self.state["notification"] = {"sent": False, "error": "ntfy_topic_missing", "attempted_at": utc_now()}
            self.save()
            return
        headers = {"Title": "OCR97 week benchmark finished", "Tags": "white_check_mark" if payload.get("status") == "complete" else "warning"}
        token = os.getenv("ARGUS_NTFY_TOKEN") or os.getenv("SKY_NTFY_TOKEN")
        if token:
            headers["Authorization"] = f"Bearer {token}"
        message = f"OCR97 benchmark {payload.get('status')}. Comparative grade: {(payload.get('comparative_grade') or {}).get('score')}/100. Report: {self.report_path}"
        try:
            response = requests.post(f"{base}/{topic}", data=message.encode("utf-8"), headers=headers, timeout=15)
            response.raise_for_status()
            self.state["notification"] = {"sent": True, "sent_at": utc_now(), "status_code": response.status_code}
        except Exception as exc:
            self.state["notification"] = {"sent": False, "attempted_at": utc_now(), "error": f"{type(exc).__name__}: {exc}"}
        self.save()

    def _load_sky_env(self) -> None:
        path = Path(__file__).resolve().parents[3] / "Sky" / ".env"
        if not path.exists():
            return
        for raw in path.read_text(encoding="utf-8-sig", errors="replace").splitlines():
            line = raw.strip()
            if not line or line.startswith("#") or "=" not in line:
                continue
            key, value = line.split("=", 1)
            os.environ.setdefault(key.strip(), value.strip().strip('"').strip("'"))

    def run_next(self, *, dry_run: bool = False) -> dict[str, Any]:
        phase = self.next_phase()
        if phase is None:
            return {"ok": True, "status": self.state.get("status"), "message": "campaign_terminal", "state_path": str(self.state_path)}
        if dry_run:
            return {"ok": True, "status": "dry_run", "next_phase": phase, "state_path": str(self.state_path)}
        phase["status"] = "running"
        phase["attempts"] = int(phase.get("attempts") or 0) + 1
        phase["started_at"] = utc_now()
        self.state["status"] = "running"
        self.event("phase_start", str(phase.get("label")), phase_id=phase.get("id"), attempt=phase["attempts"])
        try:
            if phase.get("kind") == "preflight":
                result = self._preflight()
            elif phase.get("kind") == "sroie":
                result = self._run_sroie(phase)
            else:
                result = self._final_phase()
        except Exception as exc:
            result = {"ok": False, "error": f"{type(exc).__name__}: {exc}"}
        phase["result"] = result
        phase["finished_at"] = utc_now()
        if result.get("ok"):
            phase["status"] = "complete"
        elif int(phase.get("attempts") or 0) >= self.max_phase_attempts:
            phase["status"] = "blocked"
        else:
            phase["status"] = "pending"
        self.event("phase_finish", f"{phase.get('id')} -> {phase.get('status')}", phase_id=phase.get("id"), result=result)
        return {"ok": bool(result.get("ok")), "phase_id": phase.get("id"), "phase_status": phase.get("status"), "result": result, "state_path": str(self.state_path)}


def main(argv: Optional[list[str]] = None) -> int:
    parser = argparse.ArgumentParser(description="Run one resumable phase of OCR97's seven-night comparative benchmark.")
    parser.add_argument("--campaign-root", default=str(Path("artifacts") / "week_comparative_campaign"))
    parser.add_argument("--max-phase-attempts", type=int, default=2)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--no-notify", action="store_true")
    parser.add_argument("--run-id", default="")
    args = parser.parse_args(argv)
    result = WeekCampaign(Path(args.campaign_root).expanduser().resolve(), max_phase_attempts=args.max_phase_attempts, notify=not args.no_notify, run_id=args.run_id).run_next(dry_run=args.dry_run)
    print(json.dumps(result, indent=2, ensure_ascii=True))
    return 0 if result.get("ok") or result.get("phase_status") in {"pending", "blocked"} else 1


if __name__ == "__main__":
    raise SystemExit(main())
