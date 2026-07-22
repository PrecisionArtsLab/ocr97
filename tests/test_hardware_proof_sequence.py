from __future__ import annotations

import json
import re
import sys
from datetime import datetime
from pathlib import Path
from types import SimpleNamespace
from zoneinfo import ZoneInfo

import pytest

ENGINEERING_ROOT = Path(__file__).resolve().parents[2]
if str(ENGINEERING_ROOT) not in sys.path:
    sys.path.insert(0, str(ENGINEERING_ROOT))

try:
    from Sky.core import calendar_store
    from Sky.services import test_run_reports as trr
except ModuleNotFoundError:
    calendar_store = None
    trr = None

from ocr97.hardware_escalation import decide_hardware_escalation
from ocr97 import hardware_proof_scheduler as scheduler
from ocr97 import hardware_proof_sequence as runner


requires_sky = pytest.mark.skipif(
    calendar_store is None or trr is None,
    reason="Sky integration package is not available in the standalone OCR97 public repo environment.",
)


def _proc(returncode: int = 0, stdout: str = "ok", stderr: str = ""):
    return SimpleNamespace(returncode=returncode, stdout=stdout, stderr=stderr)


def _isolated_store(monkeypatch, tmp_path):
    monkeypatch.setattr(calendar_store, "STORE_PATH", tmp_path / "calendar_events.json")
    monkeypatch.setattr(calendar_store, "LOG_DIR", tmp_path)
    calendar_store._STORE_CACHE = None
    calendar_store._STORE_MTIME_NS = None


def _write_parent(path: str, payload: dict) -> None:
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(json.dumps(payload), encoding="utf-8")


def _extract_arg(command: str, name: str) -> str:
    match = re.search(rf"{re.escape(name)}\s+\"([^\"]+)\"", command)
    return match.group(1) if match else ""


def _fake_runner(returncode: int = 0):
    def fake(command, **kwargs):
        if isinstance(command, list):
            return _proc(0, "NVIDIA GeForce RTX 3090, 24576, 555.55, 12, 1024")
        text = str(command)
        if "truth_runner" in text:
            output = _extract_arg(text, "--output")
            _write_parent(
                output,
                {
                    "mode": "gateway_image_local_image_preprocessed_best_rotated",
                    "case_count": 10,
                    "score_avg": 92,
                    "results": [{"id": "case_1", "score": {"score": 92}, "latency_ms": 1200}],
                },
            )
        if "capability_audit" in text:
            json_output = _extract_arg(text, "--json-output")
            md_output = _extract_arg(text, "--output")
            _write_parent(json_output, {"result_file_count": 2, "weak_case_count": 0, "reason_counts": {}, "recommendations": ["Keep measuring."]})
            Path(md_output).parent.mkdir(parents=True, exist_ok=True)
            Path(md_output).write_text("# audit\n", encoding="utf-8")
        return _proc(returncode)

    return fake


def test_hardware_escalation_routes_low_confidence_to_3090_when_available():
    result = decide_hardware_escalation({"score": 72, "field_misses": ["total"], "variant": "noisy_scan"}, {"gpu_ready": True})

    assert result["should_escalate"] is True
    assert result["action"] == "route_to_3090_hard_document_lane"
    assert "low_confidence" in result["reasons"]


def test_hardware_escalation_reports_gpu_gap_when_unavailable():
    result = decide_hardware_escalation({"score": 72, "table_row_gap": True}, {"gpu_ready": False})

    assert result["should_escalate"] is True
    assert result["category"] == "gpu_lane_unavailable"
    assert result["recoverable"] is False


def test_sequence_dry_run_returns_exact_command_order(capsys):
    assert runner.main(["--dry-run", "--run-id", "ocr97_hw_dry"]) == 0
    payload = json.loads(capsys.readouterr().out)

    assert payload["ok"] is True
    assert payload["commands"][0].endswith("-m ocr97.cli doctor")
    assert "ocr97.bootstrap --check-only" in payload["commands"][1]
    assert "tests/test_3090_longform_plan.py" in payload["commands"][2]
    assert "--variant rotated" in payload["commands"][3]
    assert "--variant noisy_scan" in payload["commands"][4]
    assert "ocr97.capability_audit" in payload["commands"][5]


def test_sequence_passing_commands_write_json_markdown_readme(monkeypatch, tmp_path):
    readme = tmp_path / "README.md"
    readme.write_text("# OCR97\n", encoding="utf-8")
    monkeypatch.setattr(runner, "README_PATH", readme)
    monkeypatch.setattr(
        runner,
        "_engine_readiness_snapshot",
        lambda: {
            "ok": True,
            "status_code": 200,
            "route_mode_default": "quality_first",
            "engine_count": 3,
            "ready_engine_count": 2,
            "engines": [{"name": "native_pdf_text", "ready": True}, {"name": "gb10_qwen_ocr", "ready": True}],
            "engine_names": ["native_pdf_text", "gb10_qwen_ocr"],
            "ready_engine_names": ["native_pdf_text", "gb10_qwen_ocr"],
        },
    )

    result = runner.run_sequence(
        run_id="ocr97_hw_pass",
        root=tmp_path / "runs",
        no_post=True,
        command_runner=_fake_runner(0),
    )

    assert result["status"] == "passed"
    assert result["payload"]["score"] >= 80
    assert Path(result["paths"]["json"]).exists()
    assert Path(result["paths"]["markdown"]).exists()
    readme_text = readme.read_text(encoding="utf-8")
    assert "OCR97 Hardware Proof Reports" in readme_text
    assert "ocr97_hw_pass" in readme_text
    assert result["payload"]["hardware_escalation"]["action"] == "route_to_3090_hard_document_lane"
    assert result["payload"]["low_confidence_flag"] is False
    assert "native_pdf_text" in result["payload"]["active_default_engine_chain"]
    assert result["payload"]["engine_readiness_snapshot"]["ready_engine_count"] == 2
    assert result["payload"]["last_successful_proof_timestamp"]


def test_sequence_failure_still_writes_diagnostic(monkeypatch, tmp_path):
    readme = tmp_path / "README.md"
    readme.write_text("# OCR97\n", encoding="utf-8")
    monkeypatch.setattr(runner, "README_PATH", readme)
    monkeypatch.setattr(runner, "_engine_readiness_snapshot", lambda: {"ok": False, "engines": [], "engine_names": [], "ready_engine_names": []})

    result = runner.run_sequence(
        run_id="ocr97_hw_fail",
        root=tmp_path / "runs",
        no_post=True,
        command_runner=_fake_runner(1),
    )

    assert result["status"] == "failed"
    assert result["payload"]["diagnostic"]["cause"] == "ocr97_hardware_proof_sequence_failed"
    assert result["payload"]["capabilities_not_confirmed"] == [runner.CAPABILITY]
    assert result["payload"]["low_confidence_flag"] is True
    assert Path(result["paths"]["markdown"]).exists()


@requires_sky
def test_report_payload_validates_and_records_to_ledger(tmp_path):
    run_id = "ocr97_hw_ledger"
    run_dir = tmp_path / "runs" / run_id
    run_dir.mkdir(parents=True)
    (run_dir / "rotated_result.json").write_text(json.dumps({"score_avg": 90, "case_count": 10, "mode": "rotated", "results": []}), encoding="utf-8")
    (run_dir / "noisy_scan_result.json").write_text(json.dumps({"score_avg": 86, "case_count": 10, "mode": "noisy", "results": []}), encoding="utf-8")
    (run_dir / "OCR97_HARDWARE_PROOF_AUDIT.json").write_text(json.dumps({"weak_case_count": 1, "reason_counts": {"field_miss": 1}}), encoding="utf-8")
    results = [{"command": command, "returncode": 0, "stdout": "ok", "stderr": ""} for command in runner.rendered_commands(run_id=run_id, root=tmp_path / "runs")]
    payload = runner.build_report_payload(results, run_id=run_id, run_dir=run_dir, hardware={"gpu_ready": True, "gpu_name": "RTX 3090"})

    assert trr.validate_test_queue_report_payload(payload)["ok"] is True
    report = runner.record_sky_report(payload, root=tmp_path / "sky_reports")
    ledger = trr.load_project_capability_ledger(root=tmp_path / "sky_reports")

    assert report["ok"] is True
    assert ledger["projects"]["ocr97"]["latest_capability"] == runner.CAPABILITY
    assert ledger["projects"]["ocr97"]["latest_diagnostic"]["cause"] == "ocr97_hardware_proof_sequence_passed"


def test_scheduler_dry_run_payload_has_test_queue_fields():
    start = datetime(2026, 5, 18, 8, 30, tzinfo=ZoneInfo("America/Chicago"))
    result = scheduler.schedule(start=start, dry_run=True)
    event = result["calendar"]["event"]

    assert result["ok"] is True
    assert event["title"] == scheduler.TITLE
    assert event["test_queue"] is True
    assert event["project"] == "ocr97"
    assert event["test_type"] == runner.TEST_TYPE
    assert event["capability"] == runner.CAPABILITY
    assert event["verification_command"]
    assert event["report_path"].endswith("ocr97_hardware_proof_sequence_20260518_0830.md")
    assert "README section" in event["notes"]


@requires_sky
def test_scheduler_calendar_event_is_admitted(monkeypatch, tmp_path):
    _isolated_store(monkeypatch, tmp_path)
    start = datetime(2026, 5, 18, 8, 30, tzinfo=ZoneInfo("America/Chicago"))

    result = scheduler.create_calendar_card(start)

    assert result["ok"] is True
    assert result["event"]["report_required"] is True
    assert result["event"]["project"] == "ocr97"
    assert result["event"]["verification_command"]


@requires_sky
def test_scheduler_conflict_advances_to_next_open_slot(monkeypatch, tmp_path):
    _isolated_store(monkeypatch, tmp_path)
    calendar_store.create_event(
        {
            "title": "Busy",
            "start": "2026-05-18T13:30:00Z",
            "end": "2026-05-18T14:15:00Z",
            "status": "pending",
            "source": "sky",
            "trigger": "",
            "notes": "",
        }
    )
    start = datetime(2026, 5, 18, 8, 30, tzinfo=ZoneInfo("America/Chicago"))

    selected, result = scheduler.find_calendar_slot(start, max_checks=5)

    assert result["ok"] is True
    assert selected == datetime(2026, 5, 18, 8, 30, tzinfo=ZoneInfo("America/Chicago"))
    assert result["event"]["run_id"].endswith("20260518_0830")
    assert result["event"]["start"] == "2026-05-19T04:00:00Z"
    assert result["event"]["metadata"]["calendar_queue_guardrail"]["applied"] is True


@requires_sky
def test_scheduler_missing_required_fields_rejected(monkeypatch, tmp_path):
    _isolated_store(monkeypatch, tmp_path)
    payload = scheduler._calendar_payload(datetime(2026, 5, 18, 8, 30, tzinfo=ZoneInfo("America/Chicago")))
    payload["capability"] = ""
    payload["metadata"] = dict(payload["metadata"])
    payload["metadata"]["capability"] = ""
    payload["source"] = "sky"
    payload["title"] = "Broken hardware proof test"

    with pytest.raises(calendar_store.CalendarValidationError, match="test_queue_report_requirement_failed:capability"):
        calendar_store.create_event(payload)
