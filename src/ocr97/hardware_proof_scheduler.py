from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any, Dict, Optional, Tuple
from zoneinfo import ZoneInfo

from . import hardware_proof_sequence as runner


ENGINEERING_ROOT = Path(__file__).resolve().parents[3]
LOCAL_TZ = ZoneInfo("America/Chicago")
TITLE = "OCR97 3090 hardware proof validation"


def _ensure_sky_path() -> None:
    if str(ENGINEERING_ROOT) not in sys.path:
        sys.path.insert(0, str(ENGINEERING_ROOT))


def _next_morning_slot(now: Optional[datetime] = None) -> datetime:
    now = now or datetime.now(LOCAL_TZ)
    if now.tzinfo is None:
        now = now.replace(tzinfo=LOCAL_TZ)
    now = now.astimezone(LOCAL_TZ)
    candidate = now.replace(hour=8, minute=30, second=0, microsecond=0)
    if candidate <= now:
        candidate = candidate + timedelta(days=1)
    return candidate


def _run_id_for(start: datetime) -> str:
    return f"{runner.TEST_TYPE}_{start.astimezone(LOCAL_TZ).strftime('%Y%m%d_%H%M')}"


def _command_for(run_id: str) -> str:
    script = runner.OCR97_ROOT / "tools" / "run_hardware_proof_sequence.ps1"
    return f'powershell -NoProfile -ExecutionPolicy Bypass -File "{script}" -RunId {run_id}'


def _report_path(run_id: str) -> str:
    return str(runner.REPORT_ROOT / run_id / f"{run_id}.md")


def _calendar_payload(start: datetime) -> Dict[str, Any]:
    local_start = start.astimezone(LOCAL_TZ)
    end = local_start + timedelta(minutes=45)
    run_id = _run_id_for(local_start)
    command = _command_for(run_id)
    report_path = _report_path(run_id)
    notes = "\n".join(
        [
            "OCR97 hardware proof queue card.",
            "",
            f"Runner command: {command}",
            f"Expected report: {report_path}",
            "README section: OCR97 Hardware Proof Reports, dated by the run date.",
            "",
            "This sequence captures GPU evidence, runs OCR97 focused tests, exercises degraded image OCR, audits weak cases, and writes a Sky capability report.",
            f"[local_command:{runner.OCR97_ROOT / 'tools' / 'run_hardware_proof_sequence.ps1'}]",
        ]
    )
    payload = {
        "title": TITLE,
        "start": local_start.astimezone(ZoneInfo("UTC")).replace(microsecond=0).isoformat().replace("+00:00", "Z"),
        "end": end.astimezone(ZoneInfo("UTC")).replace(microsecond=0).isoformat().replace("+00:00", "Z"),
        "status": "pending",
        "source": "ocr97",
        "trigger": "test_queue",
        "notes": notes,
        "test_queue": True,
        "report_required": True,
        "project": "ocr97",
        "test_type": runner.TEST_TYPE,
        "run_id": run_id,
        "capability": runner.CAPABILITY,
        "verification_command": command,
        "report_path": report_path,
        "metadata": {
            "test_queue": True,
            "report_required": True,
            "project": "ocr97",
            "test_type": runner.TEST_TYPE,
            "run_id": run_id,
            "capability": runner.CAPABILITY,
            "verification_command": command,
            "report_path": report_path,
        },
    }
    return payload


def create_calendar_card(start: datetime) -> Dict[str, Any]:
    _ensure_sky_path()
    try:
        from Sky.core import calendar_store
    except Exception as exc:
        raise RuntimeError(
            "Sky calendar integration is unavailable. Use --dry-run for standalone OCR97 scheduling payloads."
        ) from exc

    event = calendar_store.create_event(_calendar_payload(start))
    return {"ok": True, "event": event}


def find_calendar_slot(start: datetime, *, max_checks: int = 96) -> Tuple[datetime, Dict[str, Any]]:
    _ensure_sky_path()
    try:
        from Sky.core import calendar_store
    except Exception as exc:
        raise RuntimeError(
            "Sky calendar integration is unavailable. Use --dry-run for standalone OCR97 scheduling payloads."
        ) from exc

    candidate = start.astimezone(LOCAL_TZ)
    for _ in range(max_checks):
        try:
            return candidate, create_calendar_card(candidate)
        except calendar_store.CalendarConflictError:
            candidate = candidate + timedelta(minutes=15)
    return candidate, {"ok": False, "error": "no_open_calendar_slot", "candidate": candidate.isoformat()}


def schedule(*, start: Optional[datetime] = None, dry_run: bool = False) -> Dict[str, Any]:
    selected = start or _next_morning_slot()
    if dry_run:
        payload = _calendar_payload(selected)
        return {"ok": True, "scheduled_for_local": selected.isoformat(), "calendar": {"event": payload}}
    try:
        selected, result = find_calendar_slot(selected)
    except RuntimeError as exc:
        return {"ok": False, "scheduled_for_local": selected.isoformat(), "error": str(exc)}
    return {"ok": bool(result.get("ok")), "scheduled_for_local": selected.isoformat(), "calendar": result}


def main(argv: Optional[list[str]] = None) -> int:
    parser = argparse.ArgumentParser(description="Schedule OCR97 hardware proof sequence on Sky's test queue calendar.")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--start", default="", help="Optional ISO datetime; defaults to next 8:30 AM America/Chicago.")
    args = parser.parse_args(argv)
    start = None
    if args.start:
        start = datetime.fromisoformat(args.start.replace("Z", "+00:00"))
        if start.tzinfo is None:
            start = start.replace(tzinfo=LOCAL_TZ)
    result = schedule(start=start, dry_run=args.dry_run)
    print(json.dumps(result, indent=2))
    return 0 if result.get("ok") else 1


if __name__ == "__main__":
    raise SystemExit(main())
