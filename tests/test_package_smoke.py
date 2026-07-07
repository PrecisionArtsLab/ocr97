import json
import os
from pathlib import Path
import subprocess
import sys
import time


def test_package_files_present():
    root = Path(__file__).resolve().parents[1]
    assert (root / "pyproject.toml").exists()


def test_light_imports_smoke():
    import ocr97  # noqa: F401
    import ocr97.cli  # noqa: F401
    import ocr97.diagnostics  # noqa: F401


def test_package_root_does_not_import_heavy_runtime():
    sys.modules.pop("ocr97", None)
    sys.modules.pop("ocr97.dual_tool", None)

    import ocr97  # noqa: F401

    assert "ocr97.dual_tool" not in sys.modules


def test_doctor_payload_is_lightweight():
    from ocr97.diagnostics import doctor_payload

    started = time.perf_counter()
    payload = doctor_payload()
    elapsed = time.perf_counter() - started

    assert elapsed < 2.0
    assert payload["ok"] is True
    assert payload["diagnostic_mode"] == "lightweight_no_model_import"
    assert "gb10_paddleocr_vl" in payload["engines"]
    assert payload["hardware_scaling"]["does_not_require_specific_hardware"] is True
    assert "native_pdf_text" in payload["hardware_scaling"]["baseline_lanes"]


def test_public_profile_keeps_gb10_optional_by_default(monkeypatch):
    monkeypatch.delenv("OCR97_PROFILE", raising=False)
    monkeypatch.delenv("OCR97_GB10_OCR_ENABLED", raising=False)

    from ocr97.profiles import active_profile, gb10_default_enabled

    assert active_profile() == "github-release"
    assert gb10_default_enabled() is False


def test_hardware_profile_scales_from_cpu_to_remote_model(monkeypatch):
    monkeypatch.setenv("OCR97_HARDWARE_PROFILE", "cpu-only")

    from ocr97.diagnostics import doctor_payload

    payload = doctor_payload()
    scaling = payload["hardware_scaling"]
    assert scaling["requested_profile"] == "cpu"
    assert scaling["effective_profile"] == "cpu"
    assert scaling["does_not_require_specific_hardware"] is True

    monkeypatch.setenv("OCR97_HARDWARE_PROFILE", "gb10")
    payload = doctor_payload()
    scaling = payload["hardware_scaling"]
    assert scaling["requested_profile"] == "remote-model"
    assert scaling["effective_profile"] == "remote-model"


def test_cli_doctor_is_lightweight():
    root = Path(__file__).resolve().parents[1]
    started = time.perf_counter()
    proc = subprocess.run(
        [sys.executable, "-m", "ocr97.cli", "doctor"],
        cwd=root,
        env={**os.environ, "PYTHONPATH": str(root / "src")},
        capture_output=True,
        text=True,
        timeout=8,
    )
    elapsed = time.perf_counter() - started

    assert proc.returncode == 0, proc.stderr
    assert elapsed < 8.0
    payload = json.loads(proc.stdout)
    assert payload["diagnostic_mode"] == "lightweight_no_model_import"

