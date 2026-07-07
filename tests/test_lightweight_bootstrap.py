import json
import os
from pathlib import Path
import subprocess
import sys
import time


def test_bootstrap_check_only_is_lightweight(tmp_path):
    root = Path(__file__).resolve().parents[1]
    output = tmp_path / "ocr97_install_metadata.json"

    started = time.perf_counter()
    proc = subprocess.run(
        [
            sys.executable,
            "-m",
            "ocr97.bootstrap",
            "--check-only",
            "--output",
            str(output),
        ],
        cwd=root,
        env={**os.environ, "PYTHONPATH": str(root / "src")},
        capture_output=True,
        text=True,
        timeout=10,
    )
    elapsed = time.perf_counter() - started

    assert proc.returncode == 0, proc.stderr
    assert elapsed < 10.0
    summary = json.loads(proc.stdout)
    assert summary["output"] == str(output)

    metadata = json.loads(output.read_text(encoding="utf-8"))
    assert metadata["install_action"]["mode"] == "check_only"
    for row in metadata["engines"].values():
        assert row["status"]["diagnostic_mode"] == "lightweight_no_model_import"

