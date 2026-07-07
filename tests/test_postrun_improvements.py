import json
import os
import subprocess
import sys
from pathlib import Path


def test_overnight_drop_gateway_client_is_silent(tmp_path):
    from ocr97.overnight_benchmark import OCR97BenchAutopilot

    runner = OCR97BenchAutopilot(
        output_dir=tmp_path,
        engines=["gb10_qwen_ocr"],
        split="test",
        offset=0,
        length=1,
        retries=0,
        enable_model_debug=False,
        debug_model="",
        ollama_url="http://127.0.0.1:11434",
    )
    runner.prepare(reset=True)
    runner._client = object()
    runner._app = object()

    runner._drop_gateway_client()

    assert runner._client is None
    assert runner._app is None
    assert [row["kind"] for row in runner.state.get("events", [])] == []

    runner._client = object()
    runner._app = object()
    runner._reset_gateway_client("unit_failure")

    assert runner._client is None
    assert runner._app is None
    assert [row["kind"] for row in runner.state.get("events", [])] == ["gateway_reset"]


def test_overnight_gateway_client_sets_semantic_slo_override(tmp_path, monkeypatch):
    from ocr97.overnight_benchmark import OCR97BenchAutopilot

    monkeypatch.delenv("OCR97_OCR_SLO_P95_SEMANTIC_MS", raising=False)
    runner = OCR97BenchAutopilot(
        output_dir=tmp_path,
        engines=["gb10_qwen_ocr"],
        split="test",
        offset=0,
        length=1,
        retries=0,
        enable_model_debug=False,
        debug_model="",
        ollama_url="http://127.0.0.1:11434",
    )

    runner._get_gateway_client()

    assert os.environ["OCR97_OCR_SLO_P95_SEMANTIC_MS"] == "1800000"


def test_gateway_semantic_slo_cap_is_visible_in_health():
    root = Path(__file__).resolve().parents[1]
    script = r"""
import json
from ocr97.server import create_app

app = create_app(instance_name="ocr97_slo_cap_test")
client = app.test_client()
payload = client.get("/ocr/health").get_json()
print(json.dumps(payload["slo_policy"]["p95_caps_by_class"]))
"""
    proc = subprocess.run(
        [sys.executable, "-c", script],
        cwd=root,
        env={
            **os.environ,
            "PYTHONPATH": str(root / "src"),
            "OCR97_OCR_GATEWAY_PREWARM_ENABLED": "0",
            "OCR97_OCR_GATEWAY_PREWARM_ON_STARTUP": "0",
            "OCR97_OCR_SMOKE_REQUIRED": "0",
            "OCR97_OCR_SLO_P95_SEMANTIC_MS": "1800000",
        },
        capture_output=True,
        text=True,
        timeout=12,
    )

    assert proc.returncode == 0, proc.stderr
    caps = json.loads(proc.stdout)
    assert caps["semantic_cleanup"] == 1800000


def test_preprocessed_best_runs_original_first_then_remaining_variants_in_executor(tmp_path, monkeypatch):
    from ocr97 import gateway

    image_path = tmp_path / "input.png"
    image_path.write_bytes(b"not-a-real-image")
    variant_paths = []
    for label in ("autocontrast", "threshold", "deskew_cv2_2.0"):
        path = tmp_path / f"{label}.png"
        path.write_bytes(b"variant")
        variant_paths.append({"label": label, "path": path, "detected_angle": None})

    calls = []
    executor = {}

    def fake_variants(path, temp_dir):
        return [{"label": "original", "path": image_path, "detected_angle": None}, *variant_paths]

    def fake_ocr_dual(payload):
        path = Path(payload["path"])
        calls.append(path.stem)
        return {
            "ok": True,
            "engine": payload["engine"],
            "markdown": f"weak OCR text from {path.stem}",
            "text": f"weak OCR text from {path.stem}",
            "confidence": 0.1,
            "quality": {"score": 0.1, "numeric_fidelity_score": 0.0, "structure_score": 0.0},
        }

    class RecordingExecutor:
        def __init__(self, max_workers):
            executor["max_workers"] = max_workers

        def __enter__(self):
            return self

        def __exit__(self, exc_type, exc, tb):
            return False

        def map(self, func, items):
            rows = list(items)
            executor["labels"] = [row["label"] for row in rows]
            return [func(row) for row in rows]

    monkeypatch.setenv("OCR97_OCR_PREPROCESS_FAST_ACCEPT", "0")
    monkeypatch.setenv("OCR97_OCR_PREPROCESS_WORKERS", "2")
    monkeypatch.setattr(gateway, "_preprocessed_image_variants", fake_variants)
    monkeypatch.setattr(gateway.ocr_dual_tool, "ocr_dual", fake_ocr_dual)
    monkeypatch.setattr(gateway.concurrent.futures, "ThreadPoolExecutor", RecordingExecutor)
    monkeypatch.setattr(gateway, "_looks_like_receipt_candidates", lambda candidates: False)

    out = gateway._local_image_preprocessed_best_extract(
        image_path,
        goal="test",
        max_pages=1,
        max_chars=2000,
        route_mode="balanced",
    )

    assert calls[:2] == ["input", "input"]
    assert executor["max_workers"] == 2
    assert executor["labels"] == ["autocontrast", "threshold", "deskew_cv2_2.0"]
    assert {row["preprocess"] for row in out["local_image_candidates"]} >= {
        "original",
        "autocontrast",
        "threshold",
        "deskew_cv2_2.0",
    }
