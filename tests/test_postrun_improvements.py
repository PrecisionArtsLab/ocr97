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


def test_lane_slo_gate_waits_for_minimum_sample_count():
    from ocr97.gateway import _evaluate_lane_slo_gate

    cold_timeout = {
        "last_24h_count": 1,
        "timeout_rate": 1.0,
        "fallback_rate": 1.0,
        "p95_latency_ms": 180000.0,
    }
    warming = _evaluate_lane_slo_gate(cold_timeout, p95_cap_ms=20000, min_samples=8)

    assert warming["metrics_enforced"] is False
    assert warming["timeout_rate_ok"] is True
    assert warming["fallback_rate_ok"] is True
    assert warming["p95_ok"] is True

    steady_state = _evaluate_lane_slo_gate(
        {**cold_timeout, "last_24h_count": 8},
        p95_cap_ms=20000,
        min_samples=8,
    )

    assert steady_state["metrics_enforced"] is True
    assert steady_state["timeout_rate_ok"] is False
    assert steady_state["fallback_rate_ok"] is False
    assert steady_state["p95_ok"] is False


def test_ollama_model_preflight_returns_all_matching_models(monkeypatch):
    from ocr97 import dual_tool

    class FakeResponse:
        ok = True

        def json(self):
            return {
                "models": [
                    {"name": "qwen3-vl:32b"},
                    {"name": "gemma3:12b"},
                    {"name": "qwen2.5vl:7b"},
                ]
            }

    monkeypatch.setattr(dual_tool.requests, "get", lambda *args, **kwargs: FakeResponse())

    result = dual_tool._ollama_model_available(
        "http://127.0.0.1:11435",
        ["missing-vl:7b", "qwen3-vl:32b", "qwen2.5vl:7b"],
    )

    assert result["ok"] is True
    assert result["available_model"] == "qwen3-vl:32b"
    assert result["available_models"] == ["qwen3-vl:32b", "qwen2.5vl:7b"]


def test_qwen_runner_skips_missing_model_and_fast_accepts(tmp_path, monkeypatch):
    from ocr97 import dual_tool

    image_path = tmp_path / "phone_photo.png"
    image_path.write_bytes(b"not-a-real-image")
    calls = []

    monkeypatch.setattr(dual_tool, "DEFAULT_GB10_QWEN_OCR_MODEL", "missing-vl:7b")
    monkeypatch.setattr(dual_tool, "DEFAULT_GB10_QWEN_OCR_FALLBACK_MODEL", "qwen3-vl:32b")
    monkeypatch.setattr(dual_tool, "DEFAULT_GB10_QWEN_OLLAMA_URL", "http://127.0.0.1:11435")
    monkeypatch.setattr(dual_tool, "DEFAULT_OCR_COMPAT_ENABLED", False)
    monkeypatch.setattr(dual_tool, "DEFAULT_OCR_PHASE2_ENABLED", True)
    monkeypatch.setattr(dual_tool, "DEFAULT_OCR_QWEN_SELF_CONSISTENCY_ENABLED", False)
    monkeypatch.setattr(
        dual_tool,
        "_ollama_model_available",
        lambda base_url, models: {
            "ok": True,
            "reason": "model_available",
            "available_model": "qwen3-vl:32b",
            "available_models": ["qwen3-vl:32b"],
        },
    )

    strong_markdown = "\n".join(
        [
            "# INVOICE 2026",
            "| Item | Qty | Price |",
            "| --- | ---: | ---: |",
            *[f"| Engine part {index} | {index} | ${index * 125}.00 |" for index in range(1, 14)],
            "Total: $11375.00",
            "Payment terms: 30 days. Reference 987654321.",
        ]
    )

    def fake_generate(image_path, prompt, model, *, base_url, lane):
        calls.append({"model": model, "base_url": base_url, "lane": lane})
        return {
            "ok": True,
            "engine": "gb10_qwen_ocr",
            "model": model,
            "markdown": strong_markdown,
            "text": strong_markdown,
            "route": lane,
        }

    monkeypatch.setattr(dual_tool, "_ollama_generate_with_image", fake_generate)

    result = dual_tool._gb10_qwen_ocr(image_path, goal="literal OCR", max_chars=8000, max_pages=1)

    assert result["ok"] is True
    assert result["model"] == "qwen3-vl:32b"
    assert [row["model"] for row in calls] == ["qwen3-vl:32b"]
    assert result["model_preflight"][0]["available_models"] == ["qwen3-vl:32b"]


def test_gateway_prewarm_model_selection_falls_back_when_preferred_is_missing(monkeypatch):
    from ocr97 import gateway

    class FakeResponse:
        ok = True

        def json(self):
            return {"models": [{"name": "gemma3:12b"}, {"name": "qwen3-vl:32b"}]}

    monkeypatch.setattr(gateway.requests, "get", lambda *args, **kwargs: FakeResponse())

    selected = gateway._infer_ollama_model(
        "http://127.0.0.1:11435",
        preferred="qwen2.5vl:7b",
    )

    assert selected == "qwen3-vl:32b"


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
