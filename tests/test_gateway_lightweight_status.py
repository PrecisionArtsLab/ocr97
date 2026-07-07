import json
import os
from pathlib import Path
import subprocess
import sys


def test_gateway_health_and_capabilities_are_lightweight():
    root = Path(__file__).resolve().parents[1]
    script = r"""
import json
import sys
from ocr97.server import create_app

app = create_app(instance_name="ocr97_test")
client = app.test_client()
health = client.get("/ocr/health")
capabilities = client.get("/ocr/capabilities")
payload = {
    "health_status": health.status_code,
    "capabilities_status": capabilities.status_code,
    "health": health.get_json(),
    "capabilities": capabilities.get_json(),
    "dual_tool_loaded": "ocr97.dual_tool" in sys.modules,
    "local_inference_loaded": "ocr97.local_inference" in sys.modules,
}
print(json.dumps(payload))
"""
    proc = subprocess.run(
        [sys.executable, "-c", script],
        cwd=root,
        env={
            **os.environ,
            "PYTHONPATH": str(root / "src"),
            "OCR97_OCR_GATEWAY_PREWARM_ENABLED": "0",
            "OCR97_OCR_GATEWAY_PREWARM_ON_STARTUP": "0",
        },
        capture_output=True,
        text=True,
        timeout=10,
    )

    assert proc.returncode == 0, proc.stderr
    payload = json.loads(proc.stdout)
    assert payload["health_status"] == 200
    assert payload["capabilities_status"] == 200
    assert payload["health"]["diagnostic_mode"] == "lightweight_no_model_import"
    assert payload["health"]["prewarm"] == {
        "enabled": False,
        "on_startup": False,
    }
    assert payload["health"]["ollama"]["checked"] is False
    assert payload["health"]["feature_detection"]["document_feature_classifier"]["mode"] == "lightweight_heuristic"
    assert payload["health"]["feature_detection"]["visual_control_detector"]["mode"] == "opencv_contour_heuristic"
    assert payload["health"]["feature_detection"]["layout_model_classifier"]["runtime_loaded"] is False
    assert payload["health"]["capability"] == "ocr97_lightweight_no_model_import"
    assert payload["capabilities"]["diagnostic_mode"] == "lightweight_no_model_import"
    assert payload["capabilities"]["prewarm"] == {
        "enabled": False,
        "on_startup": False,
    }
    assert payload["capabilities"]["capability"] == "ocr97_lightweight_no_model_import"
    capability_names = {row.get("name") for row in payload["capabilities"]["engines"]}
    assert "document_feature_classifier" in capability_names
    assert "visual_control_detector" in capability_names
    assert "layout_model_classifier" in capability_names
    assert "chart_or_figure" in payload["capabilities"]["doc_classes"]
    assert "forms_or_checkboxes" in payload["capabilities"]["doc_classes"]
    assert payload["dual_tool_loaded"] is False
    assert payload["local_inference_loaded"] is False

