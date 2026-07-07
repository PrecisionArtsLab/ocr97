import json
from pathlib import Path

from ocr97.release_corpus import write_release_manifest
from ocr97.release_grade import grade_release


def test_release_corpus_expands_to_120_self_scoring_cases(tmp_path):
    root = Path(__file__).resolve().parents[1]
    output = tmp_path / "release_97_gate_manifest.json"

    result = write_release_manifest(root / "benchmarks" / "mixed_corpus_manifest.json", output)
    manifest = json.loads(output.read_text(encoding="utf-8"))
    categories = {case["category"] for case in manifest["cases"]}

    assert result["case_count"] == 120
    assert result["self_score"] == 100
    assert len(categories) >= 10
    assert any(case["id"].endswith("_table_first") for case in manifest["cases"])


def test_release_grade_marks_incomplete_smoke_below_97(tmp_path):
    summary = {
        "manifest_case_count": 120,
        "best_score_avg": 96,
        "worst_score_avg": 91,
        "steps": [
            {"summary": {"score_avg": 96, "below_75_cases": 0, "latency_avg_ms": 12000}},
            {"summary": {"score_avg": 91, "below_75_cases": 1, "latency_avg_ms": 42000}},
        ],
    }
    manifest = {
        "cases": [{"id": f"case_{idx}", "category": f"cat_{idx % 12}"} for idx in range(120)]
    }
    queue = {"runs": [{"id": "still_pending", "status": "pending"}]}
    summary_path = tmp_path / "summary.json"
    manifest_path = tmp_path / "manifest.json"
    queue_path = tmp_path / "queue.json"
    summary_path.write_text(json.dumps(summary), encoding="utf-8")
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
    queue_path.write_text(json.dumps(queue), encoding="utf-8")

    result = grade_release(summary_path, manifest_path=manifest_path, queue_path=queue_path)

    assert result["grade"] < 97
    assert result["passes_97_gate"] is False
    assert "still_pending" in result["pending_runs"]
    assert result["views"]["all_lanes_stress"]["grade"] < 97


def test_release_grade_splits_production_router_from_fallback_lanes(tmp_path):
    summary = {
        "manifest_case_count": 120,
        "steps": [
            {
                "engine": "native_pdf_text",
                "summary": {"mode": "gateway_native_pdf_text", "score_avg": 100, "below_75_cases": 0, "latency_avg_ms": 400},
            },
            {
                "engine": "local_image_preprocessed_best",
                "summary": {
                    "mode": "gateway_image_local_image_preprocessed_best_rotated",
                    "score_avg": 95,
                    "below_75_cases": 0,
                    "latency_avg_ms": 9000,
                },
            },
            {
                "engine": "tesseract",
                "summary": {"mode": "gateway_image_tesseract_rotated", "score_avg": 78, "below_75_cases": 12, "latency_avg_ms": 1200},
            },
        ],
    }
    manifest = {
        "cases": [{"id": f"case_{idx}", "category": f"cat_{idx % 12}"} for idx in range(120)]
    }
    summary_path = tmp_path / "summary.json"
    manifest_path = tmp_path / "manifest.json"
    summary_path.write_text(json.dumps(summary), encoding="utf-8")
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")

    result = grade_release(summary_path, manifest_path=manifest_path)

    assert result["views"]["production_router"]["grade"] == 100
    assert result["passes_97_gate"] is True
    assert result["views"]["all_lanes_stress"]["grade"] < 97
    assert result["views"]["production_router"]["below_75_cases"] == 0
    assert result["views"]["fallback_lane_stress"]["worst_score_avg"] == 78
    assert result["views"]["fallback_lane_stress"]["below_75_cases"] == 12


def test_select_cases_supports_source_case_id_alias():
    from ocr97.mixed_corpus_benchmark import _select_cases

    manifest = {
        "cases": [
            {"id": "invoice_line_items_baseline", "source_case_id": "invoice_line_items", "category": "invoice"},
            {"id": "vendor_invoice_services_rotated", "source_case_id": "vendor_invoice_services", "category": "invoice"},
            {"id": "purchase_order_dense", "source_case_id": "purchase_order", "category": "procurement"},
        ]
    }

    selected = _select_cases(manifest, ids=["invoice_line_items", "purchase_order"])
    assert len(selected["cases"]) == 2
    assert selected["cases"][0]["id"] == "invoice_line_items_baseline"
    assert selected["cases"][1]["id"] == "purchase_order_dense"
