import json
from pathlib import Path

from ocr97.capability_audit import audit_run_dir
from ocr97.spark_3090_plan import build_plan, validate_examples, validate_queue


def test_3090_longform_scope_files_are_valid():
    root = Path(__file__).resolve().parents[1]
    examples = json.loads((root / "benchmarks" / "ocr97_3090_longform_examples.json").read_text(encoding="utf-8"))
    queue = json.loads((root / "benchmarks" / "ocr97_3090_longform_queue.json").read_text(encoding="utf-8"))

    assert validate_examples(examples) == []
    assert validate_queue(queue) == []
    plan = build_plan(examples, queue)
    assert "Full run" in plan
    assert "3090" in plan
    assert "field consensus" in plan.lower()


def test_capability_audit_flags_field_and_table_gaps(tmp_path):
    result = {
        "mode": "gateway_image_tesseract_clean",
        "score_avg": 50,
        "results": [
            {
                "id": "invoice_line_items",
                "variant": "clean",
                "engine": "tesseract",
                "status_code": 200,
                "latency_ms": 1200,
                "field_consensus_used": False,
                "score": {
                    "score": 50,
                    "expected_table_rows": 4,
                    "actual_table_rows": 2,
                    "fields": [
                        {"name": "total", "matched": False},
                        {"name": "invoice_number", "matched": True},
                    ],
                },
            }
        ],
    }
    run_dir = tmp_path / "run"
    run_dir.mkdir()
    (run_dir / "tesseract_broad_clean.json").write_text(json.dumps(result), encoding="utf-8")

    audit = audit_run_dir(run_dir)

    assert audit["weak_case_count"] == 1
    assert audit["reason_counts"]["below_75"] == 1
    assert audit["reason_counts"]["field_miss"] == 1
    assert audit["reason_counts"]["table_row_gap"] == 1
