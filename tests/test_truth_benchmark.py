import json
from pathlib import Path
import subprocess
import sys

from ocr97.truth_benchmark import load_manifest, score_case, score_manifest


def test_truth10_manifest_scores_clean_samples_high():
    root = Path(__file__).resolve().parents[1]
    manifest = load_manifest(root / "benchmarks" / "truth10_manifest.json")

    result = score_manifest(manifest)

    assert result["case_count"] == 10
    assert result["score_avg"] >= 95
    assert all(row["field_score"] == 100 for row in result["results"])


def test_required_tokens_ignore_pdf_spacing_noise():
    case = {
        "id": "pdf_spacing_token",
        "sample_text": "TOTAL   $         567.50",
        "required_tokens": ["$ 567.50"],
    }

    result = score_case(case)

    assert result["required_token_score"] == 100
    assert result["missing_tokens"] == []


def test_required_tokens_accept_compact_ocr_spacing_and_zero_confusion():
    case = {
        "id": "ocr_compact_required_tokens",
        "sample_text": "PONumber:P04482 AmountDue:$2,481.37 ReleaseGateRef:RG-OO1-BASELINE",
        "required_tokens": ["PO Number", "Amount Due", "Release Gate Ref: RG-001-BASELINE"],
    }

    result = score_case(case)

    assert result["required_token_score"] == 100
    assert result["missing_tokens"] == []


def test_truth_benchmark_penalizes_wrong_numeric_field():
    case = {
        "id": "wrong_total",
        "required_tokens": ["Total"],
        "expected_fields": [{"name": "total", "aliases": ["Total"], "expected": "100.00", "type": "money"}],
        "sample_text": "Total: $91.00",
    }

    result = score_case(case)

    assert result["field_score"] == 0
    assert result["score"] < 50


def test_score_case_accepts_compact_ocr_text_identifier():
    case = {
        "id": "compact_invoice_number",
        "expected_fields": [{"name": "invoice_number", "aliases": ["Invoice"], "expected": "INV 88411", "type": "text"}],
        "sample_text": "Invoice with line items Invoice:INV88411 Subtotal: $1,686.20",
    }

    result = score_case(case)

    assert result["field_score"] == 100
    assert result["fields"][0]["matched"] is True


def test_score_case_accepts_common_ocr_letter_confusion_in_text_identifier():
    case = {
        "id": "po_confusion",
        "expected_fields": [{"name": "po_number", "aliases": ["PO Number"], "expected": "PO 4482", "type": "text"}],
        "sample_text": "Invoice: INV 73019 pO Number: PQ 4482 Amount Due: $2,481.37",
    }

    result = score_case(case)

    assert result["field_score"] == 100
    assert result["fields"][0]["matched"] is True


def test_score_case_handles_no_space_label_and_numeric_ocr_noise():
    case = {
        "id": "net_amount_noise",
        "expected_fields": [{"name": "net_amount", "aliases": ["Net Amount"], "expected": "5320.25", "type": "money"}],
        "sample_text": "Brokerage summary NetAmount: $5<320.25 Release Gate Ref: RG-036",
    }

    result = score_case(case)

    assert result["field_score"] == 100
    assert result["fields"][0]["matched"] is True


def test_score_case_uses_total_as_amount_due_fallback():
    case = {
        "id": "amount_due_total_fallback",
        "expected_fields": [{"name": "amount_due", "aliases": ["Amount Due"], "expected": "2481.37", "type": "money"}],
        "sample_text": "Subtotal: $2,298.50 Tax: $182.87 Field consensus: Total: $2,481.37",
    }

    result = score_case(case)

    assert result["field_score"] == 100
    assert result["fields"][0]["matched"] is True


def test_truth_benchmark_cli_writes_summary(tmp_path):
    root = Path(__file__).resolve().parents[1]
    output = tmp_path / "truth10_result.json"
    proc = subprocess.run(
        [
            sys.executable,
            "-m",
            "ocr97.truth_benchmark",
            "--manifest",
            str(root / "benchmarks" / "truth10_manifest.json"),
            "--output",
            str(output),
        ],
        cwd=root,
        env={"PYTHONPATH": str(root / "src")},
        capture_output=True,
        text=True,
        timeout=10,
    )

    assert proc.returncode == 0, proc.stderr
    summary = json.loads(proc.stdout)
    assert summary["case_count"] == 10
    assert summary["score_avg"] >= 95
    assert json.loads(output.read_text(encoding="utf-8"))["case_count"] == 10

