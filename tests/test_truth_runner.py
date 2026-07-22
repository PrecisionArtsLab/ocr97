import json
import os
from pathlib import Path
import subprocess
import sys

import pytest

from ocr97.gateway import _append_field_consensus, _classify_content_doc_type, _fast_accept_local_image_candidate, _field_consensus_from_candidates, _native_pdf_text_extract, _preprocessed_image_variants, _receipt_region_retry_candidates, _rescore_local_image_candidates, _routing_metadata
from ocr97.receipt_fields import append_receipt_fields, normalize_receipt_date, receipt_fields_from_candidates
from ocr97.overnight_benchmark import _field_totals, _score_avg
from ocr97.sroie_runner import score_sroie_payload
from ocr97.truth_benchmark import _number_partial_score, _table_row_count, load_manifest
from ocr97.truth_runner import (
    _should_escalate_image_fallback,
    run_gateway_image_truth_benchmark,
    run_gateway_truth_benchmark,
    write_image_fixture,
)


def test_preprocessing_generates_dynamic_cv2_deskew_candidate(tmp_path):
    pytest.importorskip("cv2")
    image_path = tmp_path / "rotated.png"
    write_image_fixture(
        image_path,
        "Invoice: INV 2048\nSubtotal: $1,280.00\nTax: $102.40\nTotal: $1,382.40",
        title="Invoice summary",
        variant="rotated",
    )

    variants = _preprocessed_image_variants(image_path, tmp_path)

    assert any(str(row.get("label") or "").startswith("deskew_cv2_") for row in variants)
    assert any(row.get("detected_angle") is not None for row in variants)


def test_routing_metadata_marks_last_resort_tesseract_and_selected_position():
    attempts = [
        {"engine": "local_image_best", "ok": False, "fallback_reason": "empty"},
        {"engine": "gb10_qwen_ocr", "ok": False, "fallback_reason": "timeout"},
        {"engine": "tesseract", "ok": True, "quality_score": 0.51},
    ]

    telemetry = _routing_metadata(
        attempts=attempts,
        selected_attempt_index=2,
        selected_engine="tesseract",
    )

    assert telemetry["attempted_engines"] == ["local_image_best", "gb10_qwen_ocr", "tesseract"]
    assert telemetry["selected_attempt_index"] == 2
    assert telemetry["selected_attempt_number"] == 3
    assert telemetry["chain_depth"] == 3
    assert telemetry["fallback_used"] is True
    assert telemetry["degraded_fallback"] is True
    assert telemetry["fallback_status"] == "degraded_fallback"
    assert telemetry["fallback_reason"] == "prior_engines_failed:local_image_best,gb10_qwen_ocr"


def test_candidate_scoring_penalizes_numeric_drift_and_rewards_consensus():
    candidates = [
        {"ok": True, "engine": "tesseract", "markdown": "Invoice: INV 2048\nSubtotal: $1,280.00\nTax: $102.40\nTotal: $1,382.40"},
        {"ok": True, "engine": "rapidocr", "markdown": "Invoice: INV 2048\nSubtotal: $1,280.00\nTax: $102.40\nTotal: $1,382.40"},
        {"ok": True, "engine": "tesseract", "markdown": "Invoice: INV 2648\nSubtotal: $1,280.20\nTax: $102.40\nTota1: $1,382.49"},
    ]

    _rescore_local_image_candidates(candidates)

    assert candidates[0]["_selection_score"] > candidates[2]["_selection_score"]
    assert candidates[0]["score_components"]["numeric_consensus"] > candidates[2]["score_components"]["numeric_consensus"]
    assert candidates[2]["score_components"]["penalty"] > candidates[0]["score_components"]["penalty"]


def test_field_consensus_reports_supported_values_with_confidence():
    candidates = [
        {"ok": True, "engine": "tesseract", "preprocess": "deskew_neg3", "_selection_score": 90.0, "markdown": "Invoice: INV 2048 Subtotal: $1,280.00 Tax: $102.40 Total: $1,382.40"},
        {"ok": True, "engine": "rapidocr", "preprocess": "original", "_selection_score": 88.0, "markdown": "Invoice: INV 2048 Subtotal: $1,280.00 Tax: $102.40 Tota1: $1,382.40"},
        {"ok": True, "engine": "tesseract", "preprocess": "threshold", "_selection_score": 70.0, "markdown": "Invoice: INV 2648 Subtotal: $1,280.20 Tax: $102.40 Total: $1,382.49"},
    ]

    consensus = _field_consensus_from_candidates(candidates)
    by_field = {row["field"]: row for row in consensus}

    assert by_field["invoice_number"]["normalized_value"] == "inv 2048"
    assert by_field["subtotal"]["normalized_value"] == "1280.00"
    assert by_field["total"]["normalized_value"] == "1382.40"
    assert by_field["total"]["confidence"] > 0.5
    assert by_field["total"]["evidence_reasons"]


def test_field_consensus_uses_ranked_field_evidence_on_dense_finance_lines():
    candidates = [
        {
            "ok": True,
            "engine": "tesseract",
            "preprocess": "original",
            "_selection_score": 96.0,
            "markdown": "Tax summary Adjusted Gross Income: $84,500 Taxable Income: $70,200 Tax Due: $9,180",
        },
        {
            "ok": True,
            "engine": "rapidocr",
            "preprocess": "deskew",
            "_selection_score": 92.0,
            "markdown": "Taxsummary AdjustedGross Income:$84,500 TaxableIncome:$70,200 TaxDue:$9,180",
        },
    ]

    consensus = _field_consensus_from_candidates(candidates)
    by_field = {row["field"]: row for row in consensus}

    assert by_field["agi"]["normalized_value"] == "84500"
    assert by_field["taxable_income"]["normalized_value"] == "70200"
    assert by_field["tax_due"]["normalized_value"] == "9180"
    assert any("requested label" in reason for reason in by_field["agi"]["evidence_reasons"])


def test_field_consensus_appends_ranked_summary_values_not_table_row_values():
    candidates = [
        {
            "ok": True,
            "engine": "tesseract",
            "preprocess": "original",
            "_selection_score": 94.0,
            "markdown": "Market Value: $69,800.00 Equity: $61,300.00 Cash: $8,500.00 | QQQ | 30 | $10,300.00 | SPY | 120 | $51,000.00 |",
        }
    ]

    merged = _append_field_consensus("Brokerage positions", _field_consensus_from_candidates(candidates))

    assert "Market Value: $69,800.00" in merged
    assert "Market Value: $10,300.00" not in merged


def test_field_consensus_appends_parseable_field_summary():
    merged = _append_field_consensus(
        "Digital balance sheet (Metric Amount 1 1 Assets 1 $184,250 1",
        [
            {"field": "assets", "value": "$184,250", "confidence": 0.8, "support": 3},
            {"field": "liabilities", "value": "$74,200", "confidence": 0.7, "support": 2},
        ],
    )

    assert "Field consensus:" in merged
    assert "Assets: $184,250" in merged
    assert "Liabilities: $74,200" in merged


def test_fast_accept_skips_preprocessing_for_strong_non_receipt_candidate(monkeypatch):
    monkeypatch.setenv("OCR97_OCR_PREPROCESS_FAST_ACCEPT", "1")
    candidates = [
        {
            "ok": True,
            "engine": "tesseract",
            "preprocess": "original",
            "markdown": "Invoice INV-2048\nSubtotal $1,280.00\nTax $102.40\nTotal $1,382.40\nAccount 998877\nPayment due 2026-05-01",
            "quality": {"score": 0.85, "numeric_fidelity_score": 1.0, "structure_score": 0.1},
        },
        {
            "ok": True,
            "engine": "rapidocr",
            "preprocess": "original",
            "markdown": "Invoice INV-2048\nSubtotal $1,280.00\nTax $102.40\nTotal $1,382.40\nAccount 998877\nPayment due 2026-05-01",
            "quality": {"score": 0.85, "numeric_fidelity_score": 1.0, "structure_score": 0.1},
        },
    ]
    _rescore_local_image_candidates(candidates)

    accepted = _fast_accept_local_image_candidate(candidates)

    assert accepted is not None
    assert accepted["preprocess"] == "original"


def test_fast_accept_does_not_skip_receipt_like_candidate(monkeypatch):
    monkeypatch.setenv("OCR97_OCR_PREPROCESS_FAST_ACCEPT", "1")
    candidates = [
        {
            "ok": True,
            "engine": "tesseract",
            "preprocess": "original",
            "markdown": "Tax invoice\nCashier 02\nGST 6%\nInvoice no 12345\nTotal $1,382.40\nAddress 10 Main Street 12345",
            "quality": {"score": 0.95, "numeric_fidelity_score": 1.0, "structure_score": 0.1},
        }
    ]
    _rescore_local_image_candidates(candidates)

    assert _fast_accept_local_image_candidate(candidates) is None


def test_receipt_fields_recover_company_suffix_and_ocr_date():
    candidates = [
        {
            "ok": True,
            "engine": "tesseract",
            "preprocess": "angle_sweep_pos4.0",
            "_selection_score": 96.0,
            "markdown": "PETRON BKT LANJAN $B ALSERKAM ENTERPRISE\nTAX INVOICE\nDate: O1/O2/2O18\nTotal RM inc GST: 4.90",
        },
        {
            "ok": True,
            "engine": "rapidocr",
            "preprocess": "original",
            "_selection_score": 90.0,
            "markdown": "PETRON BKT LANJAN SB\nDATE : 01/02/2018\nTOTAL 4.90",
        },
    ]

    fields = receipt_fields_from_candidates(candidates)
    by_field = {row["field"]: row for row in fields}

    assert by_field["company"]["value"] == "PETRON BKT LANJAN SB"
    assert by_field["date"]["value"] == "01/02/2018"
    assert normalize_receipt_date("O1/O2/2O18") == "01/02/2018"
    assert normalize_receipt_date("os/oz/2018") == "05/02/2018"


def test_receipt_fields_append_company_and_date_block():
    merged = append_receipt_fields(
        "PETRON BKT LANJAN $B\nTotal RM inc GST: 4.90",
        [
            {"field": "company", "value": "PETRON BKT LANJAN SB", "confidence": 0.8, "support": 2},
            {"field": "date", "value": "01/02/2018", "confidence": 0.8, "support": 2},
            {"field": "total", "value": "4.90", "confidence": 0.8, "support": 2},
        ],
    )

    assert "Receipt fields:" in merged
    assert "Company: PETRON BKT LANJAN SB" in merged
    assert "Date: 01/02/2018" in merged
    assert "Total: 4.90" in merged


def test_receipt_fields_recover_total_from_total_line():
    fields = receipt_fields_from_candidates(
        [
            {
                "ok": True,
                "engine": "gb10_qwen_ocr",
                "_selection_score": 80.0,
                "markdown": "Cash Bill\nTotal Item Discount: 0.00\nTotal Amount: 170.00\nRound Amt: 0.00\nTOTAL: 170.00",
            }
        ]
    )
    by_field = {row["field"]: row for row in fields}

    assert by_field["total"]["value"] == "170.00"


def test_sroie_score_uses_receipt_field_recovery():
    payload = {
        "markdown": "PETRON BKT LANJAN $B\nKM 458.4 BKT LANJAN UTARA L RAYA UTARA SELATAN SG BULOH 47000 SUNGAI BUL\nTotal RM inc GST: 4.90",
        "receipt_fields": [
            {"field": "company", "value": "PETRON BKT LANJAN SB"},
            {"field": "date", "value": "01/02/2018"},
            {"field": "address", "value": "KM 458 4 BKT LANJAN UTARA L RAYA UTARA SELATAN SG BULOH 47000 SUNGAI BUL"},
        ],
    }
    expected = {
        "company": "PETRON BKT LANJAN SB",
        "date": "01/02/2018",
        "address": "KM 458.4 BKT LANJAN UTARA, L/RAYA UTARA SELATAN,SG BULOH 47000 SUNGAI BUL",
        "total": "4.90",
    }

    score = score_sroie_payload(payload, expected)

    assert score["score"] == 100
    assert all(row["matched"] for row in score["fields"])


def test_overnight_summary_helpers_score_by_engine_rows():
    rows = [
        {"score": {"score": 100, "fields": [{"name": "company", "matched": True}, {"name": "date", "matched": True}]}},
        {"score": {"score": 50, "fields": [{"name": "company", "matched": True}, {"name": "date", "matched": False}]}},
    ]

    assert _score_avg(rows) == 75.0
    totals = _field_totals(rows)
    assert totals["company"]["accuracy"] == 100.0
    assert totals["date"]["accuracy"] == 50.0


def test_receipt_fields_recover_address_from_header_lines():
    candidates = [
        {
            "ok": True,
            "engine": "tesseract_receipt_region",
            "preprocess": "receipt_header_top45_psm6",
            "_selection_score": 100.0,
            "markdown": "PERNIAGAAN ZHENG HUI\nNO.S9 JALAN PERMAS 916\nBANDAR BARU PERMAS JAYA\n81750 JOHOR BAHRU\nTEL: 07-386 7524",
        }
    ]

    fields = receipt_fields_from_candidates(candidates)
    by_field = {row["field"]: row for row in fields}

    assert "address" in by_field
    assert "JALAN PERMAS" in by_field["address"]["value"]
    assert "81750 JOHOR BAHRU" in by_field["address"]["value"]


def test_receipt_region_retry_extracts_header_evidence(tmp_path):
    pytest.importorskip("pytesseract")
    image_path = tmp_path / "receipt.png"
    write_image_fixture(
        image_path,
        "SIN LIANHAP SDN BHD\nLOT 13, JALAN IPOH\nTAX INVOICE\nInvoice No: H0003939\nDate: 05/02/2018\nTotal: 7.30",
        title="",
        variant="low_contrast",
    )

    rows = _receipt_region_retry_candidates(image_path, max_chars=2000)
    text = "\n".join(str(row.get("markdown") or row.get("text") or "") for row in rows if row.get("ok"))

    assert rows
    assert any(row.get("receipt_region") for row in rows)
    assert "LIANHAP" in text.upper()


def test_truth_runner_scores_actual_gateway_native_pdf_outputs(tmp_path):
    root = Path(__file__).resolve().parents[1]
    manifest = load_manifest(root / "benchmarks" / "truth10_manifest.json")

    result = run_gateway_truth_benchmark(
        manifest,
        fixture_dir=tmp_path / "fixtures",
        output_dir=tmp_path / "artifacts",
    )

    assert result["case_count"] == 10
    assert result["score_avg"] >= 90
    assert all(Path(row["artifact_path"]).exists() for row in result["results"])
    assert all(row["ok"] for row in result["results"])
    assert all(row["engine"] == "native_pdf_text" for row in result["results"])


def test_truth_runner_scores_actual_gateway_image_outputs(tmp_path):
    root = Path(__file__).resolve().parents[1]
    manifest = load_manifest(root / "benchmarks" / "truth10_manifest.json")
    subset_ids = {"invoice_summary", "bank_statement", "receipt"}
    subset = {**manifest, "cases": [case for case in manifest["cases"] if case["id"] in subset_ids]}

    result = run_gateway_image_truth_benchmark(
        subset,
        fixture_dir=tmp_path / "image_fixtures",
        output_dir=tmp_path / "image_artifacts",
        variant="mild_degraded",
        engine="tesseract",
    )

    assert result["case_count"] == 3
    assert result["score_avg"] >= 75
    assert all(Path(row["artifact_path"]).exists() for row in result["results"])
    assert all(row["ok"] for row in result["results"])
    assert all(row["engine"] == "tesseract" for row in result["results"])


def test_hard_image_fallback_escalates_on_high_value_field_miss(monkeypatch):
    monkeypatch.delenv("OCR97_OCR_FALLBACK_ESCALATION", raising=False)
    score = {
        "score": 73,
        "fields": [
            {
                "name": "invoice_number",
                "matched": False,
                "failure_bucket": "text_candidate_wrong",
                "expected": "INV 88411",
            },
            {"name": "total", "matched": True, "failure_bucket": "matched", "expected": "1812.67"},
        ],
    }

    should, reason, failures = _should_escalate_image_fallback(score, variant="rotated", engine="tesseract")

    assert should is True
    assert reason.startswith("score_below_threshold") or reason.startswith("high_value_field_failure")
    assert failures[0]["field"] == "invoice_number"


def test_clean_fallback_does_not_escalate_when_fields_are_acceptable(monkeypatch):
    monkeypatch.delenv("OCR97_OCR_FALLBACK_ESCALATION", raising=False)
    score = {
        "score": 94,
        "fields": [
            {"name": "invoice_number", "matched": True, "failure_bucket": "matched", "expected": "INV 88411"},
            {"name": "total", "matched": True, "failure_bucket": "matched", "expected": "1812.67"},
        ],
    }

    should, reason, failures = _should_escalate_image_fallback(score, variant="clean", engine="tesseract")

    assert should is False
    assert reason == "variant_not_hard_image"
    assert failures == []


def test_truth_runner_scores_local_image_best_router(tmp_path):
    root = Path(__file__).resolve().parents[1]
    manifest = load_manifest(root / "benchmarks" / "truth10_manifest.json")
    subset = {**manifest, "cases": [case for case in manifest["cases"] if case["id"] == "invoice_summary"]}

    result = run_gateway_image_truth_benchmark(
        subset,
        fixture_dir=tmp_path / "best_fixtures",
        output_dir=tmp_path / "best_artifacts",
        variant="mild_degraded",
        engine="local_image_best",
    )

    row = result["results"][0]
    artifact = json.loads(Path(row["artifact_path"]).read_text(encoding="utf-8"))
    assert result["score_avg"] >= 75
    assert row["ok"] is True
    assert artifact["router"] == "local_image_best"
    assert len(artifact["local_image_candidates"]) >= 2


def test_truth_runner_scores_preprocessed_image_best_router(tmp_path):
    root = Path(__file__).resolve().parents[1]
    manifest = load_manifest(root / "benchmarks" / "truth10_manifest.json")
    subset = {**manifest, "cases": [case for case in manifest["cases"] if case["id"] == "invoice_summary"]}

    result = run_gateway_image_truth_benchmark(
        subset,
        fixture_dir=tmp_path / "preprocessed_fixtures",
        output_dir=tmp_path / "preprocessed_artifacts",
        variant="low_contrast",
        engine="local_image_preprocessed_best",
    )

    row = result["results"][0]
    artifact = json.loads(Path(row["artifact_path"]).read_text(encoding="utf-8"))
    assert result["score_avg"] >= 80
    assert row["ok"] is True
    assert artifact["router"] == "local_image_preprocessed_best"
    assert artifact["selected_preprocess"]
    assert len(artifact["local_image_candidates"]) >= 2
    assert "score_components" in artifact["local_image_candidates"][0]
    assert artifact["field_consensus"]
    assert artifact["field_consensus_used"] is True


def test_truth_runner_scores_rotated_preprocessed_invoice_and_table(tmp_path):
    root = Path(__file__).resolve().parents[1]
    manifest = load_manifest(root / "benchmarks" / "truth10_manifest.json")
    subset_ids = {"invoice_summary", "digital_balance_sheet"}
    subset = {**manifest, "cases": [case for case in manifest["cases"] if case["id"] in subset_ids]}

    result = run_gateway_image_truth_benchmark(
        subset,
        fixture_dir=tmp_path / "rotated_fixtures",
        output_dir=tmp_path / "rotated_artifacts",
        variant="rotated",
        engine="local_image_preprocessed_best",
    )

    assert result["score_avg"] >= 90
    assert all(row["ok"] for row in result["results"])
    assert all(json.loads(Path(row["artifact_path"]).read_text(encoding="utf-8"))["field_consensus"] for row in result["results"])


def test_truth_runner_cli(tmp_path):
    root = Path(__file__).resolve().parents[1]
    output = tmp_path / "gateway_truth.json"
    proc = subprocess.run(
        [
            sys.executable,
            "-m",
            "ocr97.truth_runner",
            "--manifest",
            str(root / "benchmarks" / "truth10_manifest.json"),
            "--fixture-dir",
            str(tmp_path / "fixtures"),
            "--artifact-dir",
            str(tmp_path / "artifacts"),
            "--output",
            str(output),
        ],
        cwd=root,
        env={**os.environ, "PYTHONPATH": str(root / "src")},
        capture_output=True,
        text=True,
        timeout=30,
    )

    assert proc.returncode == 0, proc.stderr
    summary = json.loads(proc.stdout)
    assert summary["case_count"] == 10
    assert summary["score_avg"] >= 90
    assert json.loads(output.read_text(encoding="utf-8"))["score_avg"] >= 90


def test_classify_content_doc_type_detects_banking():
    text = "opening balance $5,000\nclosing balance $4,800\ndeposits $500\nwithdrawals $700\naccount number 123456789"
    assert _classify_content_doc_type(text) == "banking"


def test_classify_content_doc_type_detects_brokerage():
    text = "shares 100\nmarket value $15,000\nsymbol AAPL\nportfolio summary\nunrealized gain $2,000\ncost basis $13,000"
    assert _classify_content_doc_type(text) == "brokerage"


def test_classify_content_doc_type_detects_table_dense():
    text = "| Date | Description | Amount |\n| --- | --- | --- |\n| 2026-01-01 | Opening | $5,000 |\n| 2026-01-02 | Deposit | $500 |\n| 2026-01-03 | Withdrawal | -$200 |"
    assert _classify_content_doc_type(text) == "table_dense"


def test_fast_accept_blocked_for_banking_without_table_structure(monkeypatch):
    monkeypatch.setenv("OCR97_OCR_PREPROCESS_FAST_ACCEPT", "1")
    banking_text = (
        "opening balance $5,000\nclosing balance $4,800\ndeposits $500\n"
        "withdrawals $700\naccount number 123456789\nstatement date 2026-01-31\n"
        "transaction date description amount balance\n"
    ) * 3
    candidates = [
        {
            "ok": True,
            "engine": "tesseract",
            "preprocess": "original",
            "markdown": banking_text,
            "quality": {"score": 0.95, "numeric_fidelity_score": 1.0, "structure_score": 0.1},
        }
    ]
    _rescore_local_image_candidates(candidates)

    assert _fast_accept_local_image_candidate(candidates) is None


def test_fast_accept_allowed_for_banking_with_table_structure(monkeypatch):
    monkeypatch.setenv("OCR97_OCR_PREPROCESS_FAST_ACCEPT", "1")
    banking_text = (
        "opening balance | $5,000 | credit\n"
        "closing balance | $4,800 | debit\n"
        "deposits | $500 | credit\n"
        "withdrawals | $700 | debit\n"
        "account number | 123456789 | active\n"
    ) * 3
    candidates = [
        {
            "ok": True,
            "engine": "tesseract",
            "preprocess": "original",
            "markdown": banking_text,
            "quality": {"score": 0.95, "numeric_fidelity_score": 1.0, "structure_score": 0.8},
        },
        {
            "ok": True,
            "engine": "rapidocr",
            "preprocess": "original",
            "markdown": banking_text,
            "quality": {"score": 0.95, "numeric_fidelity_score": 1.0, "structure_score": 0.8},
        },
    ]
    _rescore_local_image_candidates(candidates)

    assert _fast_accept_local_image_candidate(candidates) is not None


def test_number_partial_score_gradations():
    assert _number_partial_score("100", "100") == 1.0
    assert _number_partial_score("100.05", "100") == 1.0
    assert _number_partial_score("101", "100") == 0.75
    assert _number_partial_score("103", "100") == 0.40
    assert _number_partial_score("110", "100") == 0.0
    assert _number_partial_score("foo", "100") == 0.0
    assert _number_partial_score("100", "foo") == 0.0


def test_table_row_count_counts_financial_label_value_rows():
    text = (
        "Total Assets: $184,250\n"
        "Total Liabilities  $92,000\n"
        "Net Income: $45,000.00\n"
        "Revenue: 500000\n"
        "some lowercase row: $1,000\n"
    )
    count = _table_row_count(text)
    assert count >= 3


def test_native_pdf_returns_sparse_error_for_scanned_like_pdf(tmp_path, monkeypatch):
    fitz = pytest.importorskip("fitz")
    monkeypatch.setenv("OCR97_OCR_NATIVE_PDF_MIN_CHARS_PER_PAGE", "500")

    pdf_path = tmp_path / "sparse.pdf"
    doc = fitz.open()
    page = doc.new_page()
    page.insert_text((50, 100), "Hi")
    doc.save(str(pdf_path))
    doc.close()

    result = _native_pdf_text_extract(pdf_path, max_pages=5, max_chars=50000)
    assert result["ok"] is False
    assert result["error"] == "native_pdf_sparse:likely_scanned"
