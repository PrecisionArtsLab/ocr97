from ocr97.field_evidence import classify_failure, field_candidates, normalize_date, normalize_number
from ocr97.truth_benchmark import score_case


def test_normalizers_handle_business_ocr_variants():
    assert normalize_number("USD $1,382.40") == "1382.4"
    assert normalize_number("(1,382.40)") == "-1382.4"
    assert normalize_number("$5<320.25") == "5320.25"
    assert normalize_number("$5,320-25") == "5320.25"
    assert normalize_date("2026/05/15") == "2026-05-15"
    assert normalize_date("05/15/26") == "2026-05-15"
    assert normalize_date("July 8, 2011") == "2011-07-08"


def test_total_candidate_prefers_total_context_over_subtotal():
    field = {"name": "total", "aliases": ["Total"], "expected": "1382.40", "type": "money"}
    text = "Subtotal: $1,280.00\nTax: $102.40\nGrand Total: $1,382.40"
    candidates = field_candidates(text, field)

    assert candidates[0]["normalized_value"] == "1382.4"
    assert candidates[0]["confidence"] > 0.5


def test_numeric_candidate_prefers_value_near_requested_label_on_dense_line():
    field = {"name": "agi", "aliases": ["Adjusted Gross Income"], "expected": "84500", "type": "money"}
    text = "Tax summary Adjusted Gross Income: $84,500 Taxable Income: $70,200 Tax Due: $9,180"
    candidates = field_candidates(text, field)

    assert candidates[0]["normalized_value"] == "84500"
    assert candidates[0]["reason"] == "numeric candidate near requested label"


def test_percent_candidate_prefers_percent_near_margin_label():
    field = {"name": "margin", "aliases": ["Margin"], "expected": "18.4", "type": "percent"}
    text = "Margin report Revenue: $220,000 Cost: $179,520 Margin: 18.4%"
    candidates = field_candidates(text, field)

    assert candidates[0]["normalized_value"] == "18.4"


def test_date_candidate_prefers_iso_date_near_receipt_date_over_ambiguous_later_date():
    field = {"name": "date", "aliases": ["Date"], "expected": "2026-04-24", "type": "date"}
    text = "Retail receipt Receipt Date: 2026-04-24 Items: $45.00\nDate: 26/04/2024"
    candidates = field_candidates(text, field)

    assert candidates[0]["normalized_value"] == "2026-04-24"


def test_address_candidate_builds_multiline_block():
    field = {"name": "address", "aliases": ["Address"], "expected": "1408 Walnut Street Springfield IL 62704", "type": "text"}
    text = "Acme Lab\nAddress: 1408 Walnut Street\nSpringfield IL 62704\nTotal: $48.62"
    candidates = field_candidates(text, field)

    assert candidates
    assert "walnut street" in candidates[0]["normalized_value"]
    assert "62704" in candidates[0]["normalized_value"]


def test_score_case_exposes_ranked_candidates_and_failure_buckets():
    case = {
        "id": "invoice_total_evidence",
        "expected_fields": [
            {"name": "total", "aliases": ["Total"], "expected": "1382.40", "type": "money"},
            {"name": "date", "aliases": ["Date"], "expected": "2026-05-15", "type": "date"},
        ],
        "sample_text": "Subtotal: $1,280.00\nGrand Total: $1,382.40\nDate: 05/15/2026",
    }

    result = score_case(case)

    assert result["field_score"] == 100
    assert result["failure_buckets"] == {}
    assert result["fields"][0]["ranked_candidates"]
    assert result["fields"][1]["source_evidence"]["normalized_value"] == "2026-05-15"


def test_score_case_accepts_expected_text_token_inside_selected_title_span():
    case = {
        "id": "w9_title",
        "expected_fields": [{"name": "form_number", "aliases": ["Form"], "expected": "W-9", "type": "text"}],
        "sample_text": "Form Request for Taxpayer Give form to the W-9",
    }

    result = score_case(case)

    assert result["field_score"] == 100
    assert result["fields"][0]["matched"]


def test_failure_bucket_distinguishes_missing_from_wrong_candidate():
    assert classify_failure(expected="100.00", candidates=[], matched=False, field_type="money") == "field_not_found"
    assert classify_failure(
        expected="100.00",
        candidates=[{"normalized_value": "99"}],
        matched=False,
        field_type="money",
    ) == "numeric_candidate_wrong"


def test_symbolic_text_alias_does_not_match_generic_title_line():
    field = {"name": "invoice_number", "aliases": ["INVOICE #"], "expected": "4", "type": "text"}
    text = "Vendor Name INVOICE\nAnytown, USA 00000 INVOICE # 4"
    candidates = field_candidates(text, field)

    assert candidates[0]["normalized_value"] == "4"
