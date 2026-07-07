import json
from pathlib import Path

from ocr97 import real_corpus
from ocr97.helix97 import collect_failure_records
from ocr97.truth_runner import write_pdf_fixture


def test_real_corpus_scores_local_source_and_writes_helix_comparison(tmp_path):
    source_pdf = tmp_path / "invoice.pdf"
    write_pdf_fixture(
        source_pdf,
        "Invoice Number INV-100\nInvoice Date January 25, 2016\nSubtotal $90.00\nTotal Due $93.50",
        title="Sample invoice",
    )
    manifest = {
        "name": "test_real_corpus",
        "cases": [
            {
                "id": "local_invoice",
                "label": "Local invoice",
                "source_path": str(source_pdf),
                "filename": "local_invoice.pdf",
                "required_tokens": ["Invoice Number", "INV-100", "Total Due"],
                "expected_fields": [
                    {"name": "invoice_number", "aliases": ["Invoice Number"], "expected": "INV-100", "type": "text"},
                    {"name": "invoice_date", "aliases": ["Invoice Date"], "expected": "2016-01-25", "type": "date"},
                    {"name": "total_due", "aliases": ["Total Due"], "expected": "93.50", "type": "money"},
                ],
            }
        ],
    }
    manifest_path = tmp_path / "manifest.json"
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")

    summary = real_corpus.fetch_score(manifest_path, tmp_path / "run")
    comparison = json.loads((tmp_path / "run" / "baseline_comparison.json").read_text(encoding="utf-8"))

    assert summary["score_avg"] == 100
    assert comparison["engines"][0]["engine"] == "ocr97"
    assert comparison["engines"][0]["results"][0]["score"]["fields"][0]["ranked_candidates"]
    assert Path(comparison["engines"][0]["results"][0]["artifact_path"]).exists()


def test_real_corpus_failures_feed_existing_helix_collector(tmp_path):
    source_pdf = tmp_path / "invoice.pdf"
    write_pdf_fixture(source_pdf, "Subtotal $90.00\nTotal Due $93.50", title="Sample invoice")
    manifest = {
        "name": "test_real_corpus",
        "cases": [
            {
                "id": "local_invoice",
                "source_path": str(source_pdf),
                "filename": "local_invoice.pdf",
                "expected_fields": [
                    {"name": "total_due", "aliases": ["Total Due"], "expected": "100.00", "type": "money"},
                ],
            }
        ],
    }
    manifest_path = tmp_path / "manifest.json"
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")

    real_corpus.fetch_score(manifest_path, tmp_path / "run")
    helix_manifest = collect_failure_records(tmp_path / "run" / "baseline_comparison.json", output_dir=tmp_path / "helix")

    rows = [json.loads(line) for line in Path(helix_manifest["dataset_path"]).read_text(encoding="utf-8").splitlines()]
    assert helix_manifest["record_count"] == 1
    assert rows[0]["source"]["comparison_path"].endswith("baseline_comparison.json")
    assert rows[0]["field"]["failure_bucket"] in {"candidate_found_but_not_selected", "numeric_candidate_wrong"}
