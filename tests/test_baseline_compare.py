import json
from pathlib import Path

from ocr97.baseline_compare import run_baseline_comparison, summarize_comparison
from ocr97.truth_benchmark import load_manifest


def test_summarize_comparison_reports_ocr97_delta():
    summary = summarize_comparison(
        [
            {"engine": "ocr97", "available": True, "score_avg": 92, "scored_case_count": 3},
            {"engine": "tesseract", "available": True, "score_avg": 80, "scored_case_count": 3},
            {"engine": "easyocr", "available": False, "score_avg": 0, "scored_case_count": 0},
        ]
    )

    assert summary["verdict"] == "ocr97_leads"
    assert summary["best_engine"] == "ocr97"
    assert summary["best_baseline"] == "tesseract"
    assert summary["ocr97_vs_best_baseline_delta"] == 12


def test_baseline_comparison_writes_reports_and_skips_unknown_engine(tmp_path):
    root = Path(__file__).resolve().parents[1]
    manifest = load_manifest(root / "benchmarks" / "truth10_manifest.json")

    result = run_baseline_comparison(
        manifest,
        fixture_dir=tmp_path / "fixtures",
        output_dir=tmp_path / "comparison",
        variant="clean",
        engines=["unknown_engine"],
        max_cases=2,
    )

    assert result["case_count"] == 2
    assert result["engines"][0]["available"] is False
    assert result["engines"][0]["skipped_case_count"] == 2
    assert (tmp_path / "comparison" / "baseline_comparison.json").exists()
    assert (tmp_path / "comparison" / "baseline_comparison.md").exists()


def test_baseline_comparison_scores_actual_tesseract_if_available(tmp_path):
    root = Path(__file__).resolve().parents[1]
    manifest = load_manifest(root / "benchmarks" / "truth10_manifest.json")
    subset = {**manifest, "cases": [case for case in manifest["cases"] if case["id"] == "invoice_summary"]}

    result = run_baseline_comparison(
        subset,
        fixture_dir=tmp_path / "fixtures",
        output_dir=tmp_path / "comparison",
        variant="clean",
        engines=["tesseract"],
        max_cases=1,
    )

    row = result["engines"][0]
    assert row["engine"] == "tesseract"
    if row["available"]:
        assert row["scored_case_count"] == 1
        assert row["results"][0]["score"]["fields"]
        assert "ranked_candidates" in row["results"][0]["score"]["fields"][0]
    else:
        assert row["skipped_case_count"] == 1
    payload = json.loads((tmp_path / "comparison" / "baseline_comparison.json").read_text(encoding="utf-8"))
    assert payload["engines"][0]["engine"] == "tesseract"
