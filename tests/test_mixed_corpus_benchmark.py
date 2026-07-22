import json
from pathlib import Path
import pytest

from ocr97.mixed_corpus_benchmark import MixedCorpusBenchmark, _select_cases
from ocr97.truth_benchmark import load_manifest, score_manifest


def test_mixed_corpus_manifest_has_breadth_and_valid_truth():
    root = Path(__file__).resolve().parents[1]
    manifest = load_manifest(root / "benchmarks" / "mixed_corpus_manifest.json")
    categories = {str(case.get("category") or "") for case in manifest["cases"]}

    assert len(manifest["cases"]) >= 20
    assert len(categories) >= 10
    assert score_manifest(manifest)["score_avg"] == 100


def test_mixed_corpus_plan_only_writes_monitorable_files(tmp_path):
    root = Path(__file__).resolve().parents[1]
    runner = MixedCorpusBenchmark(
        manifest_path=root / "benchmarks" / "mixed_corpus_manifest.json",
        output_dir=tmp_path / "mixed",
        broad_limit=3,
        focus_limit=2,
        broad_variants=["clean"],
        focus_variants=["rotated"],
        plan_only=True,
    )

    summary = runner.run()
    progress = json.loads((tmp_path / "mixed" / "progress.json").read_text(encoding="utf-8"))

    assert summary["status"] == "planned"
    assert summary["manifest_case_count"] >= 20
    assert summary["manifest_self_score"] == 100
    assert len(progress["steps"]) == 3
    assert all(step["status"] == "planned" for step in progress["steps"])
    assert (tmp_path / "mixed" / "MIXED_CORPUS_REPORT.md").exists()


def test_scanned_fallback_plan_has_auto_route_and_nonempty_preprocessing(tmp_path):
    root = Path(__file__).resolve().parents[1]
    manifest_path = root / "benchmarks" / "scanned_fallback_manifest.json"
    manifest = load_manifest(manifest_path)

    assert score_manifest(manifest)["score_avg"] == 100

    runner = MixedCorpusBenchmark(
        manifest_path=manifest_path,
        output_dir=tmp_path / "scanned",
        broad_limit=4,
        focus_limit=3,
        broad_variants=["clean"],
        focus_variants=["noisy_scan"],
        skip_native_pdf=True,
        include_auto_route=True,
        auto_route_variants=["clean"],
        plan_only=True,
    )
    summary = runner.run()
    progress = json.loads((tmp_path / "scanned" / "progress.json").read_text(encoding="utf-8"))
    by_name = {step["name"]: step for step in progress["steps"]}

    assert summary["manifest_self_score"] == 100
    assert by_name["auto_route_broad_clean"]["requested_lane_strict"] is False
    assert by_name["auto_route_broad_clean"]["benchmark_kind"] == "end_to_end_auto_route"
    assert by_name["preprocessed_focus_noisy_scan"]["case_count"] == 3
    assert by_name["tesseract_broad_clean"]["benchmark_kind"] == "forced_engine_diagnostic"


def test_focus_ids_explicit_matching_is_strict(tmp_path):
    root = Path(__file__).resolve().parents[1]
    runner = MixedCorpusBenchmark(
        manifest_path=root / "benchmarks" / "scanned_fallback_manifest.json",
        output_dir=tmp_path / "strict_focus",
        broad_limit=2,
        focus_limit=2,
        focus_ids=["does_not_exist_foo_bar"],
        plan_only=True,
    )
    with pytest.raises(ValueError, match="focus_ids_not_found"):
        runner.run()


def test_select_cases_supports_variant_alias_and_source_case_id():
    manifest = {
        "cases": [
            {"id": "invoice_line_items_rotated", "source_case_id": "invoice_line_items", "category": "invoice"},
            {"id": "vendor_invoice_services_baseline", "source_case_id": "vendor_invoice_services", "category": "invoice"},
            {"id": "shipping_manifest", "category": "logistics"},
        ]
    }

    selected = _select_cases(manifest, ids=["invoice_line_items", "shipping_manifest"], strict=True)
    assert len(selected["cases"]) == 2
    assert selected["cases"][0]["id"] == "invoice_line_items_rotated"
    assert selected["cases"][1]["id"] == "shipping_manifest"


def test_scanned_fallback_manifest_has_modality_and_real_world_classes():
    root = Path(__file__).resolve().parents[1]
    manifest = load_manifest(root / "benchmarks" / "scanned_fallback_manifest.json")

    assert len(manifest.get("cases") or []) == 40
    assert all("modality" in case for case in manifest["cases"])

    categories = {str(case.get("category") or "") for case in manifest["cases"]}
    modalities = {str(case.get("modality") or "") for case in manifest["cases"]}
    assert categories == {
        "fax_document",
        "photocopied_legal",
        "mobile_scan_receipt",
        "government_form",
        "medical_record",
        "handwritten_form",
        "carbon_copy",
        "physical_contract",
    }
    assert "phone_photograph" in modalities
    assert "fax_scan" in modalities
    assert "photocopied_hardcopy" in modalities
    assert "photographed" in modalities or "handwritten_scanned" in modalities


def test_step_summary_captures_route_and_fallback_metadata(monkeypatch, tmp_path):
    root = Path(__file__).resolve().parents[1]
    manifest_path = tmp_path / "manifest.json"
    manifest_path.write_text(
        json.dumps(
            {
                "name": "route-metadata-fixture",
                "cases": [
                    {"id": "route_case_1", "sample_text": "Total: 10.00", "category": "invoice"},
                ],
            }
        ),
        encoding="utf-8",
    )

    def fake_gateway_image_benchmark(manifest, fixture_dir, output_dir, variant, engine, requested_lane_strict, benchmark_kind):
        return {
            "name": "fake-benchmark",
            "mode": f"gateway_image_{engine}_{variant}",
            "case_count": 1,
            "score_avg": 93,
            "artifact_dir": str(output_dir),
            "results": [
                {
                    "id": "route_case_1",
                    "score": {"score": 93},
                    "latency_ms": 110,
                    "attempted_engines": ["auto", "tesseract", "local_image_preprocessed_best"],
                    "selected_attempt_index": 2,
                    "fallback_reason": "hard_case_routed",
                    "degraded_fallback": True,
                    "confidence": 0.82,
                }
            ],
        }

    def fake_gateway_pdf_benchmark(*args, **kwargs):
        return {
            "name": "fake-pdf",
            "mode": "gateway_native_pdf_text",
            "case_count": 0,
            "score_avg": 100,
            "results": [],
            "artifact_dir": str(kwargs.get("output_dir") or args[0]),
        }

    monkeypatch.setattr("ocr97.mixed_corpus_benchmark.run_gateway_image_truth_benchmark", fake_gateway_image_benchmark)
    monkeypatch.setattr("ocr97.mixed_corpus_benchmark.run_gateway_truth_benchmark", fake_gateway_pdf_benchmark)
    monkeypatch.setattr("ocr97.mixed_corpus_benchmark._llm_grade", lambda summary: "GRADE: 100")

    runner = MixedCorpusBenchmark(
        manifest_path=manifest_path,
        output_dir=tmp_path / "route_meta",
        broad_limit=1,
        include_heavy=False,
        focus_limit=1,
        broad_variants=["clean"],
        skip_native_pdf=True,
        include_auto_route=False,
        focus_ids=["route_case_1"],
        plan_only=False,
    )
    summary = runner.run()
    assert summary["steps"][0]["summary"]["attempted_engines"] == ["auto", "local_image_preprocessed_best", "tesseract"]
    assert summary["steps"][0]["summary"]["selected_attempt_indices"] == [2]
    assert summary["steps"][0]["summary"]["fallback_reasons"] == ["hard_case_routed"]
    assert summary["steps"][0]["summary"]["degraded_fallback_cases"] == 1
