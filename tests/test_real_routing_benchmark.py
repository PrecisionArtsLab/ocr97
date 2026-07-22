from PIL import Image

from ocr97.real_routing_benchmark import _materialize_case, _score, _token_coverage


def test_real_routing_token_coverage_is_case_insensitive():
    assert _token_coverage("Put down a resolution", "PUT down a resolution now") == 1.0


def test_real_routing_materializes_image_only_pdf(tmp_path):
    source = tmp_path / "receipt.jpg"
    Image.new("RGB", (200, 100), "white").save(source)
    manifest = tmp_path / "benchmarks" / "manifest.json"
    manifest.parent.mkdir()
    manifest.write_text("{}", encoding="utf-8")

    output, metadata = _materialize_case(
        {"id": "scan_pdf", "source_path": str(source), "as_image_pdf": True},
        manifest,
        tmp_path / "sources",
    )

    assert output.suffix == ".pdf"
    assert output.exists()
    assert metadata["provenance"]["image_only_pdf"] is True


def test_real_routing_nonempty_validation_does_not_claim_accuracy():
    score = _score(
        {"validation": "nonempty", "min_chars": 5},
        {"ok": True, "text": "readable text"},
        {},
    )

    assert score["score"] == 100
    assert score["accuracy_claim"] is False
