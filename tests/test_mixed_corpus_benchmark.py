import json
from pathlib import Path

from ocr97.mixed_corpus_benchmark import MixedCorpusBenchmark
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
