from ocr97.engine_registry import (
    engine_class,
    engine_is_optional,
    engine_supports_doc_class,
    filter_optional_engines,
    normalize_engine_name,
    select_engine_chain,
)


def test_generic_aliases_normalize_to_legacy_runtime_names():
    assert normalize_engine_name("layout_vision") == "gb10_paddleocr_vl"
    assert normalize_engine_name("dense_scan_vision") == "gb10_got_ocr2"
    assert normalize_engine_name("semantic_cleanup") == "gb10_qwen_ocr"
    assert normalize_engine_name("native_text") == "native_pdf_text"
    assert normalize_engine_name("compat_rapidocr") == "rapidocr"


def test_capability_chain_prefers_native_text_for_digital_pdf():
    chain = select_engine_chain("digital_pdf", "quality_first")

    assert chain[:4] == ["native_pdf_text", "gb10_paddleocr_vl", "mineru2_5", "olmocr2"]
    assert "gb10_qwen_ocr" in chain
    assert chain[-2:] == ["rapidocr", "tesseract"]


def test_capability_chain_prefers_image_router_for_photo():
    chain = select_engine_chain("photo", "balanced")

    assert chain[:3] == ["local_image_best", "gb10_qwen_ocr", "rapidocr"]


def test_explicit_generic_alias_is_placed_first_then_falls_back_by_capability():
    chain = select_engine_chain("photo", "quality_first", forced_engine="layout_vision")

    assert chain[0] == "gb10_paddleocr_vl"
    assert "local_image_best" in chain


def test_optional_lane_filter_removes_heavy_engines_but_keeps_portable_lanes():
    filtered = filter_optional_engines(
        ["local_image_best", "gb10_qwen_ocr", "rapidocr", "tesseract"],
        allow_optional=False,
    )

    assert filtered == ["local_image_best", "rapidocr", "tesseract"]


def test_engine_supports_doc_class_comes_from_registry():
    assert engine_supports_doc_class("layout_vision", "chart_or_figure") is True
    assert engine_supports_doc_class("layout_vision", "handwritten") is False
    assert engine_class("semantic_cleanup") == "semantic_cleanup"
    assert engine_is_optional("semantic_cleanup") is True
