from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, Mapping, Tuple


ALL_DOC_CLASSES: frozenset[str] = frozenset(
    {
        "digital_pdf",
        "table_dense",
        "scanned_pdf",
        "handwritten",
        "chart_or_figure",
        "forms_or_checkboxes",
        "photo",
    }
)


@dataclass(frozen=True)
class EngineSpec:
    name: str
    label: str
    provider: str
    engine_class: str
    supported_doc_classes: frozenset[str]
    module_name: str = ""
    package_name: str = ""
    model_id_env: str = ""
    model_id_default: str = ""
    model_dir_env: str = ""
    model_dir_default: str = ""
    assets_required: bool = False
    ready_override_env: str = ""
    runtime_signatures: Tuple[str, ...] = ()
    default_lane_signature: str = ""
    optional_lane: bool = False
    native_api_module: str = ""
    cli_env: str = ""
    cli_default_cmd: str = ""
    aliases: Tuple[str, ...] = ()
    class_priority: int = 50


ENGINE_SPECS: Dict[str, EngineSpec] = {
    "native_pdf_text": EngineSpec(
        name="native_pdf_text",
        label="Native PDF Text",
        provider="native_pdf",
        engine_class="native_text",
        supported_doc_classes=frozenset({"digital_pdf", "table_dense"}),
        module_name="fitz",
        package_name="pymupdf",
        runtime_signatures=("native_pdf_text",),
        default_lane_signature="native_pdf_text",
        aliases=("native_text", "pdf_native_text"),
        class_priority=10,
    ),
    "local_image_best": EngineSpec(
        name="local_image_best",
        label="Local Image Router",
        provider="local_router",
        engine_class="image_router",
        supported_doc_classes=frozenset({"photo"}),
        module_name="pytesseract",
        package_name="pytesseract",
        runtime_signatures=("tesseract_worker", "rapidocr_worker"),
        default_lane_signature="local_image_best",
        aliases=("image_router", "local_image_router"),
        class_priority=10,
    ),
    "local_image_preprocessed_best": EngineSpec(
        name="local_image_preprocessed_best",
        label="Local Image Preprocessor",
        provider="local_preprocessor",
        engine_class="image_preprocessor",
        supported_doc_classes=frozenset({"photo", "scanned_pdf", "handwritten"}),
        module_name="pytesseract",
        package_name="pytesseract",
        runtime_signatures=("pillow_preprocess", "tesseract_worker", "rapidocr_worker"),
        default_lane_signature="local_image_preprocessed_best",
        aliases=("image_preprocessor", "local_image_preprocessor"),
        class_priority=10,
    ),
    "rapidocr": EngineSpec(
        name="rapidocr",
        label="RapidOCR",
        provider="rapidocr_local",
        engine_class="compat_fallback",
        supported_doc_classes=ALL_DOC_CLASSES,
        module_name="rapidocr_onnxruntime",
        package_name="rapidocr-onnxruntime",
        runtime_signatures=("rapidocr_worker",),
        default_lane_signature="rapidocr_worker",
        aliases=("compat_rapidocr", "rapidocr_local"),
        class_priority=10,
    ),
    "tesseract": EngineSpec(
        name="tesseract",
        label="Tesseract",
        provider="tesseract_local",
        engine_class="compat_fallback",
        supported_doc_classes=ALL_DOC_CLASSES,
        module_name="pytesseract",
        package_name="pytesseract",
        runtime_signatures=("tesseract_worker",),
        default_lane_signature="tesseract_worker",
        aliases=("compat_tesseract", "tesseract_local"),
        class_priority=20,
    ),
    "gb10_qwen_ocr": EngineSpec(
        name="gb10_qwen_ocr",
        label="Semantic Cleanup Vision OCR",
        provider="qwen_vision_cleanup",
        engine_class="semantic_cleanup",
        supported_doc_classes=ALL_DOC_CLASSES,
        runtime_signatures=("qwen_ocr_worker",),
        default_lane_signature="qwen_ocr_worker",
        optional_lane=True,
        aliases=("semantic_cleanup", "vision_cleanup", "remote_vision_cleanup"),
        class_priority=10,
    ),
    "gb10_got_ocr2": EngineSpec(
        name="gb10_got_ocr2",
        label="Dense Scan Vision OCR",
        provider="got_ocr2",
        engine_class="dense_scan",
        supported_doc_classes=frozenset({"forms_or_checkboxes", "scanned_pdf", "handwritten", "photo"}),
        runtime_signatures=("got_ocr2_worker",),
        default_lane_signature="got_ocr2_worker",
        optional_lane=True,
        aliases=("dense_scan_vision", "dense_scan", "got_ocr2"),
        class_priority=10,
    ),
    "gb10_paddleocr_vl": EngineSpec(
        name="gb10_paddleocr_vl",
        label="Layout Vision OCR",
        provider="paddleocr_vl",
        engine_class="layout",
        supported_doc_classes=frozenset({"digital_pdf", "table_dense", "chart_or_figure"}),
        module_name="paddleocr",
        package_name="paddleocr",
        model_id_env="OCR97_PADDLEOCR_VL_MODEL_ID",
        model_id_default="PaddleOCR-VL",
        model_dir_env="OCR97_PADDLEOCR_VL_MODEL_DIR",
        model_dir_default="paddleocr_vl",
        assets_required=True,
        ready_override_env="OCR97_OCR_ENGINE_PADDLEOCR_VL_READY",
        runtime_signatures=("paddleocr_vl_worker",),
        default_lane_signature="paddleocr_vl_worker",
        optional_lane=True,
        aliases=("layout_vision", "table_layout_vision", "paddleocr_vl"),
        class_priority=10,
    ),
    "mineru2_5": EngineSpec(
        name="mineru2_5",
        label="Structure Parser",
        provider="mineru",
        engine_class="structure_parser",
        supported_doc_classes=frozenset({"digital_pdf", "table_dense", "scanned_pdf", "handwritten"}),
        module_name="mineru",
        package_name="mineru",
        model_id_env="OCR97_MINERU2_5_MODEL_ID",
        model_id_default="opendatalab/MinerU",
        model_dir_env="OCR97_MINERU2_5_MODEL_DIR",
        model_dir_default="mineru2_5",
        assets_required=True,
        ready_override_env="OCR97_OCR_ENGINE_MINERU2_5_READY",
        runtime_signatures=("mineru_native_api", "mineru_cli_fallback"),
        default_lane_signature="mineru_native_unknown",
        optional_lane=True,
        native_api_module="mineru.cli.client",
        cli_env="OCR97_MINERU2_5_CMD",
        cli_default_cmd="mineru",
        aliases=("structure_parser", "mineru"),
        class_priority=10,
    ),
    "olmocr2": EngineSpec(
        name="olmocr2",
        label="Linearization OCR",
        provider="olmocr",
        engine_class="linearization",
        supported_doc_classes=frozenset({"digital_pdf", "table_dense"}),
        module_name="olmocr",
        package_name="olmocr",
        model_id_env="OCR97_OLMOCR2_MODEL_ID",
        model_id_default="allenai/olmOCR-2-7B",
        model_dir_env="OCR97_OLMOCR2_MODEL_DIR",
        model_dir_default="olmocr2",
        assets_required=True,
        ready_override_env="OCR97_OCR_ENGINE_OLMOCR2_READY",
        runtime_signatures=("olmocr_native_api", "olmocr_cli_fallback"),
        default_lane_signature="olmocr_native_unknown",
        optional_lane=True,
        native_api_module="olmocr.pipeline",
        cli_env="OCR97_OLMOCR2_CMD",
        cli_default_cmd="olmocr",
        aliases=("linearization", "olmocr"),
        class_priority=10,
    ),
}


ENGINE_SELECTION_PLANS: Dict[str, Dict[str, Tuple[str, ...]]] = {
    "quality_first": {
        "forms_or_checkboxes": ("semantic_cleanup", "dense_scan", "compat_fallback"),
        "chart_or_figure": ("semantic_cleanup", "layout", "compat_fallback"),
        "digital_pdf": ("native_text", "layout", "structure_parser", "linearization", "semantic_cleanup", "compat_fallback"),
        "table_dense": ("native_text", "layout", "structure_parser", "linearization", "semantic_cleanup", "compat_fallback"),
        "scanned_pdf": ("dense_scan", "structure_parser", "semantic_cleanup", "compat_fallback"),
        "handwritten": ("dense_scan", "structure_parser", "semantic_cleanup", "compat_fallback"),
        "photo": ("image_router", "semantic_cleanup", "dense_scan", "compat_fallback"),
    },
    "balanced": {
        "forms_or_checkboxes": ("semantic_cleanup", "compat_fallback"),
        "chart_or_figure": ("semantic_cleanup", "compat_fallback"),
        "digital_pdf": ("native_text", "layout", "semantic_cleanup", "compat_fallback"),
        "table_dense": ("native_text", "layout", "semantic_cleanup", "compat_fallback"),
        "scanned_pdf": ("dense_scan", "semantic_cleanup", "compat_fallback"),
        "handwritten": ("dense_scan", "semantic_cleanup", "compat_fallback"),
        "photo": ("image_router", "semantic_cleanup", "compat_fallback"),
    },
}


def _alias_map() -> Dict[str, str]:
    alias_to_engine: Dict[str, str] = {}
    for name, spec in ENGINE_SPECS.items():
        alias_to_engine[name] = name
        for alias in spec.aliases:
            alias_to_engine[str(alias).strip().lower()] = name
    return alias_to_engine


ENGINE_ALIASES = _alias_map()


def normalize_engine_name(name: str) -> str:
    raw = str(name or "").strip().lower()
    if raw in {"", "auto", "gb10_auto"}:
        return raw
    return ENGINE_ALIASES.get(raw, raw)


def get_engine_spec(name: str) -> EngineSpec | None:
    normalized = normalize_engine_name(name)
    return ENGINE_SPECS.get(normalized)


def engine_provider(name: str) -> str:
    spec = get_engine_spec(name)
    return spec.provider if spec else ""


def engine_class(name: str) -> str:
    spec = get_engine_spec(name)
    return spec.engine_class if spec else "unknown"


def engine_aliases(name: str) -> Tuple[str, ...]:
    spec = get_engine_spec(name)
    if not spec:
        return ()
    return tuple(spec.aliases)


def engine_names(*, include_optional: bool = True) -> Tuple[str, ...]:
    names = []
    for name, spec in ENGINE_SPECS.items():
        if include_optional or not spec.optional_lane:
            names.append(name)
    return tuple(names)


def engine_supports_doc_class(name: str, doc_class: str) -> bool:
    spec = get_engine_spec(name)
    if not spec:
        return False
    return str(doc_class or "").strip().lower() in spec.supported_doc_classes


def engine_is_optional(name: str) -> bool:
    spec = get_engine_spec(name)
    return bool(spec.optional_lane) if spec else False


def engine_runtime_signatures(name: str) -> Tuple[str, ...]:
    spec = get_engine_spec(name)
    return tuple(spec.runtime_signatures) if spec else ()


def default_lane_signature(name: str) -> str:
    spec = get_engine_spec(name)
    return str(spec.default_lane_signature or "") if spec else ""


def engine_module_name(name: str) -> str:
    spec = get_engine_spec(name)
    return str(spec.module_name or "") if spec else ""


def engine_package_name(name: str) -> str:
    spec = get_engine_spec(name)
    if not spec:
        return ""
    return str(spec.package_name or spec.module_name or "")


def engine_model_id(name: str) -> str:
    spec = get_engine_spec(name)
    if not spec or not spec.model_id_env:
        return ""
    return str(os.getenv(spec.model_id_env, spec.model_id_default)).strip() or str(spec.model_id_default or "")


def engine_model_dir(name: str) -> Path:
    spec = get_engine_spec(name)
    if not spec:
        return Path.home() / ".cache" / "ocr97" / str(normalize_engine_name(name) or "unknown")
    if spec.model_dir_env:
        raw = str(os.getenv(spec.model_dir_env, "")).strip()
        if raw:
            return Path(raw)
    default_dir = str(spec.model_dir_default or spec.name).strip()
    return Path.home() / ".cache" / default_dir


def engine_assets_required(name: str) -> bool:
    spec = get_engine_spec(name)
    return bool(spec.assets_required) if spec else False


def engine_ready_override_env(name: str) -> str:
    spec = get_engine_spec(name)
    return str(spec.ready_override_env or "") if spec else ""


def engine_native_api_module(name: str) -> str:
    spec = get_engine_spec(name)
    return str(spec.native_api_module or "") if spec else ""


def engine_cli_command(name: str) -> str:
    spec = get_engine_spec(name)
    if not spec or not spec.cli_default_cmd:
        return ""
    return str(os.getenv(spec.cli_env, spec.cli_default_cmd)).strip() or str(spec.cli_default_cmd)


def class_engine_names(class_name: str) -> Tuple[str, ...]:
    key = str(class_name or "").strip().lower()
    rows = [
        spec.name
        for spec in sorted(ENGINE_SPECS.values(), key=lambda item: (item.engine_class, int(item.class_priority), item.name))
        if spec.engine_class == key
    ]
    return tuple(rows)


def filter_optional_engines(engine_list: Iterable[str], *, allow_optional: bool) -> list[str]:
    filtered: list[str] = []
    for engine in engine_list:
        normalized = normalize_engine_name(engine)
        if not normalized or normalized in {"auto", "gb10_auto"}:
            continue
        if not allow_optional and engine_is_optional(normalized):
            continue
        filtered.append(normalized)
    return dedupe_engine_names(filtered)


def dedupe_engine_names(engine_list: Iterable[str]) -> list[str]:
    seen = set()
    out: list[str] = []
    for engine in engine_list:
        normalized = normalize_engine_name(engine)
        if not normalized or normalized in seen:
            continue
        seen.add(normalized)
        out.append(normalized)
    return out


def select_engine_chain(doc_class: str, route_mode: str, forced_engine: str = "") -> list[str]:
    normalized_doc_class = str(doc_class or "photo").strip().lower() or "photo"
    normalized_route_mode = "balanced" if str(route_mode or "").strip().lower() == "balanced" else "quality_first"
    plan = ENGINE_SELECTION_PLANS.get(normalized_route_mode, ENGINE_SELECTION_PLANS["quality_first"])
    class_plan = plan.get(normalized_doc_class, plan["photo"])
    chain: list[str] = []
    forced = normalize_engine_name(forced_engine)
    if forced and forced not in {"auto", "gb10_auto"}:
        chain.append(forced)
    for engine_class_name in class_plan:
        for engine_name in class_engine_names(engine_class_name):
            if engine_supports_doc_class(engine_name, normalized_doc_class):
                chain.append(engine_name)
    return dedupe_engine_names(chain)


def public_capability_rows() -> list[Mapping[str, object]]:
    rows = []
    for spec in ENGINE_SPECS.values():
        rows.append(
            {
                "name": spec.name,
                "label": spec.label,
                "provider": spec.provider,
                "class": spec.engine_class,
                "supported_doc_classes": sorted(spec.supported_doc_classes),
                "aliases": sorted(spec.aliases),
                "optional_lane": bool(spec.optional_lane),
            }
        )
    return rows
