"""OCR97 package."""

__all__ = ["ocr_dual"]
__version__ = "0.1.0"


def __getattr__(name: str):
    if name == "ocr_dual":
        from .dual_tool import ocr_dual

        return ocr_dual
    raise AttributeError(f"module 'ocr97' has no attribute {name!r}")

