"""
clip_model.py
─────────────
Shared CLIP model singleton — loaded once into memory, reused across
detector.py and quality.py to avoid double-loading the large weights.
"""
import torch

try:
    import clip as openai_clip
    _CLIP_AVAILABLE = True
except ImportError:
    _CLIP_AVAILABLE = False

_model     = None
_preprocess = None
_device    = None


def get_clip():
    """Return (model, preprocess, device). Loads weights on first call."""
    global _model, _preprocess, _device
    if _model is None:
        if not _CLIP_AVAILABLE:
            raise RuntimeError(
                "CLIP not installed. Run: pip install openai-clip"
            )
        _device = "cuda" if torch.cuda.is_available() else "cpu"
        _model, _preprocess = openai_clip.load("ViT-B/32", device=_device)
        _model.eval()
    return _model, _preprocess, _device


def tokenize(texts: list[str]):
    """Thin wrapper around clip.tokenize so callers don't import clip directly."""
    if not _CLIP_AVAILABLE:
        raise RuntimeError("CLIP not available")
    return openai_clip.tokenize(texts, truncate=True)
