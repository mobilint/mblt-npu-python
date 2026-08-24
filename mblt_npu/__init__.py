"""Shared NPU access for the mblt Python packages.

`MobilintNPUBackend` is defined exactly once, here, because the alternative
was a copy in each of mblt-vision-python, mblt-transformers-python and
mblt-MeloTTS-python that nothing would keep in sync.

`logging` lives here too rather than in vision, its only caller, because the
two are mutually dependent: npu_backend imports log_model_details and
log_model_details reads a MobilintNPUBackend's fields.
"""

from .npu_backend import (
    BACKEND_CLASSES,
    DEFAULT_TARGET_DEVICE,
    MobilintBackendAllocError,
    MobilintAriesBackend,
    MobilintNPUBackend,
    MobilintRegulusBackend,
    backend_class_for,
    normalize_target_device,
)
from .logging import log_model_details
from .onnx_backend import ONNXBackend

__version__ = "0.0.0"

__all__ = [
    "BACKEND_CLASSES",
    "DEFAULT_TARGET_DEVICE",
    "MobilintBackendAllocError",
    "MobilintAriesBackend",
    "MobilintNPUBackend",
    "MobilintRegulusBackend",
    "ONNXBackend",
    "backend_class_for",
    "log_model_details",
    "normalize_target_device",
]
