"""Optional ONNX Runtime backend shared by Mobilint Python packages."""

from __future__ import annotations

import os
from collections.abc import Sequence
from typing import Any


class ONNXBackend:
    """Run an ONNX model through an optional ONNX Runtime installation.

    ``onnxruntime`` is imported only by :meth:`create`, keeping the base NPU
    package usable for MXQ-only applications.
    """

    def __init__(
        self,
        model_path: str,
        *,
        providers: Sequence[str] | None = None,
        ort_module: Any | None = None,
    ) -> None:
        self.model_path = model_path
        self.providers = (
            list(providers) if providers is not None else ["CPUExecutionProvider"]
        )
        self._ort_module = ort_module
        self.session: Any | None = None

    def _load_onnxruntime(self) -> Any:
        if self._ort_module is not None:
            return self._ort_module
        try:
            import onnxruntime
        except ImportError as exc:
            raise ImportError(
                "onnxruntime is required for ONNX inference. "
                "Install it with `pip install mblt-npu-python[onnxruntime]`."
            ) from exc
        return onnxruntime

    def create(self) -> None:
        """Create the ONNX Runtime inference session for ``model_path``."""

        if not os.path.isfile(self.model_path):
            raise FileNotFoundError(f"ONNX file not found at: {self.model_path}")
        self.session = self._load_onnxruntime().InferenceSession(
            self.model_path, providers=self.providers
        )

    def launch(self) -> None:
        """Match the MXQ backend lifecycle; ONNX Runtime needs no launch step."""

        self._require_session()

    def __call__(self, inputs: dict[str, Any]) -> Any:
        """Run inference and return every model output in ONNX Runtime order."""

        return self._require_session().run(None, inputs)

    def run(self, output_names: Sequence[str] | None, inputs: dict[str, Any]) -> Any:
        """Run inference for the requested ONNX output names."""

        return self._require_session().run(output_names, inputs)

    def get_inputs(self) -> Any:
        """Return ONNX Runtime input metadata."""

        return self._require_session().get_inputs()

    def get_outputs(self) -> Any:
        """Return ONNX Runtime output metadata."""

        return self._require_session().get_outputs()

    def get_dtype(self) -> str:
        """Return the first ONNX input element type."""

        return str(self.get_inputs()[0].type)

    def dispose(self) -> None:
        """Release the session reference held by this backend."""

        self.session = None

    def _require_session(self) -> Any:
        if self.session is None:
            raise RuntimeError("ONNX backend is not initialized; call create() first.")
        return self.session
