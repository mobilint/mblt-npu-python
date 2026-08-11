"""Tests for the optional ONNX Runtime backend."""

from __future__ import annotations

from pathlib import Path

import pytest

from mblt_npu import ONNXBackend


def test_onnx_backend_runs_a_session_with_injected_runtime(tmp_path: Path) -> None:
    """Create, run, inspect, and dispose an ONNX session without importing ONNX Runtime."""

    model_path = tmp_path / "model.onnx"
    model_path.write_bytes(b"onnx")

    class _Input:
        name = "images"
        type = "tensor(float)"

    class _Output:
        name = "scores"

    class _Session:
        def __init__(self, path: str, providers: list[str]) -> None:
            self.path = path
            self.providers = providers

        def run(self, output_names: object, inputs: dict[str, object]) -> list[object]:
            return [output_names, inputs]

        @staticmethod
        def get_inputs() -> list[_Input]:
            return [_Input()]

        @staticmethod
        def get_outputs() -> list[_Output]:
            return [_Output()]

    class _Runtime:
        InferenceSession = _Session

    backend = ONNXBackend(
        str(model_path), providers=["CPUExecutionProvider"], ort_module=_Runtime()
    )
    backend.create()
    backend.launch()

    assert backend({"images": "input"}) == [None, {"images": "input"}]
    assert backend.run(["scores"], {"images": "input"}) == [
        ["scores"],
        {"images": "input"},
    ]
    assert backend.get_dtype() == "tensor(float)"
    assert backend.get_outputs()[0].name == "scores"

    backend.dispose()
    with pytest.raises(RuntimeError, match=r"call create\(\) first"):
        backend.get_inputs()


def test_onnx_backend_requires_an_existing_model_file(tmp_path: Path) -> None:
    """Report a clear error before attempting to construct a missing model."""

    backend = ONNXBackend(str(tmp_path / "missing.onnx"), ort_module=object())

    with pytest.raises(FileNotFoundError, match="ONNX file not found"):
        backend.create()
