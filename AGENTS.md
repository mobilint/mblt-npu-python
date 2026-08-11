---
description: Guidance for coding agents working on the shared Mobilint NPU Python runtime.
paths:
  - "**"
---

# Mobilint NPU Python Agent Guide

## Scope

`mblt-npu-python` owns the shared runtime boundary used by Mobilint Python
packages: MXQ backend selection and lifecycle, board normalization, Hugging
Face MXQ resolution, optional ONNX Runtime inference, logging, and shared
pytest options. Vision and Model Zoo own their own model APIs and CLI behavior.

Before editing, run `git status --short`; read `pyproject.toml`, package exports,
the affected backend, and focused tests. Preserve unrelated changes.

## Runtime Contracts

- Keep `MobilintNPUBackend` as the MXQ compatibility surface. Preserve `create`,
  `launch`, callable inference, `get_dtype`, `dispose`, serialization, and board
  selection behavior.
- Normalize legacy `aries` to `aries-rb` and `regulus` to `regulus-ra`. Supported
  board identifiers are `aries-rb`, `regulus-ra`, and `regulus-rb`.
- Keep core-mode validation board-specific. Do not let an unsupported mode reach
  the native runtime when it can be rejected clearly in Python.
- Keep Hub artifact lookup deterministic. Callers supply the resolved artifact
  path; do not add Vision-specific folder or core-mode fallback policy here.
- Keep `ONNXBackend` independent of the MXQ runtime lifecycle. Import
  `onnxruntime` only from its lazy session-creation path and retain the clear
  `mblt-npu-python[onnxruntime]` installation error.

## Packaging and Tests

- Export every intended public backend from `mblt_npu.__init__` and declare any
  optional runtime in `pyproject.toml` extras.
- Do not require hardware, a model download, or ONNX Runtime for ordinary unit
  tests. Use injected runtime/session doubles for backend tests.
- Begin with focused tests, run Ruff on touched files, and run `git diff --check`.

## Documentation Synchronization

For a significant package change—public API, backend contract, supported board,
runtime dependency, artifact-resolution behavior, or package structure—update
this guide, `.agents/skills/mblt-npu/SKILL.md`, the Claude entry point when its
workflow changes, and the relevant README in the same change.
