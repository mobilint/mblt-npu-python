---
name: mblt-npu
description: >-
  Work on the shared Mobilint NPU Python runtime, including MXQ backend lifecycle,
  board-specific target selection, Hub MXQ resolution, optional ONNX Runtime inference,
  logging, and pytest helpers. Use for changes in mblt-npu-python.
---

# Mobilint NPU Python

1. Read `AGENTS.md`, run `git status --short`, and inspect `pyproject.toml`, public exports,
   the affected backend, and focused tests before editing.
   Keep the Python 3.10 minimum and supported-version classifiers aligned with language syntax.
2. Preserve the `MobilintNPUBackend` compatibility contract: lifecycle methods, callable MXQ
   inference, input dtype inspection, serialization, and board-selected subclasses.
3. Normalize legacy `aries`/`regulus` target values to board identifiers. Keep accepted boards and
   core modes explicit; reject unsupported combinations before calling qbruntime.
4. Keep Vision-specific artifact folder policy out of this package. This package resolves an MXQ
   path; Vision chooses its board-specific Hub artifact path.
5. Keep `ONNXBackend` optional and lazy. It must work with injected ONNX Runtime doubles in unit
   tests and raise the documented extra-installation error only when a real session is requested.
6. Export public backends from `mblt_npu`, update optional extras with runtime changes, and avoid
   making hardware, native bindings, downloads, or ONNX Runtime mandatory for normal tests.
7. For significant changes to APIs, backends, targets, dependencies, or artifact behavior, update
   `AGENTS.md`, this skill, the Claude entry point when its workflow changes, and the README in the
   same change. Run focused tests, Ruff, and `git diff --check`.
