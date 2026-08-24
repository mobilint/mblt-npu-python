# Mobilint NPU Python

<!-- markdownlint-disable MD033 -->
<div align="center">
<p>
<a href="https://www.mobilint.com/" target="_blank">
<img src="https://raw.githubusercontent.com/mobilint/.github/main/assets/Mobilint_Logo_Primary.png" alt="Mobilint Logo" width="60%">
</a>
</p>
</div>
<!-- markdownlint-enable MD033 -->

Shared runtime support for applications that run MXQ models on Mobilint NPUs or ONNX models through ONNX Runtime.
`mblt-npu-python` provides the common backend, device-selection rules, Hugging Face
artifact resolution, and model-detail logging used by Mobilint Python packages. It
is a library dependency, rather than an end-user model catalog.

Version `0.0.0` is the initial standalone release.

`logging` ships here rather than with its only caller because the two are mutually
dependent — `npu_backend` imports `log_model_details`, and `log_model_details` reads
a `MobilintNPUBackend`'s fields.

## Installation

[![PyPI - Version](https://img.shields.io/pypi/v/mblt-npu-python?logo=pypi&logoColor=white)](https://pypi.org/project/mblt-npu-python/)
[![PyPI Downloads](https://static.pepy.tech/badge/mblt-npu-python?period=total&units=INTERNATIONAL_SYSTEM&left_color=BLACK&right_color=GREEN&left_text=downloads)](https://clickpy.clickhouse.com/dashboard/mblt-npu-python)
[![PyPI - Python Version](https://img.shields.io/pypi/pyversions/mblt-npu-python?logo=python&logoColor=gold)](https://pypi.org/project/mblt-npu-python/)

```bash
pip install mblt-npu-python
```

This package requires a supported Linux environment with
[`mobilint-qb-runtime`](https://pypi.org/project/mobilint-qb-runtime/) available
and Python 3.10 through 3.12.

## Public API

Import the backend from `mblt_npu`:

```python
from mblt_npu import MobilintNPUBackend

backend = MobilintNPUBackend(
    mxq_path="model.mxq",
    core_mode="single",
)
backend.create()
try:
    backend.launch()
    outputs = backend.mxq_model.infer([input_tensor])
finally:
    backend.dispose()
```

`MobilintNPUBackend` selects the appropriate implementation from
`target_device` (default: `"aries-rb"`). `"aries-rb"` selects
`MobilintAriesBackend`; `"regulus-ra"` and `"regulus-rb"` select
`MobilintRegulusBackend`. The former generic values `"aries"` and `"regulus"`
remain accepted when loading older configurations.
`backend_class_for()` and `BACKEND_CLASSES` are available for integrations that
need to inspect the supported targets.

### Multi-slot MXQ execution

`max_batch_size` is aggregate capacity. At `create()`, the backend probes the
compiled per-model capacity `K` and loads `ceil(max_batch_size / K)` model slots.
Slots are distributed round-robin over the devices named by canonical target
strings and reuse one accelerator per device. `mxq_model` and `acc` continue to
refer to slot zero for compatibility; concurrent callers can use
`infer_slot(slot_index, inputs)`. Allocation failures dispose all created slots
and raise `MobilintBackendAllocError` with the failed slot and device.

Hub-backed configurations retain `name_or_path`, `revision`, and `commit_hash`
through `to_dict()` / `from_dict()`. Artifact lookup never substitutes an
unpinned revision or an unrelated cached MXQ.

For ONNX inference, install the optional runtime extra and use `ONNXBackend`:

```bash
pip install "mblt-npu-python[onnxruntime]"
```

```python
from mblt_npu import ONNXBackend

backend = ONNXBackend("model.onnx")
backend.create()
outputs = backend({"images": input_array})
backend.dispose()
```

`ONNXBackend` imports `onnxruntime` only when it creates a session.

Most users should access the backend through a model package such as
[`mblt-vision-python`](https://github.com/mobilint/mblt-vision-python), which owns
model configuration, preprocessing, and postprocessing.

## Testing helpers

The optional `test` extra provides a shared pytest plugin with NPU options and the
`npu_params` fixture used by Mobilint package test suites:

```bash
pip install "mblt-npu-python[test]"
```

Import `mblt_npu.pytest_plugin` from a repository's root `tests/conftest.py` to
register its options. The plugin is intentionally not auto-registered, so projects
control when those command-line options are exposed.

## Support and issues

For installation, runtime, or integration support, visit the
[Mobilint forum](https://discuss.mobilint.com/). Report reproducible package issues in the
[mblt-npu-python issue tracker](https://github.com/mobilint/mblt-npu-python/issues).

## License

Distributed under the [BSD 3-Clause License](LICENSE).
