# Mobilint NPU Python

Shared runtime support for applications that run MXQ models on Mobilint NPUs.
`mblt-npu-python` provides the common backend, device-selection rules, Hugging Face
artifact resolution, and model-detail logging used by Mobilint Python packages. It
is a library dependency, rather than an end-user model catalog.

Version `0.0.1` is the initial standalone release.

`logging` ships here rather than with its only caller because the two are mutually
dependent — `npu_backend` imports `log_model_details`, and `log_model_details` reads
a `MobilintNPUBackend`'s fields.

## Installation

```bash
pip install mblt-npu-python
```

This package requires a supported Linux environment with
[`mobilint-qb-runtime`](https://pypi.org/project/mobilint-qb-runtime/) available.

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
`target_device` (default: `"aries"`). Use `MobilintAriesBackend` or
`MobilintRegulusBackend` when an application needs to name a product explicitly.
`backend_class_for()` and `BACKEND_CLASSES` are available for integrations that
need to inspect the supported targets.

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
