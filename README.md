# mblt-npu-python

**Shared NPU access** for Mobilint NPUs.

`MobilintNPUBackend` and the model-detail logging that the other Mobilint Python
packages need. It is a separate package for one reason: the alternative was a copy
of `npu_backend.py` in each of `mblt-vision-python`, `mblt-transformers-python` and
`mblt-MeloTTS-python`, and nothing would have kept the three in sync.

`logging` ships here rather than with its only caller because the two are mutually
dependent — `npu_backend` imports `log_model_details`, and `log_model_details` reads
a `MobilintNPUBackend`'s fields.

## Installation

```bash
pip install mblt-npu-python
```

## License

BSD-3-Clause.
