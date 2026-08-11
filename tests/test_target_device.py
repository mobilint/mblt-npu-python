"""Tests for board-specific NPU backend selection."""

from __future__ import annotations

from pathlib import Path
from typing import Any, cast

import pytest

from mblt_npu import MobilintAriesBackend, MobilintNPUBackend, MobilintRegulusBackend
import mblt_npu.npu_backend as npu_backend


@pytest.mark.parametrize(
    ("target_device", "backend_class"),
    [
        ("aries-rb", MobilintAriesBackend),
        ("regulus-ra", MobilintRegulusBackend),
        ("regulus-rb", MobilintRegulusBackend),
    ],
)
def test_target_device_selects_the_product_backend(
    target_device: str, backend_class: type[MobilintNPUBackend]
) -> None:
    """Select an implementation from each supported board identifier."""

    backend = MobilintNPUBackend(target_device=target_device)

    assert isinstance(backend, backend_class)
    assert backend.target_device == target_device
    assert backend.to_dict()["target_device"] == target_device


def test_default_target_device_is_aries_rb() -> None:
    """Use Aries RB when callers do not declare a board."""

    backend = MobilintNPUBackend()

    assert isinstance(backend, MobilintAriesBackend)
    assert backend.target_device == "aries-rb"


@pytest.mark.parametrize("target_device", ["regulus-ra", "regulus-rb"])
def test_positional_target_device_selects_the_product_backend(
    target_device: str,
) -> None:
    """Dispatch on ``target_device`` when callers pass it positionally."""

    backend = MobilintNPUBackend("", 0, "single", None, None, None, None, target_device)

    assert isinstance(backend, MobilintRegulusBackend)
    assert backend.target_device == target_device


def test_regulus_subclass_round_trip_preserves_regulus_rb() -> None:
    """Keep a supported Regulus board variant during concrete deserialization."""

    original = MobilintRegulusBackend(target_device="regulus-rb")
    restored = MobilintRegulusBackend.from_dict(original.to_dict())

    assert restored.target_device == "regulus-rb"


def test_from_dict_does_not_consume_callers_configuration() -> None:
    """Leave reusable serialized backend configuration untouched."""

    configuration = {
        "mxq_path": "model.mxq",
        "core_mode": "single",
        "target_device": "regulus-rb",
    }
    original_configuration = dict(configuration)

    backend = MobilintNPUBackend.from_dict(configuration)

    assert backend.target_device == "regulus-rb"
    assert configuration == original_configuration


@pytest.mark.parametrize("target_clusters", [[0, 0], [1, 1], [0]])
def test_global8_requires_both_distinct_aries_clusters(
    target_clusters: list[int],
) -> None:
    """Reject incomplete or duplicate cluster selections before runtime setup."""

    class _ModelConfig:
        def set_global8_core_mode(self) -> None:
            pytest.fail("global8 must not configure an invalid cluster selection")

    backend = MobilintAriesBackend(core_mode="global8", target_clusters=target_clusters)

    with pytest.raises(ValueError, match="both Aries clusters"):
        backend._configure_core_mode(_ModelConfig())


@pytest.mark.parametrize(
    ("legacy_target_device", "canonical_target_device", "backend_class"),
    [
        ("aries", "aries-rb", MobilintAriesBackend),
        ("regulus", "regulus-ra", MobilintRegulusBackend),
    ],
)
def test_legacy_target_device_values_remain_readable(
    legacy_target_device: str,
    canonical_target_device: str,
    backend_class: type[MobilintNPUBackend],
) -> None:
    """Read old generic product values but serialize a canonical board name."""

    backend = MobilintNPUBackend.from_dict({"target_device": legacy_target_device})

    assert isinstance(backend, backend_class)
    assert backend.target_device == canonical_target_device


def test_legacy_core_and_cluster_assignments_round_trip() -> None:
    """Keep Model Zoo's ordinal core and cluster configuration values usable."""

    single_core = MobilintNPUBackend(target_cores=["0:0"])
    cluster = MobilintNPUBackend(core_mode="global4", target_clusters=[0])

    assert len(single_core.target_cores) == 1
    assert single_core.to_dict()["target_cores"] == ["0:0"]
    assert len(cluster.target_clusters) == 1
    assert cluster.to_dict()["target_clusters"] == [0]


def test_target_cores_supports_no_argument_core_id_bindings(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Assign CoreId fields for qbruntime bindings without constructor arguments."""

    class _CoreId:
        def __init__(self) -> None:
            self.cluster: object | None = None
            self.core: object | None = None

    monkeypatch.setattr(npu_backend, "CoreId", _CoreId)
    backend = MobilintNPUBackend(target_cores=["0:0"])

    target_core = backend.target_cores[0]

    assert target_core.cluster == npu_backend.Cluster.Cluster0
    assert target_core.core == npu_backend.Core.Core0


def test_hub_retry_keeps_an_explicit_revision(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Never replace a pinned Hub artifact with one from the default branch."""

    calls: list[dict[str, object]] = []

    def _download(**kwargs: object) -> str:
        calls.append(kwargs)
        if len(calls) == 1:
            raise npu_backend.EntryNotFoundError("missing from pinned revision")
        return "/tmp/model.mxq"

    monkeypatch.setattr(npu_backend, "hf_hub_download", _download)
    backend = MobilintNPUBackend(mxq_path="model.mxq", revision="pinned-commit")
    backend.name_or_path = "mobilint/example"

    assert backend.check_model_path("model.mxq") == "/tmp/model.mxq"
    assert [call["revision"] for call in calls] == ["pinned-commit", "pinned-commit"]


def test_from_dict_preserves_name_or_path_for_hub_resolution(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Use the serialized repository name when an MXQ must be downloaded."""

    download_call: dict[str, object] = {}

    def _download(**kwargs: object) -> str:
        download_call.update(kwargs)
        return "/tmp/model.mxq"

    monkeypatch.setattr(npu_backend, "hf_hub_download", _download)
    backend = MobilintNPUBackend.from_dict(
        {"name_or_path": "example", "mxq_path": "model.mxq"}
    )

    assert backend.name_or_path == "example"
    assert backend.check_model_path("model.mxq") == "/tmp/model.mxq"
    assert download_call["repo_id"] == "mobilint/example"


def test_cached_mxq_lookup_rejects_unrelated_artifacts(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """Never substitute an arbitrary cached MXQ for a missing requested file."""

    cache_root = tmp_path / "hub"
    unrelated = (
        cache_root
        / "models--mobilint--example"
        / "snapshots"
        / "commit"
        / "other-model.mxq"
    )
    unrelated.parent.mkdir(parents=True)
    unrelated.touch()
    monkeypatch.setenv("HUGGINGFACE_HUB_CACHE", str(cache_root))

    assert (
        MobilintNPUBackend._find_cached_mxq("mobilint/example", "requested.mxq") is None
    )


def test_backend_exposes_vision_runtime_compatibility_methods() -> None:
    """Keep the callable inference and input-dtype APIs used by Vision."""

    class _Model:
        def infer(self, value: object) -> object:
            return ("output", value)

        def get_model_input_data_type(self) -> str:
            return "DataType.Uint8"

    backend = MobilintNPUBackend()
    backend.mxq_model = cast(Any, _Model())

    assert backend("input") == ("output", "input")
    assert backend.get_dtype() == "DataType.Uint8"
