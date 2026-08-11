"""Tests for board-specific NPU backend selection."""

from __future__ import annotations

from typing import Any, cast

import pytest

from mblt_npu import MobilintAriesBackend, MobilintNPUBackend, MobilintRegulusBackend


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
