import logging
import os
from typing import Any, Dict, List, Literal, Optional, Union

from huggingface_hub import HfApi, hf_hub_download
from huggingface_hub.errors import EntryNotFoundError
from qbruntime import Accelerator, Cluster, Core, CoreId, Model, ModelConfig

from .logging import log_model_details

logger = logging.getLogger(__name__)

# Keyed by the enum's own `.value`, because that is exactly what the setters
# serialize (`f"{v.cluster.value}:{v.core.value}"`). The previous maps were keyed
# by ordinal — 0, 1, 2, 3 — and the values are not ordinals: Cluster0 is 65536 and
# Core0 is 1. So a Cluster round trip raised KeyError, which the getter's
# `except: pass` swallowed into an empty list, and a Core round trip silently
# returned the *next* core (core_map[1] is Core1, not Core0). Only callers who
# happened to pass plain ints hit a working path.
#
# Built from the enums rather than written out, so the two cannot drift again.
cluster_map = {c.value: c for c in (Cluster.Cluster0, Cluster.Cluster1)}
core_map = {c.value: c for c in (Core.Core0, Core.Core1, Core.Core2, Core.Core3)}


class MobilintNPUBackend:
    """Shared NPU access. Subclassed per product, because core allocation differs.

    The SDK exposes every core-mode setter on `ModelConfig` regardless of product
    and offers no capability query, so "does this device support global8" is only
    answerable by trying it and reading a load failure. Aries has two clusters of
    four cores and four modes; Regulus has one core and only single. Encoding that
    in types is what turns a late `StatusCode(16)` into an argument error.

    Instantiating this class directly still works and yields an Aries backend, so
    existing callers — `MobilintNPUBackend(...)` in mblt_melotts and
    `MobilintNPUBackend.from_dict(...)` in mblt_transformers — are unaffected.
    """

    # Aries geometry. Overridden per product.
    num_of_clusters = 2
    num_of_cores_in_cluster = 4

    #: Core modes this product accepts. Empty on the base class, which never runs.
    supported_core_modes: tuple = ()
    #: Value of `target_device` that selects this subclass.
    target_device: str = ""

    def __new__(cls, *args, **kwargs):
        # Dispatch, rather than making this an ABC, so that a bare
        # MobilintNPUBackend(...) keeps working for the callers that predate the
        # split into products. `target_device` picks the subclass; the default
        # preserves the previous behaviour exactly.
        if cls is MobilintNPUBackend:
            device = kwargs.get("target_device", "aries")
            cls = backend_class_for(device)
        return super().__new__(cls)

    def __init__(
        self,
        mxq_path: str = "",
        dev_no: int = 0,
        core_mode: Literal["auto", "single", "multi", "global4", "global8"] = "single",
        target_cores: Optional[List[Union[str, "CoreId"]]] = None,
        target_clusters: Optional[List[Union[int, "Cluster"]]] = None,
        revision: Optional[str] = None,
        commit_hash: Optional[str] = None,
        target_device: str = "aries",
        **kwargs,
    ):
        self.name_or_path: str = ""  # will be populated in MobilintModelMixin
        self.revision = revision
        self._commit_hash = commit_hash
        self.mxq_path = mxq_path
        self.dev_no = dev_no
        self.core_mode = core_mode
        if core_mode not in self.supported_core_modes:
            raise ValueError(
                f"core_mode {core_mode!r} is not available on "
                f"{self.target_device}; supported: "
                f"{', '.join(self.supported_core_modes)}. Refusing here rather than "
                "letting the SDK fail at Model::create, which is where an "
                "unsupported mode surfaces otherwise."
            )

        self._target_cores_serialized: List[str] = []
        self.target_cores = target_cores if target_cores is not None else []

        self._target_clusters_serialized: List[str] = []
        self.target_clusters = target_clusters if target_clusters is not None else []

    def check_model_path(self, mxq_path: str) -> str:
        # 1. current relative/absolute path
        if os.path.exists(mxq_path):
            return mxq_path

        # 2. inside the local path
        if os.path.isdir(self.name_or_path):
            local_path = os.path.join(self.name_or_path, mxq_path)
            if os.path.exists(local_path):
                return local_path

        # 3. If none of above, download mxq file from hub
        else:
            name_or_path = (
                self.name_or_path
                if self.name_or_path.startswith("mobilint/")
                else "mobilint/" + self.name_or_path
            )
            revision = (
                getattr(self, "revision", None)
                or getattr(self, "_commit_hash", None)
                or self._infer_hf_revision_from_cache(name_or_path)
            )
            try:
                return hf_hub_download(
                    repo_id=name_or_path,
                    filename=mxq_path,
                    revision=revision,
                )
            except EntryNotFoundError:
                try:
                    return hf_hub_download(
                        repo_id=name_or_path,
                        filename=mxq_path,
                    )
                except EntryNotFoundError:
                    cached = self._find_cached_mxq(name_or_path, mxq_path)
                    if cached is not None:
                        return cached
                    mxq_candidate = self._find_mxq_from_hub(name_or_path, mxq_path)
                    if mxq_candidate is None:
                        raise
                    return hf_hub_download(
                        repo_id=name_or_path,
                        filename=mxq_candidate,
                        revision=revision,
                    )

        raise Exception(f"[Mobilint] Error: Could not locate {mxq_path}.")

    @staticmethod
    def _infer_hf_revision_from_cache(repo_id: str) -> Optional[str]:
        if not repo_id or "/" not in repo_id:
            return None

        cache_root = os.getenv("HUGGINGFACE_HUB_CACHE") or os.getenv("HF_HUB_CACHE")
        if not cache_root:
            hf_home = os.getenv("HF_HOME") or os.path.join(
                os.path.expanduser("~"),
                ".cache",
                "huggingface",
            )
            cache_root = os.path.join(hf_home, "hub")

        repo_dir = os.path.join(cache_root, f"models--{repo_id.replace('/', '--')}")
        refs_dir = os.path.join(repo_dir, "refs")
        if os.path.isdir(refs_dir):
            for ref_name in ("main", "master"):
                ref_path = os.path.join(refs_dir, ref_name)
                if os.path.isfile(ref_path):
                    try:
                        with open(ref_path, "r", encoding="utf-8") as f:
                            ref = f.read().strip()
                        if ref:
                            return ref
                    except OSError:
                        pass
            try:
                for entry in os.listdir(refs_dir):
                    ref_path = os.path.join(refs_dir, entry)
                    if os.path.isfile(ref_path):
                        with open(ref_path, "r", encoding="utf-8") as f:
                            ref = f.read().strip()
                        if ref:
                            return ref
            except OSError:
                pass

        snapshots_dir = os.path.join(repo_dir, "snapshots")
        if os.path.isdir(snapshots_dir):
            try:
                for entry in os.listdir(snapshots_dir):
                    if os.path.isdir(os.path.join(snapshots_dir, entry)):
                        return entry
            except OSError:
                pass

        return None

    @staticmethod
    def _find_cached_mxq(repo_id: str, mxq_path: str) -> Optional[str]:
        if not repo_id or "/" not in repo_id:
            return None

        cache_root = os.getenv("HUGGINGFACE_HUB_CACHE") or os.getenv("HF_HUB_CACHE")
        if not cache_root:
            hf_home = os.getenv("HF_HOME") or os.path.join(
                os.path.expanduser("~"),
                ".cache",
                "huggingface",
            )
            cache_root = os.path.join(hf_home, "hub")

        repo_dir = os.path.join(cache_root, f"models--{repo_id.replace('/', '--')}")
        snapshots_dir = os.path.join(repo_dir, "snapshots")
        if not os.path.isdir(snapshots_dir):
            return None

        rel_candidates = [mxq_path, os.path.basename(mxq_path)]
        try:
            for snapshot in os.listdir(snapshots_dir):
                snapshot_dir = os.path.join(snapshots_dir, snapshot)
                if not os.path.isdir(snapshot_dir):
                    continue
                for rel in rel_candidates:
                    candidate = os.path.join(snapshot_dir, rel)
                    if os.path.isfile(candidate):
                        return candidate
        except OSError:
            return None

        # Last resort: find any mxq in snapshots
        for root, _, files in os.walk(snapshots_dir):
            for name in files:
                if name.endswith(".mxq"):
                    return os.path.join(root, name)

        return None

    @staticmethod
    def _find_mxq_from_hub(repo_id: str, mxq_path: str) -> Optional[str]:
        try:
            files = HfApi().list_repo_files(repo_id=repo_id)
        except Exception:
            return None

        basename = os.path.basename(mxq_path)
        if basename in files:
            return basename
        if mxq_path in files:
            return mxq_path

        raise ValueError(
            f"Cannot find {mxq_path} file from HuggingFace repo: f{repo_id}"
        )

    def _configure_core_mode(self, mc: "ModelConfig") -> None:
        raise NotImplementedError(
            "MobilintNPUBackend is a base class; use MobilintAriesBackend, "
            "MobilintRegulusBackend, or backend_class_for(target_device)."
        )

    def create(self):
        self.acc = Accelerator(self.dev_no)
        mc = ModelConfig()
        self._configure_core_mode(mc)

        model_path = self.check_model_path(self.mxq_path)
        self.mxq_model = Model(model_path, mc)
        log_model_details(model_path, self)

    def launch(self):
        self.mxq_model.launch(self.acc)

    def dispose(self):
        self.mxq_model.dispose()

    @property
    def target_cores(self) -> List["CoreId"]:
        result = []
        if not hasattr(self, "_target_cores_serialized"):
            return []

        for s in self._target_cores_serialized:
            try:
                c_val, r_val = map(int, s.split(":"))
                # CoreId takes no constructor arguments in the qbruntime binding;
                # the fields are assigned afterwards. Calling CoreId(cluster, core)
                # raised TypeError, which the except below swallowed — so a value
                # the setter had accepted came back as an empty list, with only a
                # log line to say so.
                core_id = CoreId()
                core_id.cluster = cluster_map[c_val]
                core_id.core = core_map[r_val]
                result.append(core_id)
            except Exception as e:
                # Raising rather than warning-and-skipping: this used to drop the
                # entry and return a shorter list, so a caller who asked for two
                # specific cores silently got none and ran on whatever the default
                # allocation gave them.
                raise ValueError(
                    f"cannot deserialize target core {s!r}: expected "
                    f'"<cluster value>:<core value>" with cluster in '
                    f"{sorted(cluster_map)} and core in {sorted(core_map)}"
                ) from e
        return result

    @target_cores.setter
    def target_cores(self, values: List[Union[str, "CoreId"]]):
        serialized = []
        for v in values:
            if isinstance(v, CoreId):
                serialized.append(f"{v.cluster.value}:{v.core.value}")
            elif isinstance(v, str):
                if ":" in v:
                    serialized.append(v)
                else:
                    raise ValueError(f"Invalid format: {v}")
            else:
                raise TypeError(f"Unsupported type: {type(v)}")

        self._target_cores_serialized = serialized

    @property
    def target_clusters(self) -> List["Cluster"]:
        result = []
        if not hasattr(self, "_target_clusters_serialized"):
            return []

        for s in self._target_clusters_serialized:
            try:
                c_val = int(s)
                result.append(cluster_map[c_val])
            except Exception as e:
                raise ValueError(
                    f"cannot deserialize target cluster {s!r}: expected one of "
                    f"{sorted(cluster_map)}"
                ) from e
        return result

    @target_clusters.setter
    def target_clusters(self, values: List[Union[int, "Cluster"]]):
        serialized = []
        for v in values:
            if isinstance(v, Cluster):
                serialized.append(v.value)
            elif isinstance(v, int):
                # Callers pass 0 and 1 meaning "first cluster", "second cluster" —
                # `set_multi_core_mode([0, 1])` in the vision wrapper's ancestor did
                # exactly that. Normalize to the enum value so both spellings
                # deserialize, instead of one of them working by accident.
                ordinals = [Cluster.Cluster0, Cluster.Cluster1]
                if 0 <= v < len(ordinals):
                    serialized.append(ordinals[v].value)
                elif v in cluster_map:
                    serialized.append(v)
                else:
                    raise ValueError(
                        f"cluster {v} is neither an index into "
                        f"{[c.name for c in ordinals]} nor one of {sorted(cluster_map)}"
                    )
            else:
                raise TypeError(f"Unsupported type: {type(v)}")

        self._target_clusters_serialized = serialized

    def to_dict(self, prefix="") -> Dict[str, Any]:
        p = prefix
        result = {
            f"{p}mxq_path": self.mxq_path,
            f"{p}dev_no": self.dev_no,
            f"{p}core_mode": self.core_mode,
            f"{p}target_device": self.target_device,
        }

        if self.core_mode == "single":
            result[f"{p}target_cores"] = self._target_cores_serialized
        else:
            result[f"{p}target_clusters"] = self._target_clusters_serialized

        return result

    @classmethod
    def from_dict(cls, data: Dict[str, Any], prefix: str = "") -> "MobilintNPUBackend":
        """Rebuilds a backend from a flat config dict.

        Called as `MobilintNPUBackend.from_dict(...)`, `cls` is the base and
        `__new__` picks the subclass from `target_device`. Called on a subclass,
        that subclass wins — a caller who asked for Regulus explicitly should not
        have a stale `target_device` in the dict silently override them.
        """
        p = prefix
        if cls is not MobilintNPUBackend:
            data = dict(data)
            data.pop(f"{prefix}target_device", None)
            data[f"{prefix}target_device"] = cls.target_device
        if f"{p}target_cores" in data.keys() and f"{p}target_clusters" in data.keys():
            logger.warning(f"{p}target_cores and {p}target_clusters are both set!")
            logger.warning(
                f"If {p}core_mode is `single`, only {p}target_cores will be used."
            )
            logger.warning(
                f"If {p}core_mode is `multi`, `global4`, or `global8`, only {p}target_clusters will be used."
            )

        return cls(
            name_or_path=data.pop("name_or_path", ""),
            mxq_path=data.pop(f"{p}mxq_path", ""),
            dev_no=data.pop(f"{p}dev_no", 0),
            core_mode=data.pop(f"{p}core_mode", "single"),
            target_cores=data.pop(f"{p}target_cores", None),
            target_clusters=data.pop(f"{p}target_clusters", None),
            revision=data.pop(f"{p}revision", None),
            commit_hash=data.pop(f"{p}commit_hash", None),
            # Default "aries" so a config written before products existed keeps
            # resolving to the backend it always got.
            target_device=data.pop(f"{p}target_device", "aries"),
        )


class MobilintAriesBackend(MobilintNPUBackend):
    """Aries: two clusters of four cores, four allocation modes plus auto."""

    num_of_clusters = 2
    num_of_cores_in_cluster = 4
    supported_core_modes = ("auto", "single", "multi", "global4", "global8")
    target_device = "aries"

    def _configure_core_mode(self, mc: "ModelConfig") -> None:
        if self.core_mode == "auto":
            # What an unconfigured ModelConfig already is. Named explicitly because
            # the previous code reached it by doing nothing under the label
            # "single", and measured, the default is CoreMode.Auto with
            # num_cores=0 — not single with all cores as its comment claimed.
            mc.set_auto_core_mode()
        elif self.core_mode == "single":
            cores = self.target_cores
            if cores:
                mc.set_single_core_mode(cores)
            else:
                # The two accepted signatures are (int) and (list[CoreId]); the
                # previous `set_single_core_mode(None, target_cores)` matched
                # neither and raised TypeError on every default-constructed
                # backend, since "single" is the default core_mode.
                mc.set_single_core_mode(1)
        elif self.core_mode == "multi":
            mc.set_multi_core_mode(self.target_clusters)
        elif self.core_mode == "global4":
            mc.set_global4_core_mode(self.target_clusters)
        elif self.core_mode == "global8":
            assert len(self.target_clusters) == 2, "global8 must contain every cores!"
            mc.set_global8_core_mode()
        else:  # unreachable: __init__ validates against supported_core_modes
            raise ValueError(f"unhandled core_mode {self.core_mode!r}")


class MobilintRegulusBackend(MobilintNPUBackend):
    """Regulus: one core, single only.

    Measured on a board rather than inferred: `global4` is rejected by qbruntime
    and `global8` fails Model::create with StatusCode(16), while `auto` and
    `single` both run. Cluster arguments have nothing to address, so they are
    refused instead of ignored — ignoring them would let a caller believe an
    allocation happened.
    """

    num_of_clusters = 1
    num_of_cores_in_cluster = 1
    supported_core_modes = ("auto", "single")
    target_device = "regulus"

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        if self.target_clusters:
            raise ValueError(
                "target_clusters is meaningless on regulus, which has a single "
                f"core: got {self._target_clusters_serialized}. Remove it, or use "
                "target_device='aries'."
            )

    def _configure_core_mode(self, mc: "ModelConfig") -> None:
        if self.core_mode == "auto":
            mc.set_auto_core_mode()
        else:
            cores = self.target_cores
            mc.set_single_core_mode(cores if cores else 1)


#: target_device -> backend class. A mapping rather than a chain of ifs so that
#: adding a product is one entry and an unknown one is an error naming the
#: choices, instead of silently behaving like Aries.
BACKEND_CLASSES = {
    MobilintAriesBackend.target_device: MobilintAriesBackend,
    MobilintRegulusBackend.target_device: MobilintRegulusBackend,
}


def backend_class_for(target_device: str):
    try:
        return BACKEND_CLASSES[target_device]
    except KeyError:
        raise ValueError(
            f"unknown target_device {target_device!r}; "
            f"expected one of {', '.join(sorted(BACKEND_CLASSES))}"
        ) from None
