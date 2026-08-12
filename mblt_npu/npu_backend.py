import logging
import os
from typing import Any, Dict, List, Literal, Optional, Sequence, Union

from huggingface_hub import HfApi, hf_hub_download
from huggingface_hub.errors import EntryNotFoundError
from qbruntime import Accelerator, Cluster, Core, CoreId, Model, ModelConfig

from .logging import log_model_details

logger = logging.getLogger(__name__)


def _enum_value(value: Any) -> int:
    """Return an integer enum value across qbruntime binding versions."""

    while hasattr(value, "value"):
        value = value.value
    return int(value)


# qbruntime has shipped both a Python enum wrapping a native enum and a direct
# enum binding. Peel every ``.value`` layer so serialized fields are stable
# integers across both forms (Cluster0 is 65536; Core0 is 1).
cluster_map = {
    _enum_value(cluster): cluster for cluster in (Cluster.Cluster0, Cluster.Cluster1)
}
core_map = {
    _enum_value(core): core for core in (Core.Core0, Core.Core1, Core.Core2, Core.Core3)
}


DEFAULT_TARGET_DEVICE = "aries-rb"
"""Default supported Mobilint NPU board."""

_TARGET_DEVICE_ALIASES = {
    # Configurations written before board-specific target devices were exposed
    # used these generic product names. Keep them readable, but always serialize
    # one of the supported board identifiers below.
    "aries": "aries-rb",
    "regulus": "regulus-ra",
}


def normalize_target_device(target_device: str) -> str:
    """Return a supported board identifier, accepting legacy generic product names."""

    if not isinstance(target_device, str):
        raise TypeError(
            f"target_device must be a string, got {type(target_device).__name__}."
        )
    return _TARGET_DEVICE_ALIASES.get(target_device.lower(), target_device.lower())


class MobilintNPUBackend:
    """Shared NPU access. Subclassed per product, because core allocation differs.

    The SDK exposes every core-mode setter on `ModelConfig` regardless of product
    and offers no capability query, so "does this device support global8" is only
    answerable by trying it and reading a load failure. Aries has two clusters of
    four cores and four modes; Regulus has one core and only single. Encoding that
    in types is what turns a late `StatusCode(16)` into an argument error.

    Instantiating this class directly selects a backend from ``target_device``.
    The default is ``aries-rb``; ``regulus-ra`` and ``regulus-rb`` select the
    Regulus implementation. Existing generic ``aries`` and ``regulus`` config
    values remain accepted as input compatibility aliases.
    """

    # Aries geometry. Overridden per product.
    num_of_clusters = 2
    num_of_cores_in_cluster = 4

    #: Core modes this product accepts. Empty on the base class, which never runs.
    supported_core_modes: tuple = ()
    #: Default board when this subclass is instantiated directly.
    default_target_device: str = DEFAULT_TARGET_DEVICE
    #: Board identifiers accepted by this backend implementation.
    supported_target_devices: tuple[str, ...] = ()

    def __new__(cls, *args, **kwargs):
        # MobilintNPUBackend(...) remains the public construction point. The
        # board identifier picks its implementation.
        if cls is MobilintNPUBackend:
            # ``target_device`` is the eighth positional parameter of
            # ``__init__`` as well as a keyword argument. Respect either form
            # before selecting the product-specific implementation.
            requested_device = kwargs.get("target_device")
            if requested_device is None and len(args) > 7:
                requested_device = args[7]
            requested_device = requested_device or DEFAULT_TARGET_DEVICE
            device = normalize_target_device(requested_device)
            cls = backend_class_for(device)
        return super().__new__(cls)

    def __init__(
        self,
        mxq_path: str = "",
        dev_no: int = 0,
        core_mode: Literal["auto", "single", "multi", "global4", "global8"] = "single",
        target_cores: Optional[List[Union[str, "CoreId"]]] = None,
        target_clusters: Optional[Sequence[Union[int, "Cluster"]]] = None,
        revision: Optional[str] = None,
        commit_hash: Optional[str] = None,
        target_device: str | None = None,
        **kwargs,
    ):
        resolved_target_device = normalize_target_device(
            target_device or self.default_target_device
        )
        if resolved_target_device not in self.supported_target_devices:
            raise ValueError(
                f"target_device {resolved_target_device!r} is not supported by "
                f"{type(self).__name__}; expected one of "
                f"{', '.join(self.supported_target_devices)}."
            )

        # ``from_dict`` supplies a serialized repository name through ``kwargs``.
        # Keep it so an MXQ path can still be resolved from the Hub after a
        # backend round trip; model mixins may replace it later when applicable.
        self.name_or_path: str = kwargs.get("name_or_path", "")
        self.target_device = resolved_target_device
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
                        revision=revision,
                    )
                except EntryNotFoundError:
                    cached = self._find_cached_mxq(name_or_path, mxq_path, revision)
                    if cached is not None:
                        return cached
                    mxq_candidate = self._find_mxq_from_hub(
                        name_or_path, mxq_path, revision
                    )
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
    def _find_cached_mxq(
        repo_id: str, mxq_path: str, revision: Optional[str] = None
    ) -> Optional[str]:
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
            snapshots = os.listdir(snapshots_dir)
            if revision is not None:
                snapshots = [snapshot for snapshot in snapshots if snapshot == revision]
            for snapshot in snapshots:
                snapshot_dir = os.path.join(snapshots_dir, snapshot)
                if not os.path.isdir(snapshot_dir):
                    continue
                for rel in rel_candidates:
                    candidate = os.path.join(snapshot_dir, rel)
                    if os.path.isfile(candidate):
                        return candidate
        except OSError:
            return None

        return None

    @staticmethod
    def _find_mxq_from_hub(
        repo_id: str, mxq_path: str, revision: Optional[str] = None
    ) -> Optional[str]:
        try:
            files = HfApi().list_repo_files(repo_id=repo_id, revision=revision)
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

    def __call__(self, x: Any) -> Any:
        """Run inference with the loaded MXQ model.

        This compatibility entry point is used by the Vision engine and mirrors
        the historical Model Zoo backend contract.
        """

        return self.mxq_model.infer(x)

    def get_dtype(self) -> str:
        """Return the loaded model input data type as a runtime string."""

        return str(self.mxq_model.get_model_input_data_type())

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
                if c_val in (0, 1):
                    cluster = (Cluster.Cluster0, Cluster.Cluster1)[c_val]
                    core = (Core.Core0, Core.Core1, Core.Core2, Core.Core3)[r_val]
                else:
                    cluster = cluster_map[c_val]
                    core = core_map[r_val]
                core_id_factory: Any = CoreId
                try:
                    # qbruntime's current binding exposes a no-argument
                    # constructor, although older type stubs declare only the
                    # legacy two-argument form.
                    core_id = core_id_factory()
                    core_id.cluster = cluster
                    core_id.core = core
                except TypeError:
                    core_id = CoreId(cluster, core)
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
                cluster_index = next(
                    index
                    for index, cluster in enumerate(
                        (Cluster.Cluster0, Cluster.Cluster1)
                    )
                    if _enum_value(cluster) == _enum_value(v.cluster)
                )
                core_index = next(
                    index
                    for index, core in enumerate(
                        (Core.Core0, Core.Core1, Core.Core2, Core.Core3)
                    )
                    if _enum_value(core) == _enum_value(v.core)
                )
                serialized.append(f"{cluster_index}:{core_index}")
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
                result.append(
                    (Cluster.Cluster0, Cluster.Cluster1)[c_val]
                    if c_val in (0, 1)
                    else cluster_map[c_val]
                )
            except Exception as e:
                raise ValueError(
                    f"cannot deserialize target cluster {s!r}: expected one of "
                    f"{sorted(cluster_map)}"
                ) from e
        return result

    @target_clusters.setter
    def target_clusters(self, values: Sequence[Union[int, "Cluster"]]):
        serialized = []
        for v in values:
            if isinstance(v, Cluster):
                serialized.append(
                    next(
                        index
                        for index, cluster in enumerate(
                            (Cluster.Cluster0, Cluster.Cluster1)
                        )
                        if _enum_value(cluster) == _enum_value(v)
                    )
                )
            elif isinstance(v, int):
                # Callers pass 0 and 1 meaning "first cluster", "second cluster".
                # Preserve this public serialized form while accepting native enum
                # values from older callers as well.
                ordinals = [Cluster.Cluster0, Cluster.Cluster1]
                if 0 <= v < len(ordinals):
                    serialized.append(v)
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
            "name_or_path": self.name_or_path,
            f"{p}mxq_path": self.mxq_path,
            f"{p}dev_no": self.dev_no,
            f"{p}core_mode": self.core_mode,
            f"{p}revision": self.revision,
            f"{p}commit_hash": self._commit_hash,
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
        a serialized board supported by that subclass is retained; a board for a
        different backend falls back to the subclass default.
        """
        p = prefix
        data = dict(data)
        if cls is not MobilintNPUBackend:
            serialized_target_device = data.get(f"{prefix}target_device")
            if serialized_target_device is None:
                data[f"{prefix}target_device"] = cls.default_target_device
            else:
                target_device = normalize_target_device(serialized_target_device)
                data[f"{prefix}target_device"] = (
                    target_device
                    if target_device in cls.supported_target_devices
                    else cls.default_target_device
                )
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
            target_device=data.pop(f"{p}target_device", None),
        )


class MobilintAriesBackend(MobilintNPUBackend):
    """Aries: two clusters of four cores, four allocation modes plus auto."""

    num_of_clusters = 2
    num_of_cores_in_cluster = 4
    supported_core_modes = ("auto", "single", "multi", "global4", "global8")
    default_target_device = "aries-rb"
    supported_target_devices = ("aries-rb",)

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
                mc.set_single_core_mode(core_ids=cores)
            else:
                # With no explicit cores, let qbruntime allocate one local core.
                mc.set_single_core_mode(1)
        elif self.core_mode == "multi":
            mc.set_multi_core_mode(self.target_clusters)
        elif self.core_mode == "global4":
            mc.set_global4_core_mode(self.target_clusters)
        elif self.core_mode == "global8":
            clusters = self.target_clusters
            expected_clusters = {
                _enum_value(Cluster.Cluster0),
                _enum_value(Cluster.Cluster1),
            }
            if (
                len(clusters) != len(expected_clusters)
                or {_enum_value(cluster) for cluster in clusters} != expected_clusters
            ):
                raise ValueError(
                    "global8 requires target_clusters to select both Aries clusters."
                )
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
    default_target_device = "regulus-ra"
    supported_target_devices = ("regulus-ra", "regulus-rb")

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        if self.target_clusters:
            raise ValueError(
                "target_clusters is meaningless on regulus, which has a single "
                f"core: got {self._target_clusters_serialized}. Remove it, or use "
                "target_device='aries-rb'."
            )
        cores = self.target_cores
        expected_cluster = _enum_value(Cluster.Cluster0)
        expected_core = _enum_value(Core.Core0)
        if len(cores) > 1 or any(
            _enum_value(core.cluster) != expected_cluster
            or _enum_value(core.core) != expected_core
            for core in cores
        ):
            raise ValueError(
                "target_cores on regulus may select only its sole core (0:0)."
            )

    def _configure_core_mode(self, mc: "ModelConfig") -> None:
        if self.core_mode == "auto":
            mc.set_auto_core_mode()
        else:
            cores = self.target_cores
            if cores:
                mc.set_single_core_mode(core_ids=cores)
            else:
                mc.set_single_core_mode(1)


#: target_device -> backend class. A mapping rather than a chain of ifs so that
#: adding a product is one entry and an unknown one is an error naming the
#: choices, instead of silently behaving like Aries.
BACKEND_CLASSES = {
    "aries-rb": MobilintAriesBackend,
    "regulus-ra": MobilintRegulusBackend,
    "regulus-rb": MobilintRegulusBackend,
}


def backend_class_for(target_device: str):
    normalized_target_device = normalize_target_device(target_device)
    try:
        return BACKEND_CLASSES[normalized_target_device]
    except KeyError:
        raise ValueError(
            f"unknown target_device {target_device!r}; "
            f"expected one of {', '.join(sorted(BACKEND_CLASSES))}"
        ) from None
