"""PRTReID setup, weight management, and feature extraction."""

from __future__ import annotations

import hashlib
import logging
import os
import sys
import types
import urllib.request
from importlib.machinery import ModuleSpec
from pathlib import Path
from typing import Any, Optional, Union

import numpy as np
import torch

from .config import PlayerTeamClassifierConfig

log = logging.getLogger(__name__)
_FORCE_TORCH_LOAD_CPU = False

PRTREID_WEIGHTS_URL = (
    "https://zenodo.org/records/10653453/files/"
    "prtreid-soccernet-baseline.pth.tar?download=1"
)
PRTREID_WEIGHTS_MD5 = "9633825232bc89f23a94522c5561650e"

HRNET_PRETRAINED_URL = (
    "https://zenodo.org/records/10604211/files/"
    "hrnetv2_w32_imagenet_pretrained.pth?download=1"
)
HRNET_PRETRAINED_MD5 = "58ea12b0420aa3adaa2f74114c9f9721"


class MissingDependencyError(ImportError):
    """Raised when an optional inference dependency is not installed."""


def require_dependency(import_name: str, install_hint: str) -> None:
    try:
        __import__(import_name)
    except ModuleNotFoundError as exc:
        missing_name = exc.name or import_name
        if missing_name != import_name:
            raise MissingDependencyError(
                f"Dependency '{import_name}' could not import because "
                f"'{missing_name}' is missing. Install it with: pip install {missing_name}"
            ) from exc
        raise MissingDependencyError(
            f"Missing dependency '{import_name}'. Install it with: {install_hint}"
        ) from exc


def _add_local_prtreid_source(weights_path: Union[str, Path]) -> None:
    """Use a local PRTReID source checkout when the wheel cannot be built."""

    source_dir = Path(weights_path).resolve().parent / "prtreid_src"
    if source_dir.exists() and str(source_dir) not in sys.path:
        sys.path.insert(0, str(source_dir))


def _md5(path: Union[str, Path]) -> str:
    digest = hashlib.md5()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _download(url: str, dest: Union[str, Path], md5: Optional[str] = None) -> None:
    destination = Path(dest)
    if destination.is_file():
        if md5 is None or _md5(destination) == md5:
            log.info("Already downloaded: %s", destination)
            return
        log.warning("Checksum mismatch for %s; downloading again.", destination)

    destination.parent.mkdir(parents=True, exist_ok=True)
    try:
        urllib.request.urlretrieve(url, destination)
    except Exception as exc:  # pragma: no cover - depends on network availability
        raise RuntimeError(
            f"Could not download {url}. Place the file manually at {destination}."
        ) from exc

    if md5 and _md5(destination) != md5:
        raise RuntimeError(f"MD5 mismatch for downloaded file: {destination}")


def _patch_torch_load(force_cpu: bool = False) -> None:
    """Allow legacy PRTReID checkpoints with PyTorch >= 2.6."""

    global _FORCE_TORCH_LOAD_CPU
    _FORCE_TORCH_LOAD_CPU = _FORCE_TORCH_LOAD_CPU or force_cpu

    original_load = torch.load
    if getattr(original_load, "_patched_prtreid", False):
        return

    def safe_load(*args: Any, **kwargs: Any) -> Any:
        kwargs.setdefault("weights_only", False)
        if _FORCE_TORCH_LOAD_CPU or torch.cuda.device_count() == 0:
            if len(args) >= 2:
                args_list = list(args)
                if args_list[1] is None:
                    args_list[1] = "cpu"
                    args = tuple(args_list)
            elif kwargs.get("map_location") is None:
                kwargs["map_location"] = "cpu"
        return original_load(*args, **kwargs)

    safe_load._patched_prtreid = True  # type: ignore[attr-defined]
    torch.load = safe_load  # type: ignore[assignment]


def _prepare_wandb_for_prtreid() -> None:
    """Disable wandb or provide a tiny stub if local wandb is broken."""

    os.environ.setdefault("WANDB_DISABLED", "true")
    os.environ.setdefault("WANDB_MODE", "disabled")

    try:
        import wandb  # noqa: F401

        return
    except Exception as exc:
        log.warning("wandb import failed (%s). Using a lightweight stub.", exc)

    stub = types.ModuleType("wandb")

    def noop(*args: Any, **kwargs: Any) -> None:
        return None

    class DummyRun:
        config: dict[str, Any] = {}
        name = "disabled"

    stub.init = lambda *args, **kwargs: DummyRun()  # type: ignore[attr-defined]
    stub.log = noop  # type: ignore[attr-defined]
    stub.finish = noop  # type: ignore[attr-defined]
    stub.watch = noop  # type: ignore[attr-defined]
    stub.define_metric = noop  # type: ignore[attr-defined]
    stub.Image = lambda *args, **kwargs: None  # type: ignore[attr-defined]
    stub.run = None  # type: ignore[attr-defined]
    stub.sdk = types.SimpleNamespace()  # type: ignore[attr-defined]
    stub.__spec__ = ModuleSpec(name="wandb", loader=None)
    stub.__package__ = "wandb"
    sys.modules["wandb"] = stub


def _build_prtreid_cfg(
    weights_path: Union[str, Path],
    hrnet_pretrained_dir: Union[str, Path],
    save_dir: Union[str, Path],
    force_cpu_load: bool = False,
) -> Any:
    """Build the minimal PRTReID config needed by FeatureExtractor."""

    _add_local_prtreid_source(weights_path)
    require_dependency("yacs", "pip install yacs")
    require_dependency(
        "prtreid",
        'pip install "prtreid @ git+https://github.com/VlSomers/prtreid"',
    )
    require_dependency("albumentations", 'pip install "albumentations<2.0"')

    _patch_torch_load(force_cpu=force_cpu_load)
    _prepare_wandb_for_prtreid()

    from prtreid.scripts.default_config import get_default_config
    from prtreid.scripts.main import build_config

    cfg = get_default_config()

    cfg.project.name = "PlayerRoleTeamClassifier"
    cfg.project.logger.use_tensorboard = False
    cfg.project.logger.use_wandb = False

    cfg.model.name = "bpbreid"
    cfg.model.pretrained = True
    cfg.model.load_weights = str(weights_path)
    cfg.model.load_config = True
    cfg.model.save_model_flag = False

    cfg.model.bpbreid.pooling = "gwap"
    cfg.model.bpbreid.normalization = "identity"
    cfg.model.bpbreid.mask_filtering_training = False
    cfg.model.bpbreid.mask_filtering_testing = False
    cfg.model.bpbreid.training_binary_visibility_score = True
    cfg.model.bpbreid.testing_binary_visibility_score = True
    cfg.model.bpbreid.last_stride = 1
    cfg.model.bpbreid.learnable_attention_enabled = False
    cfg.model.bpbreid.dim_reduce = "after_pooling"
    cfg.model.bpbreid.dim_reduce_output = 256
    cfg.model.bpbreid.backbone = "hrnet32"
    cfg.model.bpbreid.test_embeddings = ["globl"]
    cfg.model.bpbreid.test_use_target_segmentation = "none"
    cfg.model.bpbreid.shared_parts_id_classifier = False
    cfg.model.bpbreid.hrnet_pretrained_path = str(hrnet_pretrained_dir)
    cfg.model.bpbreid.masks.type = "disk"
    cfg.model.bpbreid.masks.preprocess = "id"

    cfg.data.height = 256
    cfg.data.width = 128
    cfg.data.save_dir = str(save_dir)
    cfg.data.sources = ["market1501"]
    cfg.data.targets = ["market1501"]

    cfg.loss.name = "part_based"
    cfg.loss.part_based.name = "part_averaged_triplet_loss"
    cfg.loss.part_based.ppl = "cl"

    cfg.test.evaluate = True
    cfg.test.normalize_feature = True
    cfg.test.dist_metric = "euclidean"
    cfg.use_gpu = torch.cuda.is_available()

    return build_config(config=cfg)


def as_numpy_embedding(value: Any) -> np.ndarray:
    if isinstance(value, torch.Tensor):
        array = value.detach().cpu().numpy()
    else:
        array = np.asarray(value)
    if array.ndim > 2:
        array = array.reshape(array.shape[0], -1)
    return array


class PRTReIDBackend:
    """Lazy PRTReID FeatureExtractor wrapper."""

    def __init__(self, config: PlayerTeamClassifierConfig):
        self.config = config
        self.config.reid_weights_dir.mkdir(parents=True, exist_ok=True)
        self.prtreid_ckpt = (
            self.config.reid_weights_dir / "prtreid-soccernet-baseline.pth.tar"
        )
        self.hrnet_pretrained = (
            self.config.reid_weights_dir / "hrnetv2_w32_imagenet_pretrained.pth"
        )
        self.cfg: Any = None
        self._feature_extractor: Any = None
        self.test_embeddings: list[str] = ["globl"]
        self.ensure_weights()

    def ensure_weights(self) -> None:
        if self.config.download_reid_weights:
            _download(PRTREID_WEIGHTS_URL, self.prtreid_ckpt, PRTREID_WEIGHTS_MD5)
            _download(HRNET_PRETRAINED_URL, self.hrnet_pretrained, HRNET_PRETRAINED_MD5)
            return

        missing = [
            str(path)
            for path in (self.prtreid_ckpt, self.hrnet_pretrained)
            if not path.exists()
        ]
        if missing:
            raise FileNotFoundError(
                "Missing ReID weights and download_reid_weights=False: "
                + ", ".join(missing)
            )

    def get_extractor(self) -> Any:
        if self._feature_extractor is None:
            self.cfg = _build_prtreid_cfg(
                weights_path=self.prtreid_ckpt,
                hrnet_pretrained_dir=self.config.reid_weights_dir,
                save_dir=self.config.reid_weights_dir / "logs",
                force_cpu_load=str(self.config.device).lower().startswith("cpu"),
            )
            self.test_embeddings = list(self.cfg.model.bpbreid.test_embeddings)
            from prtreid.tools.feature_extractor import FeatureExtractor

            self._feature_extractor = FeatureExtractor(
                self.cfg,
                model_path=self.cfg.model.load_weights,
                device=self.config.device,
                image_size=(self.cfg.data.height, self.cfg.data.width),
                verbose=False,
            )
            log.info("PRTReID FeatureExtractor ready")
        return self._feature_extractor

    def extract_features(
        self,
        crops: list[np.ndarray],
    ) -> tuple[np.ndarray, Optional[np.ndarray]]:
        extractor = self.get_extractor()
        from prtreid.utils.tools import extract_test_embeddings

        embedding_chunks: list[np.ndarray] = []
        role_chunks: list[np.ndarray] = []
        batch_size = max(1, int(self.config.reid_batch_size))

        for start in range(0, len(crops), batch_size):
            batch = crops[start : start + batch_size]
            reid_output = extractor(batch, external_parts_masks=None)
            embeddings, _, _, _, role_cls_scores = extract_test_embeddings(
                reid_output,
                self.test_embeddings,
            )
            embedding_chunks.append(as_numpy_embedding(embeddings))

            if role_cls_scores is not None:
                scores = (
                    role_cls_scores.get("globl")
                    if isinstance(role_cls_scores, dict)
                    else role_cls_scores
                )
                if scores is not None:
                    role_chunks.append(as_numpy_embedding(scores))

        embeddings_np = np.concatenate(embedding_chunks, axis=0)
        role_scores_np = np.concatenate(role_chunks, axis=0) if role_chunks else None
        return embeddings_np, role_scores_np
