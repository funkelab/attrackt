import logging
import os
from collections import defaultdict
from pathlib import Path
from typing import Callable, Dict, List, Literal

import numpy as np
import tifffile
import torch
import yaml
from tqdm import tqdm

from ..data import build_windows, get_features, load_tiff_timeseries
from ..tracking import TrackGraph, build_graph, track_greedy
from ..utils import normalize
from .model import TrackingTransformer
from .predict import predict_windows
from .pretrained import download_pretrained

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class Trackastra:
    def __init__(
        self,
        transformer: TrackingTransformer,
        train_args: dict = None,
        device: Literal["cuda", "mps", "cpu", "automatic", None] = None,
    ):
        if device == "cuda":
            if torch.cuda.is_available():
                self.device = "cuda"
            else:
                logger.info("Cuda not available, falling back to cpu.")
                self.device = "cpu"
        elif device == "mps":
            if (
                torch.backends.mps.is_available()
                and os.getenv("PYTORCH_ENABLE_MPS_FALLBACK") is not None
                and os.getenv("PYTORCH_ENABLE_MPS_FALLBACK") != "0"
            ):
                self.device = "mps"
            else:
                logger.info("Mps not available, falling back to cpu.")
                self.device = "cpu"
        elif device == "cpu":
            self.device = "cpu"
        elif device == "automatic" or device is None:
            should_use_mps = (
                torch.backends.mps.is_available()
                and os.getenv("PYTORCH_ENABLE_MPS_FALLBACK") is not None
                and os.getenv("PYTORCH_ENABLE_MPS_FALLBACK") != "0"
            )
            self.device = (
                "cuda"
                if torch.cuda.is_available()
                else (
                    "mps"
                    if should_use_mps and os.getenv("PYTORCH_ENABLE_MPS_FALLBACK")
                    else "cpu"
                )
            )
        else:
            raise ValueError(f"Device {device} not recognized.")

        logger.info(f"Using device {self.device}")

        self.transformer = transformer.to(self.device)
        self.train_args = train_args

    @classmethod
    def from_folder(cls, dir: Path, device: str | None = None):
        transformer = TrackingTransformer.from_folder(dir, map_location="cpu")
        train_args = yaml.load(open(dir / "train_config.yaml"), Loader=yaml.FullLoader)
        return cls(transformer=transformer, train_args=train_args, device=device)

    @classmethod
    def from_pretrained(
        cls, name: str, device: str | None = None, download_dir: Path | None = None
    ):
        folder = download_pretrained(name, download_dir)
        return cls.from_folder(folder, device=device)

    def _predict(
        self,
        imgs: np.ndarray,
        masks: np.ndarray,
        feature_type: Literal["none", "wrfeat"],
        edge_threshold: float,
        n_workers: int = 0,
        progbar_class=tqdm,
    ):
        logger.info("Predicting weights for candidate graph")
        imgs = normalize(imgs)
        self.transformer.eval()
        logger.info(
            f"ndim or spatial dims being used in _predict is {self.transformer.config['coord_dim']}."
        )

        features = get_features(
            detections=masks,
            imgs=imgs,
            features=feature_type,
            ndim=self.transformer.config["coord_dim"],
            n_workers=n_workers,
            progbar_class=progbar_class,
        )

        logger.info("Building windows")
        windows = build_windows(
            features,
            window_size=self.transformer.config["window"],
            progbar_class=progbar_class,
        )

        logger.info("Predicting windows")
        predictions = predict_windows(
            windows=windows,
            features=features,
            model=self.transformer,
            edge_threshold=edge_threshold,
            spatial_dim=masks.ndim - 1,
            progbar_class=progbar_class,
            intra_window_weight=0,
            delta_t=1,
            window=self.transformer.config["window"],
        )

        return predictions

    def _track_from_predictions(
        self,
        predictions,
        mode: Literal["greedy_nodiv", "greedy", "ilp"] = "ilp",
        use_distance: bool = False,
        max_distance: int = 256,
        max_neighbors: int = 10,
        delta_t: int = 1,
        **kwargs,
    ):
        logger.info("Running greedy tracker")
        nodes = predictions["nodes"]
        weights = predictions["weights"]

        candidate_graph = build_graph(
            nodes=nodes,
            weights=weights,
            use_distance=use_distance,
            max_distance=max_distance,
            max_neighbors=max_neighbors,
            delta_t=delta_t,
        )
        if mode == "greedy":
            return track_greedy(candidate_graph)
        elif mode == "greedy_nodiv":
            return track_greedy(candidate_graph, allow_divisions=False)
        elif mode == "ilp":
            from trackastra.tracking.ilp import track_ilp

            return track_ilp(candidate_graph, ilp_config="gt", **kwargs)
        else:
            raise ValueError(f"Tracking mode {mode} does not exist.")

    def average_weights(self, predictions_list: List[Dict]) -> Dict[str, List]:
        weight_accumulator = defaultdict(list)

        for prediction in predictions_list:
            for (a, b), score in prediction["weights"]:
                key = (a, b)  # Ensure edge symmetry if applicable
                weight_accumulator[key].append(score)

        averaged_weights = [
            (edge, sum(scores) / len(scores))
            for edge, scores in weight_accumulator.items()
        ]

        return {
            "nodes": predictions_list[0][
                "nodes"
            ],  # Assume nodes are consistent across TTAs
            "weights": averaged_weights,
        }

    def track(
        self,
        imgs: np.ndarray,
        masks: np.ndarray,
        feature_type: Literal["none", "wrfeat"],
        edge_threshold: float,
        test_time_augmentation: bool,
        progbar_class: Callable = tqdm,
        **kwargs,
    ):
        def get_tta_variants(images, masks):
            """Return list of (aug_img, aug_mask) pairs"""
            rotations = [np.rot90(images, k, (1, 2)) for k in range(1, 4)]
            flips = [np.flip(images, 1)]
            flips += [np.flip(rot, 1) for rot in rotations]
            aug_imgs = [images] + rotations + flips

            mask_rotations = [np.rot90(masks, k, (1, 2)) for k in range(1, 4)]
            mask_flips = [np.flip(masks, 1)]
            mask_flips += [np.flip(rot, 1) for rot in mask_rotations]
            aug_masks = [masks] + mask_rotations + mask_flips

            return list(zip(aug_imgs, aug_masks))

        def predict_all(variants):
            """Run predictions on all image/mask variants"""
            predictions = []
            for aug_img, aug_mask in variants:
                pred = self._predict(
                    aug_img,
                    aug_mask,
                    feature_type=feature_type,
                    edge_threshold=edge_threshold,
                    progbar_class=progbar_class,
                )
                predictions.append(pred)
            return predictions

        if test_time_augmentation:
            variants = get_tta_variants(imgs, masks)
            all_predictions = predict_all(variants)
            predictions = self.average_weights(all_predictions)
        else:
            predictions = self._predict(
                imgs,
                masks,
                feature_type=feature_type,
                edge_threshold=edge_threshold,
                progbar_class=progbar_class,
            )

        return predictions

    def track_from_disk(
        self,
        imgs_path: Path,
        masks_path: Path,
        mode: Literal["greedy_nodiv", "greedy", "ilp"] = "greedy",
        **kwargs,
    ) -> tuple[TrackGraph, np.ndarray]:
        """Track directly from two series of tiff files.

        Args:
            imgs_path:
                Options
                - Directory containing a series of numbered tiff files. Each file contains an image of shape (C),(Z),Y,X.
                - Single tiff file with time series of shape T,(C),(Z),Y,X.
            masks_path:
                Options
                - Directory containing a series of numbered tiff files. Each file contains an image of shape (C),(Z),Y,X.
                - Single tiff file with time series of shape T,(Z),Y,X.
            mode (optional):
                Mode for candidate graph pruning.
        """
        if not imgs_path.exists():
            raise FileNotFoundError(f"{imgs_path=} does not exist.")
        if not masks_path.exists():
            raise FileNotFoundError(f"{masks_path=} does not exist.")

        if imgs_path.is_dir():
            imgs = load_tiff_timeseries(imgs_path)
        else:
            imgs = tifffile.imread(imgs_path)

        if masks_path.is_dir():
            masks = load_tiff_timeseries(masks_path)
        else:
            masks = tifffile.imread(masks_path)

        if len(imgs) != len(masks):
            raise RuntimeError(
                f"#imgs and #masks do not match. Found {len(imgs)} images,"
                f" {len(masks)} masks."
            )

        if imgs.ndim - 1 == masks.ndim:
            if imgs[1] == 1:
                logger.info(
                    "Found a channel dimension with a single channel. Removing dim."
                )
                masks = np.squeeze(masks, 1)
            else:
                raise RuntimeError(
                    "Trackastra currently only supports single channel images."
                )

        if imgs.shape != masks.shape:
            raise RuntimeError(
                f"Img shape {imgs.shape} and mask shape {masks.shape} do not match."
            )

        return self.track(imgs, masks, mode, **kwargs), masks
