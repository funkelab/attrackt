import logging
import random
from collections.abc import Sequence
from pathlib import Path
from timeit import default_timer

import joblib
import numpy as np
import torch
import zarr
from torch.utils.data import Dataset
from tqdm import tqdm

from attrackt.scripts import load_csv_embeddings, load_csv_ilp_result

from ..utils import normalize
from . import wrfeat

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)


class CTCZarrData(Dataset):
    """Zarr-only dataset for cell tracking windowing.

    Assumptions:
      - Zarr arrays are channel-first: C, T, [Z], Y, X
      - We select a single channel per key and then operate on T, [Z], Y, X
      - Embeddings CSV adds per-(t,id) vectors
      - Supervised CSV provides information about links.
      Should contain 5 space delimited columns: Sequence t_previous id_previous t_current id_current.

    Parameters (Zarr):
      zarr_path: path to root Zarr group
      zarr_sequence: subgroup name inside zarr_path (if None, first subgroup)
      zarr_img_key, zarr_mask_key: dataset names under the sequence group
      zarr_img_channel, zarr_mask_channel: which C index to use (default 0)
    """

    def __init__(
        self,
        ndim: int,
        window_size: int,
        max_tokens: int | None,
        augment: int,
        crop_size: tuple[int, ...] | None,
        zarr_path: str | Path,
        zarr_sequence: str | None,
        zarr_img_key: str,
        zarr_mask_key: str,
        zarr_img_channel: int,
        zarr_mask_channel: int,
        embeddings_csv: str | Path | None = None,
        supervised_csv: str | Path | None = None,
        n_jobs: int = 8,
    ) -> None:
        super().__init__()

        if embeddings_csv is None and supervised_csv is None:
            raise ValueError("One of embeddings_csv or supervised_csv must be given.")
        if embeddings_csv is not None and supervised_csv is not None:
            raise ValueError(
                "Only one of embeddings_csv or supervised_csv can be given."
            )

        # --- config ---
        self.ndim = ndim
        self.feat_dim = 7 if ndim == 2 else 12
        self.window_size = int(window_size)
        if self.window_size <= 1:
            raise ValueError("window_size must be > 1")
        self.max_tokens = max_tokens

        # --- zarr wiring ---
        self.zarr_path = Path(zarr_path)
        self.zarr_sequence = zarr_sequence
        self.zarr_img_key = zarr_img_key
        self.zarr_mask_key = zarr_mask_key
        self.zarr_img_channel = int(zarr_img_channel)
        self.zarr_mask_channel = int(zarr_mask_channel)
        self.n_jobs = n_jobs
        self.embeddings_csv = embeddings_csv
        self.supervised_csv = supervised_csv

        logger.info(
            "Using Zarr at %s (seq=%s, img=%s[ch=%d], mask=%s[ch=%d])",
            self.zarr_path,
            self.zarr_sequence,
            self.zarr_img_key,
            self.zarr_img_channel,
            self.zarr_mask_key,
            self.zarr_mask_channel,
        )

        # --- embeddings ---
        if self.embeddings_csv is not None:
            self.embeddings_dictionary = {}
            logger.info("Loading embeddings from %s", self.embeddings_csv)
            node_embedding_data = load_csv_embeddings(
                str(self.embeddings_csv), sequences=[self.zarr_sequence]
            )
            # make (N, 64) float embeddings
            emb_cols = [f"emb_{i}" for i in range(64)]
            emb_matrix = np.column_stack(
                [node_embedding_data[c] for c in emb_cols]
            ).astype(np.float32)

            for (id_, t), emb in zip(
                zip(node_embedding_data["id"], node_embedding_data["t"]),
                emb_matrix,
            ):
                node_id = f"{t}_{id_}"
                self.embeddings_dictionary[node_id] = emb

        # --- supervised (optional) ---
        if supervised_csv is not None:
            self.supervised_dictionary = {}
            logger.info("Loading supervised labels from %s", supervised_csv)
            supervised_data = load_csv_ilp_result(
                str(supervised_csv), sequences=[self.zarr_sequence]
            )
            for id_previous, t_previous, id_current, t_current in zip(
                supervised_data["id_previous"],
                supervised_data["t_previous"],
                supervised_data["id_current"],
                supervised_data["t_current"],
            ):
                node_id_previous = f"{t_previous}_{id_previous}"
                node_id_current = f"{t_current}_{id_current}"
                if node_id_previous not in self.supervised_dictionary:
                    self.supervised_dictionary[node_id_previous] = []
                self.supervised_dictionary[node_id_previous].append(node_id_current)
            logger.info("Loaded %d supervised links.", len(self.supervised_dictionary))

        # --- augs/feature heads ---
        self.augmenter, self.cropper = self._setup_features_augs_wrfeat(
            ndim, augment, crop_size
        )

        # --- load windows ---
        start = default_timer()
        self.windows = self._load_wrfeat()

        if len(self.windows) > 0:
            self.ndim = self.windows[0]["coords"].shape[1]
            self.n_objects = tuple(len(t["coords"]) for t in self.windows)
            logger.info(
                "Found %d objects in %d windows (%.1fs)",
                int(np.sum(self.n_objects)),
                len(self.windows),
                default_timer() - start,
            )
        else:
            self.n_objects = 0
            logger.warning("No tracks/windows found.")

    # --------------------------------------------------------------------- utils

    def _setup_features_augs_wrfeat(
        self,
        ndim: int,
        augment: int,
        crop_size: tuple[int, ...] | None,
    ):
        """
        Configure WRFeat-specific feature dim, augmenter, and cropper.

        Returns:
            augmenter: Optional[wrfeat.WRAugmentationPipeline]
            cropper: Optional[wrfeat.WRRandomCrop]
        """
        # heuristic: 2D -> 7, 3D -> 12 (same as your original)
        if augment == 1:
            augmenter = wrfeat.WRAugmentationPipeline(
                [
                    wrfeat.WRRandomFlip(p=0.5),
                    wrfeat.WRRandomAffine(
                        p=0.8, degrees=180, scale=(0.5, 2), shear=(0.1, 0.1)
                    ),
                ]
            )
        elif augment > 1:
            augmenter = wrfeat.WRAugmentationPipeline(
                [
                    wrfeat.WRRandomFlip(p=0.5),
                    wrfeat.WRRandomAffine(
                        p=0.8, degrees=180, scale=(0.5, 2), shear=(0.1, 0.1)
                    ),
                    wrfeat.WRRandomBrightness(p=0.8),
                    wrfeat.WRRandomOffset(p=0.8, offset=(-3, 3)),
                ]
            )
        else:
            augmenter = None

        cropper = (
            wrfeat.WRRandomCrop(
                crop_size=crop_size,
                ndim=ndim,
            )
            if crop_size is not None
            else None
        )
        return augmenter, cropper

    # --------------------------------------------------------------------- zarr

    def _open_zarr_seq(self) -> zarr.Group:
        """Open the per-sequence Zarr group."""
        group = zarr.open_group(str(self.zarr_path), mode="r")
        seq = self.zarr_sequence
        if seq is None:
            if len(group.groups()) == 0:
                return group
            seq = sorted(group.group_keys())[0]
            self.zarr_sequence = seq
        if seq not in group:
            raise ValueError(f"Zarr group '{seq}' not found in {self.zarr_path}")
        return group[seq]

    def _select_channel_and_reorder(self, arr: np.ndarray, channel: int) -> np.ndarray:
        """From C,T,[Z],Y,X -> select channel -> T,[Z],Y,X."""
        if arr.ndim < 4:
            raise ValueError(f"Expected at least 4D (C,T,*,Y,X). Got shape {arr.shape}")
        if channel < 0 or channel >= arr.shape[0]:
            raise IndexError(
                f"Channel index {channel} out of range for shape {arr.shape}"
            )
        arr = arr[channel]  # now T,[Z],Y,X
        return arr

    def _load_zarr_ds(self, key: str, channel: int, dtype=None) -> np.ndarray:
        """Load C,T,[Z],Y,X; select channel;

        Returns T,[Z],Y,X after:
          - selecting `channel` along C
        """
        g = self._open_zarr_seq()
        if key not in g:
            raise ValueError(f"Zarr dataset '{key}' not found in group '{g.name}'")
        arr = np.asarray(g[key])
        if dtype is not None:
            arr = arr.astype(dtype)

        # Expect C,T,[Z],Y,X
        if arr.ndim not in (4, 5):
            raise ValueError(f"Expected C,T,Y,X or C,T,Z,Y,X; got shape {arr.shape}")

        # Select channel -> T,[Z],Y,X
        arr = self._select_channel_and_reorder(arr, channel)

        return arr

    def _load_masks(self):
        """Load ground-truth masks (Zarr-only, channel-first)."""
        masks = self._load_zarr_ds(
            self.zarr_mask_key, channel=self.zarr_mask_channel, dtype=np.int32
        )
        return masks

    # ----------------------------------------------------------- masks/images io

    def _check_dimensions(self, x: np.ndarray) -> np.ndarray:
        """Ensure data matches requested ndim after channel selection.

        Expect:
          - 2D: T,H,W
          - 3D: T,Z,H,W
        """
        if self.ndim == 2:
            if x.ndim != 3:
                raise ValueError(f"Expected 2D data (T,H,W), got {x.shape}")
        else:
            if x.ndim == 3:
                x = np.expand_dims(x, axis=1)  # T,Z,H,W with Z=1
            elif x.ndim != 4:
                raise ValueError(f"Expected 3D data (T,Z,H,W), got {x.shape}")
        return x

    # -------------------------------------------------------------- main loaders

    def _load_wrfeat(self):
        """wrfeat pipeline: builds WRFeatures from (mask, img) per frame."""
        self.gt_masks = self._load_masks()
        self.gt_masks = self._check_dimensions(self.gt_masks)

        imgs = self._load_zarr_ds(
            self.zarr_img_key, channel=self.zarr_img_channel, dtype=np.float32
        )
        self.imgs = np.stack(
            [normalize(_x) for _x in tqdm(imgs, desc="Normalizing", leave=False)]
        )
        self.imgs = self._check_dimensions(self.imgs)
        self.gt_masks = self.gt_masks.astype(np.int32, copy=False)
        self.imgs = self.imgs.astype(np.float32, copy=False)

        assert len(self.gt_masks) == len(self.imgs)

        det_masks = self.gt_masks
        logger.info("DET MASK: Using Zarr masks")

        features = joblib.Parallel(n_jobs=self.n_jobs)(
            joblib.delayed(wrfeat.WRFeatures.from_mask_img)(
                mask=mask[None], img=img[None], t_start=t
            )
            for t, (mask, img) in enumerate(zip(det_masks, self.imgs))
        )

        windows = self._build_windows_wrfeat(features, det_masks)
        del self.imgs
        del self.gt_masks

        return windows

    # -------------------------------------------------------------- window build

    def _build_windows_wrfeat(
        self, features: Sequence[wrfeat.WRFeatures], det_masks: np.ndarray
    ):
        """Build sliding windows for wrfeat."""
        windows = []
        for t1, t2 in tqdm(
            zip(range(0, len(det_masks)), range(self.window_size, len(det_masks) + 1)),
            total=len(det_masks) - self.window_size + 1,
            leave=False,
            desc="Building windows",
        ):
            feat = wrfeat.WRFeatures.concat(features[t1:t2])

            labels = feat.labels
            timepoints = feat.timepoints
            coords = feat.coords
            autoencoder_features = None
            if hasattr(self, "embeddings_dictionary"):

                def get_or_zero(key, D=64):
                    return self.embeddings_dictionary.get(key, np.zeros(D, np.float32))

                # infer D once if available
                D = (
                    len(next(iter(self.embeddings_dictionary.values())))
                    if len(getattr(self, "embeddings_dictionary", {})) > 0
                    else 64
                )
                autoencoder_features = np.stack(
                    [
                        get_or_zero(f"{int(t)}_{int(l)}", D)
                        for t, l in zip(timepoints, labels)
                    ]
                )

            w = dict(
                coords=coords,
                t1=t1,
                labels=labels,
                timepoints=timepoints,
                wrfeat=feat,
            )
            if autoencoder_features is not None:
                w["autoencoder_features"] = autoencoder_features

            windows.append(w)

        logger.debug("Built %d wrfeat windows.", len(windows))
        return windows

    # ----------------------------------------------------------------- indexing

    def __len__(self):
        return len(self.windows)

    def __getitem__(self, n: int):
        if self.supervised_csv is None and self.embeddings_csv is not None:
            return self._getitem_wrfeat(n)
        else:
            return self._getitem_wrfeat_supervised()

    def _getitem_wrfeat(self, n: int):
        track = self.windows[n]
        feat: wrfeat.WRFeatures = track[
            "wrfeat"
        ]  # holds coords, labels, timepoints, features_stacked
        aef = track.get("autoencoder_features", None)  # optional (N, D_e)

        # --- cropping (keeps WRFeatures + aligns A and embeddings) ---
        if self.cropper is not None:
            feat, idx = self.cropper(feat)
            if aef is not None:
                aef = aef[idx]

        # --- augmentation on WRFeatures (coords/features-intensity space) ---
        if self.augmenter is not None:
            feat = self.augmenter(feat)

        # --- pack arrays (numpy) ---
        coords0_np = np.concatenate(
            (feat.timepoints[:, None], feat.coords), axis=-1
        ).astype(np.float32, copy=False)
        features_np = feat.features_stacked.astype(np.float32, copy=False)
        labels_np = feat.labels.astype(np.int64, copy=False)
        timepoints_np = feat.timepoints.astype(np.int64, copy=False)

        # --- token cap (cap by number of detections/tokens) ---
        n_tokens = features_np.shape[0]
        if self.max_tokens is not None and n_tokens > self.max_tokens:
            n_cap = int(self.max_tokens)
            features_np = features_np[:n_cap]
            labels_np = labels_np[:n_cap]
            timepoints_np = timepoints_np[:n_cap]
            coords0_np = coords0_np[:n_cap]
            if aef is not None:
                aef = aef[:n_cap]

        # --- to torch ---
        coords0 = torch.from_numpy(coords0_np)  # (N, 1+ndim) float32; first column is t
        features = torch.from_numpy(features_np)  # (N, F) float32
        labels = torch.from_numpy(labels_np)  # (N,) int64
        timepoints = torch.from_numpy(timepoints_np)  # (N,) int64

        # --- final coord variant (optionally jitter spatial dims) ---
        if self.augmenter is not None:
            coords = coords0.clone()
            # jitter only spatial columns (exclude time column at [:, 0])
            coords[:, 1:] += torch.randint(0, 256, (1, self.ndim))
        else:
            coords = coords0.clone()

        res = {
            "features": features,
            "coords0": coords0,
            "coords": coords,
            "timepoints": timepoints,
            "labels": labels,
        }

        res["autoencoder_features"] = torch.as_tensor(aef, dtype=torch.float32)

        return res

    def _getitem_wrfeat_supervised(self):
        supervised_node = random.choice(list(self.supervised_dictionary.keys()))

        t_target, id_target = map(int, supervised_node.split("_"))

        # pick a window that contains t_target
        len_windows = len(self.windows)
        low = max(0, t_target - self.window_size + 1)
        high = min(t_target + 1, len_windows)
        n = np.random.randint(low, high)

        track = self.windows[n]
        feat: wrfeat.WRFeatures = track["wrfeat"]

        # --- crop first ---
        if self.cropper is not None:
            feat, _ = self.cropper(feat)

        # --- augmentation on WRFeatures (coords/features-intensity space) ---
        if self.augmenter is not None:
            feat = self.augmenter(feat)

        # --- now build A/mask on the CROPPED tokens ---
        labels = feat.labels.astype(int)
        timepoints = feat.timepoints.astype(int)
        N = len(labels)
        A = np.zeros((N, N), dtype=np.float32)
        M = np.zeros((N, N), dtype=bool)

        # map time→indices within the cropped set
        tmin, tmax = int(timepoints.min()), int(timepoints.max())
        time_to_idx = {
            tt: np.nonzero(timepoints == tt)[0] for tt in range(tmin, tmax + 1)
        }

        # string ids for lookup into your supervision dict
        obj_ids = [f"{timepoints[i]}_{labels[i]}" for i in range(N)]
        obj_to_row = {obj: i for i, obj in enumerate(obj_ids)}

        for j in range(N):
            obj = obj_ids[j]
            if obj in self.supervised_dictionary:
                next_t = timepoints[j] + 1
                current_rows = time_to_idx[timepoints[j]]
                if next_t in time_to_idx:
                    next_rows = time_to_idx[next_t]  # candidates in next frame
                    M[j, next_rows] = True
                    M[next_rows, j] = True  # and vice versa (symmetric)
                    next_objs = self.supervised_dictionary[obj]  # this is a list
                    for next_obj in next_objs:
                        if next_obj in obj_to_row:
                            A[obj_to_row[next_obj], j] = 1.0
                            A[j, obj_to_row[next_obj]] = 1.0
                            M[current_rows, obj_to_row[next_obj]] = True
                            M[obj_to_row[next_obj], current_rows] = True
                            # note if a positive link exists from obj to
                            # next_obj, then all other incoming links to
                            # next_obj should be 0.

        # for j in range(N):
        #    obj = obj_ids[j]
        #    if obj in self.supervised_dictionary:
        #        prev_t = timepoints[j] - 1
        #        if prev_t in time_to_idx:
        #            prev_rows = time_to_idx[
        #                prev_t
        #            ]  # candidates in prev frame (cropped)
        #            M[prev_rows, j] = True  # supervise this column on prev frame row
        #            M[j, prev_rows] = True  # and vice versa (symmetric)
        #            prev_obj = self.supervised_dictionary[obj]
        #            if (
        #                prev_obj in obj_to_row
        #            ):  # only if the true prev survived the crop
        #                A[obj_to_row[prev_obj], j] = 1.0
        #                A[j, obj_to_row[prev_obj]] = 1.0

        # pack tensors from feat.* (already cropped)
        coords0_np = np.concatenate(
            (feat.timepoints[:, None], feat.coords), axis=-1
        ).astype(np.float32, copy=False)
        feats_np = feat.features_stacked.astype(np.float32, copy=False)
        labels_np = labels.astype(np.int64, copy=False)
        times_np = timepoints.astype(np.int64, copy=False)

        # token cap (apply same cap to A/M and aef)
        if self.max_tokens is not None and len(labels_np) > self.max_tokens:
            n_cap = int(self.max_tokens)
            coords0_np = coords0_np[:n_cap]
            feats_np = feats_np[:n_cap]
            labels_np = labels_np[:n_cap]
            times_np = times_np[:n_cap]
            A = A[:n_cap, :n_cap]
            M = M[:n_cap, :n_cap]

        # to torch
        coords0 = torch.from_numpy(coords0_np)
        features = torch.from_numpy(feats_np)
        labels_t = torch.from_numpy(labels_np)
        times_t = torch.from_numpy(times_np)
        A_t = torch.from_numpy(A)
        M_t = torch.from_numpy(M)

        # --- final coord variant (optionally jitter spatial dims) ---
        if self.augmenter is not None:
            coords = coords0.clone()
            # jitter only spatial columns (exclude time column at [:, 0])
            coords[:, 1:] += torch.randint(0, 256, (1, self.ndim))
        else:
            coords = coords0.clone()

        out = {
            "features": features,
            "coords0": coords0,
            "coords": coords,
            "timepoints": times_t,
            "labels": labels_t,
            "assoc_matrix": A_t,
            "assoc_matrix_mask": M_t,
        }

        return out


def pad_tensor(x, n_max: int, dim=0, value=0):
    n = x.shape[dim]
    if n_max < n:
        raise ValueError(f"pad_tensor: n_max={n_max} must be larger than n={n} !")
    pad_shape = list(x.shape)
    pad_shape[dim] = n_max - n
    pad = torch.full(pad_shape, fill_value=value, dtype=x.dtype).to(x.device)
    return torch.cat((x, pad), dim=dim)


def collate_sequence_padding(max_len: int | None = None):
    def collate_sequence(batch):
        lens = tuple(len(x["coords"]) for x in batch)
        n_max_len = max(lens) if max_len is None else max_len

        present = set().union(*[b.keys() for b in batch])

        seq_pad_vals = {"coords": 0, "features": 0, "labels": 0, "timepoints": -1}
        if "autoencoder_features" in present:
            seq_pad_vals["autoencoder_features"] = 0

        out = {}
        for k, v in seq_pad_vals.items():
            out[k] = torch.stack(
                [pad_tensor(b[k], n_max=n_max_len, value=v) for b in batch], dim=0
            )

        # Square-pad NxN matrices if they exist
        for k in ("assoc_matrix", "assoc_matrix_mask"):
            if k in present:

                def pad_sq(M, nmax, val):
                    N = M.shape[-1]
                    if N == nmax:
                        return M
                    # pad=(left,right,top,bottom)
                    return torch.nn.functional.pad(
                        M, (0, nmax - N, 0, nmax - N), value=val
                    )

                fill = 0.0 if k == "assoc_matrix" else False
                mats = [pad_sq(b[k], n_max_len, fill) for b in batch]
                out[k] = torch.stack(mats, dim=0)

        pad_mask = torch.zeros((len(batch), n_max_len), dtype=torch.bool)
        for i, L in enumerate(lens):
            pad_mask[i, L:] = True
        out["padding_mask"] = pad_mask
        return out

    return collate_sequence
