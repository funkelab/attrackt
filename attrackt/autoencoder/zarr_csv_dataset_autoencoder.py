import logging
from typing import Tuple

import numpy as np
import torch
import zarr
from torch.utils.data import IterableDataset, get_worker_info

from attrackt.autoencoder.utils import get_corners_bbox, update_corners
from attrackt.scripts import load_csv_data

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class ZarrCsvDatasetAutoencoder(IterableDataset):
    """ZarrCsvDatasetAutoencoder.
    This class is used to create an iterable dataset for training an autoencoder model.
    The dataset is created by cropping the zarr dataset around the detected positions.
    The detected positions are read from a csv file.
    The csv file should have the following format (space-separated):
        sequence id time [z] y x parent_id original_id.
    This class is used both during training and inference.
    """

    def __init__(
        self,
        zarr_container_name: str,
        detections_csv_file_name: str,
        num_spatial_dims: int,
        crop_size: Tuple[int, ...],
        scale_factor: int = 65535,
        dataset_name: str = "img",
        length: int | None = None,
        shuffle: bool = True,
        num_in_out_channels: int = 1,
    ):
        """__init__.

        Parameters
        ----------
        zarr_container_name : str
            zarr_container_name is the path to the zarr container.
        detections_csv_file_name : str
            detections_csv_file_name is the path to the csv file containing the detected positions.
        num_spatial_dims : int
            num_spatial_dims is the number of spatial dimensions.
        scale_factor: int
            divides the image intensities  by a constant factor.
        crop_size : Tuple[int, ...]
            crop_size is the spatial size of the image crop.
        dataset_name: str (default = 'img')
            the name of the raw image dataset in the zarr container.
        length : int | None = None
            length is the number of samples to generate.
        num_in_out_channels : int = 1
            number of channels in the image dataset.
        """

        super().__init__()
        self.detections_csv_file_name = detections_csv_file_name
        self.num_spatial_dims = num_spatial_dims
        self.create_detections_data()
        self.length = length
        self.shuffle = shuffle
        self.crop_size = (num_in_out_channels, *crop_size)
        self.num_in_out_channels = num_in_out_channels
        self.zarr_container_name = zarr_container_name
        self.dataset_name = dataset_name
        # self.f = zarr.open(self.zarr_container_name, mode="r")
        self.scale_factor = scale_factor

    def _open_zarr(self):
        return zarr.open(self.zarr_container_name, mode="r")

    def create_detections_data(self):
        """create_detections_data.
        This function reads the detections from the csv file.
        """

        voxel_size = {"x": 1.0, "y": 1.0}
        if self.num_spatial_dims == 3:
            voxel_size["z"] = 1.0

        self.detections_data, self.sequence_data, *_ = load_csv_data(
            csv_file_name=self.detections_csv_file_name,
            voxel_size=voxel_size,
            delimiter=" ",
        )

        logger.info(f"Loaded detections file {self.detections_csv_file_name}.")

    def __iter__(self):
        f = self._open_zarr()

        wi = get_worker_info()
        worker_id = wi.id if wi is not None else 0
        num_workers = wi.num_workers if wi is not None else 1
        # logger.info(f"Worker {worker_id} of {num_workers}.")
        rng = np.random.default_rng(seed=(hash((id(self), worker_id)) & 0xFFFFFFFF))

        N = len(self.detections_data)

        if self.length is None:
            while True:
                perm = rng.permutation(N)
                for idx in perm[worker_id::num_workers]:
                    yield self.create_sample(index=int(idx), f=f)
        else:
            # Finite number of samples
            if self.shuffle:
                perm = rng.permutation(N)
                idxs = perm[worker_id::num_workers][
                    : (self.length + num_workers - 1) // num_workers
                ]
            else:
                idxs = np.arange(worker_id, min(N, self.length), num_workers, dtype=int)

            for idx in idxs:
                yield self.create_sample(index=int(idx), f=f)

    def create_sample(self, index: int | None = None, f=None) -> Tuple:
        if f is None:
            f = self._open_zarr()

        if index is None:
            index = np.random.randint(0, len(self.detections_data))

        if self.num_spatial_dims == 2:
            node_id, time, y, x, *_ = self.detections_data[index]
        elif self.num_spatial_dims == 3:
            node_id, time, z, y, x, *_ = self.detections_data[index]
        sequence = str(self.sequence_data[index])

        time, node_id = int(time), int(node_id)

        assert len(self.crop_size) == self.num_spatial_dims + 1
        if self.num_spatial_dims == 2:
            tly, tlx, bry, brx = get_corners_bbox(
                position=(y, x), crop_size=self.crop_size[1:]
            )
            ds_crop = f[sequence][self.dataset_name][
                : self.num_in_out_channels, time, tly:bry, tlx:brx
            ]
        elif self.num_spatial_dims == 3:
            tlz, tly, tlx, brz, bry, brx = get_corners_bbox(
                position=(z, y, x), crop_size=self.crop_size[1:]
            )
            ds_crop = f[sequence][self.dataset_name][
                : self.num_in_out_channels, time, tlz:brz, tly:bry, tlx:brx
            ]

        if ds_crop.shape != self.crop_size:
            # This can happen if the detection is right at the edge of the image.
            # In that case, we expand the image, slightly.
            ds_t = f[sequence][self.dataset_name][
                : self.num_in_out_channels, time
            ]  # 3 h w
            if self.num_spatial_dims == 2:
                ds_t = np.pad(
                    ds_t,
                    (
                        (0, 0),
                        (self.crop_size[1] // 2, self.crop_size[1] // 2),
                        (self.crop_size[2] // 2, self.crop_size[2] // 2),
                    ),
                    mode="constant",
                )
                tly, tlx, bry, brx = update_corners(
                    tly=tly, tlx=tlx, bry=bry, brx=brx, crop_size=self.crop_size[1:]
                )

                ds_crop = ds_t[:, tly:bry, tlx:brx]
                assert ds_crop.shape == self.crop_size
            elif self.num_spatial_dims == 3:
                ds_t = np.pad(
                    ds_t,
                    (
                        (0, 0),
                        (self.crop_size[1] // 2, self.crop_size[1] // 2),
                        (self.crop_size[2] // 2, self.crop_size[2] // 2),
                        (self.crop_size[3] // 2, self.crop_size[3] // 2),
                    ),
                    mode="constant",
                )
                tlz, tly, tlx, brz, bry, brx = update_corners(
                    tlz=tlz,
                    tly=tly,
                    tlx=tlx,
                    brz=brz,
                    bry=bry,
                    brx=brx,
                    crop_size=self.crop_size[1:],
                )
                ds_crop = ds_t[:, tlz:brz, tly:bry, tlx:brx]

                assert ds_crop.shape == self.crop_size

        ds_crop = ds_crop.astype(np.float32)
        ds_crop = ds_crop / self.scale_factor
        return torch.from_numpy(ds_crop), sequence, node_id, time
