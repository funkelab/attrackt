import logging
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import tifffile
import zarr

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def _load_mapping(mapping_csv_file_name: str) -> np.ndarray:
    """Load mapping CSV, detecting whether 'z' column exists."""
    with open(mapping_csv_file_name, "r", encoding="utf-8") as f:
        header = f.readline().strip().split()  # space-delimited

    dtype = [
        ("sequence", "U50"),
        ("id", "i8"),
        ("t", "i8"),
        ("y", "f8"),
        ("x", "f8"),
        ("parent_id", "i8"),
        ("original_id", "i8"),
    ]
    if "z" in header:
        dtype.insert(3, ("z", "f8"))

    arr = np.genfromtxt(
        mapping_csv_file_name,
        delimiter=" ",  # <-- use "," here if the file is CSV
        names=True,
        dtype=dtype,
        encoding="utf-8",
    )
    return arr


def _build_lookup(mapping: np.ndarray) -> Dict[Tuple[int, int], int]:
    """Map (t, original_id) -> id from a structured numpy array."""
    lookup: Dict[Tuple[int, int], int] = {}
    for row in mapping:
        lookup[(int(row["t"]), int(row["original_id"]))] = int(row["id"])
    return lookup


def _relabel_block(
    arr: np.ndarray, t: int, lookup: Dict[Tuple[int, int], int]
) -> np.ndarray:
    """Relabel a 2D or 3D mask block at time t using (t, original_id) -> new_id, safely."""
    src = arr  # NEVER mutate or compare against the same array
    dst = src.copy()

    old_vals = np.unique(src)
    old_vals = old_vals[old_vals != 0]

    for old in old_vals:
        new = lookup.get((t, int(old)), int(old))
        if new != old:
            dst[src == old] = new  # compare to src, write to dst

    return dst


def create_zarr(
    container_path: str,
    img_dir_names: List[str],
    mask_dir_names: List[str],
    sequence_names: List[str],
    mapping_csv_file_name: str,
) -> None:
    """
    Create/update a Zarr with images and relabeled masks.

    mapping CSV columns expected (space-delimited):
      sequence id t [z] y x parent_id original_id
    - z is optional
    - Only (t, original_id) are used for relabeling
    """
    assert len(img_dir_names) == len(mask_dir_names) == len(sequence_names)
    container = zarr.open(container_path, mode="a")

    # Load mapping
    mapping_all = _load_mapping(mapping_csv_file_name)

    for seq_name, img_dir_str, mask_dir_str in zip(
        sequence_names, img_dir_names, mask_dir_names
    ):
        image_dir = Path(img_dir_str)
        mask_dir = Path(mask_dir_str)

        image_fns = sorted(image_dir.glob("*.tif"))
        mask_fns = sorted(mask_dir.glob("*.tif"))
        if len(image_fns) != len(mask_fns):
            logger.info(
                f"Sequence '{seq_name}': #images ({len(image_fns)}) != #masks ({len(mask_fns)})"
            )
            logger.info(f"Using the first {len(mask_fns)} frames.")
            # Assume, that masks are only instance-labeled for the first n frames.
            image_fns = image_fns[: len(mask_fns)]

            # raise ValueError(
            #    f"Sequence '{seq_name}': #images ({len(image_fns)}) "
            #    f"!= #masks ({len(mask_fns)})"
            # )

        # Read stacks
        image_list, mask_list = [], []
        for im_fn, ma_fn in zip(image_fns, mask_fns):
            im = tifffile.imread(im_fn).astype(np.float32)
            ma = tifffile.imread(ma_fn).astype(np.uint32)
            image_list.append(im)
            mask_list.append(ma)

        image_arr = np.asarray(image_list)  # TZYX or TYX
        mask_arr = np.asarray(mask_list)

        if mask_arr.ndim not in (3, 4):
            raise ValueError(
                f"Unexpected mask ndim {mask_arr.ndim}; expected 3 (T,Y,X) "
                "or 4 (T,Z,Y,X)."
            )

        # Filter mapping for this sequence
        mapping = mapping_all[mapping_all["sequence"] == seq_name]
        logger.info("Sequence '%s': using %d mapping rows.", seq_name, len(mapping))
        lookup = _build_lookup(mapping)

        # Relabel masks
        relabeled = np.empty_like(mask_arr)
        T = mask_arr.shape[0]
        for t in range(T):
            relabeled[t] = _relabel_block(mask_arr[t], t, lookup)

        # Write to Zarr (add leading channel dimension)
        seq_grp = container.require_group(seq_name)
        seq_grp.create_dataset("img", data=image_arr[np.newaxis], overwrite=True)
        seq_grp.create_dataset("mask", data=relabeled[np.newaxis], overwrite=True)

        # Axis names (C, T, (Z,) Y, X)
        if image_arr.ndim == 4:  # T,Z,Y,X
            axis_names = ("c", "t", "z", "y", "x")
        elif image_arr.ndim == 3:  # T,Y,X
            axis_names = ("c", "t", "y", "x")
        else:
            raise ValueError(f"Unexpected image ndim {image_arr.ndim}")

        seq_grp["img"].attrs.update(
            {
                "resolution": (1,) * (len(axis_names) - 1),
                "offset": (0,) * (len(axis_names) - 1),
                "axis_names": axis_names,
            }
        )
        seq_grp["mask"].attrs.update(
            {
                "resolution": (1,) * (len(axis_names) - 1),
                "offset": (0,) * (len(axis_names) - 1),
                "axis_names": axis_names,
            }
        )

        before, after = np.unique(mask_arr).size, np.unique(relabeled).size
        logger.info(
            "Sequence '%s': relabeled masks written. Unique labels: %d -> %d",
            seq_name,
            before,
            after,
        )

    logger.info("Created/updated container at %s.", container_path)
