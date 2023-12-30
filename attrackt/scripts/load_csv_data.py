from typing import Dict, Tuple, cast

import numpy as np
import numpy.typing as npt


def load_csv_data(
    csv_file_name: str,
    voxel_size: Dict[str, float] = {"x": 1.0, "y": 1.0},
    delimiter: str = " ",
    sequences: list[str] | None = None,
) -> Tuple[
    npt.NDArray[np.float64], npt.NDArray[np.str_], dict[int, int], dict[str, int]
]:
    """
    Load CSV data with scaling based on voxel size and optional sequence filtering.

    Args:
        csv_file_name (str): Path to the CSV file.
        voxel_size (Dict[str, float]): Scaling factors for 'x', 'y', (optionally) 'z'.
        delimiter (str, optional): Delimiter used in the CSV. Defaults to ' '.
        sequences (list[str] | None): Optional list of sequence names to keep.
            If None, all sequences are returned.

    Returns:
        Tuple[np.ndarray, np.ndarray, dict[int, int], dict[str, int]]:
            - Numerical data array with scaled coordinates
            - Sequence column as ndarray[str]
            - Mapping from id to original_id (if available)
            - Reverse mapping from "t_original_id" to id
    """
    dtype = [
        ("sequence", "U20"),  # force Unicode string
        ("id", "i4"),
        ("t", "i4"),
        ("y", "f8"),
        ("x", "f8"),
        ("parent_id", "i4"),
        ("original_id", "i4"),
    ]

    data = np.genfromtxt(
        csv_file_name, delimiter=delimiter, names=True, dtype=dtype, encoding="utf-8"
    )

    if sequences is not None:
        data = data[np.isin(data["sequence"], sequences)]

    # Extract column names safely
    colnames = (
        cast(tuple[str, ...], data.dtype.names) if data.dtype.names is not None else ()
    )
    has_names = colnames is not None

    # Apply voxel scaling
    data["x"] *= voxel_size.get("x", 1.0)
    data["y"] *= voxel_size.get("y", 1.0)

    if has_names and "z" in colnames and "z" in voxel_size:
        data["z"] *= voxel_size["z"]

    # Define numeric columns dynamically
    numerical_cols = ["id", "t", "y", "x", "parent_id"]
    if has_names and "z" in colnames:
        numerical_cols.insert(2, "z")  # Insert 'z' at correct position

    # Stack numeric data
    numerical_data = np.column_stack([data[col] for col in numerical_cols])
    sequence_data = data["sequence"]

    # Mapping from id to original_id (if exists)
    if has_names and "original_id" in colnames:
        mapping = {
            int(id_): int(orig_id)
            for id_, orig_id in zip(data["id"], data["original_id"])
        }
        reverse_mapping = {
            f"{int(t)}_{int(orig_id)}": int(id_)
            for id_, t, orig_id in zip(data["id"], data["t"], data["original_id"])
        }
    else:
        mapping = {}
        reverse_mapping = {}

    return numerical_data, sequence_data, mapping, reverse_mapping
