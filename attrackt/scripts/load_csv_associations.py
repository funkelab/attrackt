import numpy as np


def load_csv_associations(edge_embedding_file_name, sequences=None):
    """Load associations CSV and optionally filter by one or more sequences.

    Args:
        edge_embedding_file_name (str | Path): Path to the associations file.
        sequences (list[str] | None): List of sequence names to keep.
            If None, all sequences are returned.

    Returns:
        np.ndarray: Structured array with association data.
    """
    dtype = np.dtype(
        [
            ("sequence", "U20"),
            ("id_previous", "i8"),
            ("t_previous", "i8"),
            ("id_current", "i8"),
            ("t_current", "i8"),
            ("weight", "f8"),
        ]
    )

    edge_embedding_data = np.genfromtxt(
        edge_embedding_file_name,
        delimiter=" ",
        names=True,
        dtype=dtype,
        encoding="utf-8",
        autostrip=True,
    )

    # filter rows for chosen sequences
    if sequences is not None:
        edge_embedding_data = edge_embedding_data[
            np.isin(edge_embedding_data["sequence"], sequences)
        ]

    return edge_embedding_data
