import numpy as np


def load_csv_ilp_result(ilp_result_file_name, sequences=None):
    dtype = np.dtype(
        [
            ("sequence", "U20"),
            ("id_previous", "i8"),
            ("t_previous", "i8"),
            ("id_current", "i8"),
            ("t_current", "i8"),
        ]
    )

    ilp_result_data = np.genfromtxt(
        ilp_result_file_name,
        delimiter=" ",
        names=True,
        dtype=dtype,
        encoding="utf-8",
        autostrip=True,
    )

    # filter rows for chosen sequences
    if sequences is not None:
        ilp_result_data = ilp_result_data[
            np.isin(ilp_result_data["sequence"], sequences)
        ]

    return ilp_result_data
