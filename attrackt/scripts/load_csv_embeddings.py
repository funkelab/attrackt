import numpy as np


def load_csv_embeddings(node_embedding_file_name, sequences=None):
    float_fields = [(f"emb_{i}", "f8") for i in range(64)]
    dtype = np.dtype([("sequence", "U20"), ("id", "i8"), ("t", "i8")] + float_fields)

    node_embedding_data = np.genfromtxt(
        node_embedding_file_name,
        delimiter=" ",
        names=True,
        dtype=dtype,
        encoding="utf-8",
        autostrip=True,
    )

    # filter rows for chosen sequences
    if sequences is not None:
        node_embedding_data = node_embedding_data[
            np.isin(node_embedding_data["sequence"], sequences)
        ]

    return node_embedding_data
