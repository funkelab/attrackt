import numpy as np


def load_csv_embeddings(node_embedding_file_name, sequences=None):
    with open(node_embedding_file_name, "r") as f:
        header = f.readline().strip().split()

    emb_cols = [h for h in header if h.startswith("emb_")]
    #d = len(emb_cols)  # embedding dimension
    float_fields = [(name, "f8") for name in emb_cols]
    dtype = np.dtype([("sequence", "U20"), ("id", "i8"), ("t", "i8")] + float_fields)

    node_embedding_data = np.genfromtxt(
        node_embedding_file_name,
        delimiter=" ",
        names=True,
        dtype=dtype,
        encoding="utf-8",
        autostrip=True,
    )

    if sequences is not None:
        node_embedding_data = node_embedding_data[
            np.isin(node_embedding_data["sequence"], sequences)
        ]

    return node_embedding_data
