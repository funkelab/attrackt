import argparse
import logging
import math
import random
from pathlib import Path
from typing import List, Literal

import yaml

from attrackt.scripts.load_csv_associations import load_csv_associations
from attrackt.scripts.load_csv_data import load_csv_data
from attrackt.scripts.load_csv_ilp_result import load_csv_ilp_result

# Set random seed
random.seed(42)

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
)


def round_to_next_power_of_10(n: int) -> int:
    """Rounds a number up to the next nearest power of 10."""
    if n <= 0:
        raise ValueError("Input must be a positive number")
    return 10 ** math.ceil(math.log10(n))


def _compute_outgoing_edges(data, keys) -> dict:
    """Loads edge data and returns a dictionary of cumulated outgoing edge values."""
    outgoing_edges = dict.fromkeys(keys, 0)
    if "weight" in data.dtype.names:
        for sequence, id_previous, t_previous, id_current, t_current, weight in zip(
            data["sequence"],
            data["id_previous"],
            data["t_previous"],
            data["id_current"],
            data["t_current"],
            data["weight"],
        ):
            node_in = f"{sequence}__{int(t_previous)}__{int(id_previous)}"
            if int(t_current) == int(t_previous) + 1:
                outgoing_edges[node_in] = outgoing_edges.get(node_in, 0) + weight
    else:
        for sequence, id_previous, t_previous, id_current, t_current in zip(
            data["sequence"],
            data["id_previous"],
            data["t_previous"],
            data["id_current"],
            data["t_current"],
        ):
            node_in = f"{sequence}__{int(t_previous)}__{int(id_previous)}"
            if int(t_current) == int(t_previous) + 1:
                outgoing_edges[node_in] = outgoing_edges.get(node_in, 0) + 1

    return outgoing_edges


def _save_nodes(
    nodes: list,
    output_dir: Path,
    output_csv_file_name: str,
    top_bottom: Literal["top", "bottom"],
    k: int,
):
    """Saves the top or bottom nodes per file index to .csv files."""

    output_path = output_dir / str(k) / top_bottom / f"{output_csv_file_name}"
    output_path.parent.mkdir(parents=True, exist_ok=True)
    header = ["# sequence", "id", "t", "uncertainty"]
    with open(output_path, "w") as f:
        f.write(" ".join(header) + "\n")
        for row in nodes:
            sequence, id_, t, score = row
            f.write(f"{str(sequence)} {id_} {t} {score:.3f}\n")


def _save_edges_for_nodes(
    nodes,
    numerical_data,
    sequence_data,
    output_dir: Path,
    output_csv_file_name: str,
    top_bottom: Literal["top", "bottom"],
    k: int,
):
    id_time_mapping = {}
    for sequence, row in zip(sequence_data, numerical_data):
        id_time_mapping[sequence + "_" + str(int(row[0]))] = int(row[1])

    parent_daughter_mapping = {}
    for sequence, row in zip(sequence_data, numerical_data):
        if row[-1] != 0:
            parent_time = int(id_time_mapping[sequence + "_" + str(int(row[-1]))])
            node_key = f"{sequence}__{parent_time}__{int(row[-1])}"
            if node_key in parent_daughter_mapping:
                parent_daughter_mapping[node_key].append(
                    f"{sequence}__{int(row[1])}__{int(row[0])}"
                )
            else:
                parent_daughter_mapping[node_key] = [
                    f"{sequence}__{int(row[1])}__{int(row[0])}"
                ]
    output_path = output_dir / str(k) / top_bottom / f"edges_{output_csv_file_name}"
    output_path.parent.mkdir(parents=True, exist_ok=True)
    header = ["# sequence", "id_previous", "t_previous", "id_current", "t_current"]

    with open(output_path, "w") as f:
        f.write(" ".join(header) + "\n")
        for row in nodes:
            sequence, id_previous, t_previous, *_ = row
            node_key = f"{sequence}__{int(t_previous)}__{int(id_previous)}"
            if node_key in parent_daughter_mapping:
                for daughter in parent_daughter_mapping[node_key]:
                    sequence, t_current, id_current = daughter.split("__")
                    sequence = str(sequence)
                    t_current = int(t_current)
                    id_current = int(id_current)
                    f.write(
                        f"{str(sequence)} {id_previous} {t_previous} {id_current} {t_current}\n"
                    )
            else:
                pass  # likely nodes at the very last frame of the sequence
                # logging.info(f"node_key {node_key} not found in parent_daughter_mapping.")
    return parent_daughter_mapping


def _save_edges_for_nodes_2(
    nodes,
    ilp_data,
    output_dir: Path,
    output_csv_file_name: str,
    top_bottom: Literal["top", "bottom"],
    k: int,
    top_nodes,
):
    parent_daughter_mapping = {}
    for sequence, id_previous, t_previous, id_current, t_current in zip(
        ilp_data["sequence"],
        ilp_data["id_previous"],
        ilp_data["t_previous"],
        ilp_data["id_current"],
        ilp_data["t_current"],
    ):
        node_key = f"{sequence}__{t_previous}__{id_previous}"
        if node_key in parent_daughter_mapping:
            parent_daughter_mapping[node_key].append(
                f"{sequence}__{t_current}__{id_current}"
            )
        else:
            parent_daughter_mapping[node_key] = [
                f"{sequence}__{t_current}__{id_current}"
            ]

    output_path = output_dir / str(k) / top_bottom / f"edges_{output_csv_file_name}"
    output_path.parent.mkdir(parents=True, exist_ok=True)
    header = ["# sequence", "id_previous", "t_previous", "id_current", "t_current"]

    with open(output_path, "w") as f:
        f.write(" ".join(header) + "\n")
        for row in nodes:
            sequence, id_previous, t_previous, *_ = row
            node_key = f"{sequence}__{int(t_previous)}__{int(id_previous)}"
            if node_key in parent_daughter_mapping:
                if row not in top_nodes:
                    for daughter in parent_daughter_mapping[node_key]:
                        sequence, t_current, id_current = daughter.split("__")
                        sequence = str(sequence)
                        t_current = int(t_current)
                        id_current = int(id_current)
                        f.write(
                            f"{str(sequence)} {id_previous} {t_previous} {id_current} {t_current}\n"
                        )


def sort(
    detection_csv_file_name: str,
    prediction_csv_file_name: str,
    ilp_csv_file_name: str,
    output_csv_file_name: str,
    sequences: List[str] = None,
    bottom_k_fr: float = 0.1,
    method: Literal["confidence", "random"] = "confidence",
    suffix: str | None = None,
):
    if method not in ["confidence", "random"]:
        raise ValueError(
            f"Invalid method '{method}'. Must be one of ['confidence', 'random']."
        )

    numerical_data, sequence_data, *_ = load_csv_data(
        Path(detection_csv_file_name), sequences=sequences
    )
    keys = []
    for sequence, row in zip(sequence_data, numerical_data):
        node_key = f"{sequence}__{int(row[1])}__{int(row[0])}"
        keys.append(node_key)
    bottom_k = int(bottom_k_fr * len(keys))
    target_dict = {}
    top_ks = [10, 100, 1000, 10000, 100000]

    ilp_result_data = load_csv_ilp_result(Path(ilp_csv_file_name), sequences)
    prediction_data = load_csv_associations(Path(prediction_csv_file_name), sequences)

    outgoing_edges_ilp = _compute_outgoing_edges(ilp_result_data, keys)
    outgoing_edges_prediction = _compute_outgoing_edges(prediction_data, keys)

    for k in outgoing_edges_prediction:
        diff = abs(outgoing_edges_prediction[k] - outgoing_edges_ilp.get(k))
        target_dict[k] = diff

    logging.info(f"Sorted dictionary has {len(target_dict)} keys.")
    sorted_dict = (
        dict(sorted(target_dict.items(), key=lambda item: item[1], reverse=True))
        if method == "confidence"
        else dict(random.sample(list(target_dict.items()), len(target_dict)))
    )

    for idx, top_k in enumerate(top_ks):
        top_nodes, bottom_nodes = [], []
        for node_key, value in list(sorted_dict.items())[:top_k]:
            sequence, t, id_ = node_key.split("__")
            sequence = str(sequence)
            t = int(t)
            id_ = int(id_)
            top_nodes.append([sequence, id_, t, value])

        for node_key, value in list(sorted_dict.items())[-bottom_k:]:
            sequence, t, id_ = node_key.split("__")
            sequence = str(sequence)
            t = int(t)
            id_ = int(id_)
            bottom_nodes.append([sequence, id_, t, value])

        _save_nodes(
            top_nodes,
            Path(f"{method}-{suffix}") if suffix else Path(method),
            output_csv_file_name,
            "top",
            round_to_next_power_of_10(top_k),
        )

        _save_edges_for_nodes(
            top_nodes,
            numerical_data,
            sequence_data,
            Path(f"{method}-{suffix}") if suffix else Path(method),
            output_csv_file_name,
            "top",
            round_to_next_power_of_10(top_k),
        )

        _save_nodes(
            bottom_nodes,
            Path(f"{method}-{suffix}") if suffix else Path(method),
            output_csv_file_name,
            "bottom",
            round_to_next_power_of_10(top_k),
        )

        _save_edges_for_nodes_2(
            bottom_nodes,
            ilp_result_data,
            Path(f"{method}-{suffix}") if suffix else Path(method),
            output_csv_file_name,
            "bottom",
            round_to_next_power_of_10(top_k),
            top_nodes,
        )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--yaml_config_file_name", type=str, required=True)
    args = parser.parse_args()

    config_path = Path(args.yaml_config_file_name)
    if not config_path.exists():
        logging.error(f"Config file not found: {config_path}")
        exit(1)

    with config_path.open() as stream:
        config = yaml.safe_load(stream)

    sort(
        prediction_csv_file_names=config["prediction_csv_file_names"],
        ilp_csv_file_names=config["ilp_csv_file_names"],
        bottom_k=config["bottom_k"],
        output_csv_file_names=config["output_csv_file_names"],
        method=config["method"],
    )


if __name__ == "__main__":
    main()
