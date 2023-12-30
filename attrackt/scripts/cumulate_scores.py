import argparse
import json
from pathlib import Path
from typing import List

import numpy as np
import yaml


def load_config(config_path):
    """Load YAML configuration file with dir_names and run_names."""
    try:
        with open(config_path, "r") as f:
            config = yaml.safe_load(f)
            sequence_names = config.get("sequence_names")
            return sequence_names
    except FileNotFoundError:
        raise FileNotFoundError(f"Config file '{config_path}' not found.")


def extract_metrics_from_file(filename):
    """Extracts relevant metrics from a single results.json file."""
    with open(filename) as f:
        data = json.load(f)

    res0 = data[0]["results"]
    res1 = data[1]["results"]["Frame Buffer 0"]

    return {
        "AOGM": res0["AOGM"],
        "fp_nodes": res0["fp_nodes"],
        "fn_nodes": res0["fn_nodes"],
        "fp_edges": res0["fp_edges"],
        "fn_edges": res0["fn_edges"],
        "TRA": res0["TRA"],
        "ws_edges": res0["ws_edges"],
        "ns_nodes": res0["ns_nodes"],
        "MBC": res1["Mitotic Branching Correctness"],
        "fp_divs": res1["False Positive Divisions"],
        "fn_divs": res1["False Negative Divisions"],
        "tp_divs": res1["True Positive Divisions"],
        "division_f1": res1["Division F1"],
    }


def aggregate_metrics(metric_list):
    """Computes the mean of each metric in a list of dicts."""
    keys = metric_list[0].keys()
    return {key: np.mean([m[key] for m in metric_list]) for key in keys}


def cumulate_scores(
    config_path: str | None = None, sequence_names: List[str] | None = None
):
    if (config_path is None and sequence_names is None) or (
        config_path is not None and sequence_names is not None
    ):
        raise ValueError(
            "Exactly one of config_path and sequence_names should be specified."
        )
    if sequence_names is None:
        sequence_names = load_config(config_path)
    all_sequence_metrics = []

    for sequence_name in sequence_names:
        filename = Path(sequence_name) / "results.json"
        metrics = extract_metrics_from_file(filename)
        all_sequence_metrics.append(metrics)

    # === Aggregate and Print ===
    averages = aggregate_metrics(all_sequence_metrics)
    header = "AOGM | MBC | division_f1 | fp_divs | fn_divs | tp_divs | ws_edges | fp_edges | fn_edges | TRA"
    summary_line = (
        f"{averages['AOGM']:.3f} | "
        f"{averages['MBC']:.3f} | "
        f"{averages['division_f1']:.3f} | "
        f"{averages['fp_divs']:.3f} | "
        f"{averages['fn_divs']:.3f} | "
        f"{averages['tp_divs']:.3f} | "
        f"{averages['ws_edges']:.3f} | "
        f"{averages['fp_edges']:.3f} | "
        f"{averages['fn_edges']:.3f} | "
        f"{averages['TRA']:.3f}"
    )

    print(header)
    print(summary_line)

    # Save to file
    # with open("results.txt", "w") as f:
    #    f.write(header + "\n")
    #    f.write(summary_line + "\n")


def main():
    parser = argparse.ArgumentParser(
        description="Aggregate tracking results from JSON."
    )
    parser.add_argument(
        "--config",
        type=str,
        required=True,
        help="Path to YAML config file (specify dir_names and run_names)",
    )
    args = parser.parse_args()

    cumulate_scores(args.config)


if __name__ == "__main__":
    main()
