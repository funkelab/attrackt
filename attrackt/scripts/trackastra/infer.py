import os
from argparse import ArgumentParser
from typing import List

import torch
import yaml
import zarr

from attrackt.trackastra.model import Trackastra, TrackingTransformer

device = (
    "cuda"
    if torch.cuda.is_available()
    else "mps"
    if torch.backends.mps.is_available()
    else "cpu"
)


def infer(
    model_checkpoint: str,
    transformer: TrackingTransformer,
    zarr_img_channel: int,
    zarr_img_key: str,
    zarr_mask_channel: int,
    zarr_mask_key: str,
    zarr_path: str,
    test_zarr_sequences: List[str],
    output_csv_file_name: str,
    edge_threshold: float = 0.01,
    test_time_augmentation: bool = True,
):
    """
    Perform inference using the Trackastra model.

    Args:
    """
    if model_checkpoint is not None:
        # model = Trackastra.from_folder(Path(model_checkpoint), device=device)
        checkpoint = torch.load(
            model_checkpoint, map_location=torch.device(device), weights_only=False
        )
        transformer.load_state_dict(checkpoint["model_state_dict"])
    trackastra_model = Trackastra(transformer=transformer, device=device)
    if os.path.exists(output_csv_file_name):
        os.remove(output_csv_file_name)

    with open(output_csv_file_name, "w") as file:
        file.write("#sequence id_previous t_previous id_current t_current weight\n")

    f = zarr.open_group(zarr_path, "r")
    for sequence in test_zarr_sequences:
        imgs = f[sequence][zarr_img_key][zarr_img_channel]  # T Y X or T Z Y X
        masks = f[sequence][zarr_mask_key][zarr_mask_channel]  # T Y X or T Z Y X
        with torch.no_grad():
            # Track the cells
            predictions = trackastra_model.track(
                imgs,
                masks,
                feature_type="wrfeat",
                edge_threshold=edge_threshold,
                test_time_augmentation=test_time_augmentation,
            )

            id_time_dictionary = {}
            for node in predictions["nodes"]:
                id_time_dictionary[node["id"]] = (int(node["time"]), int(node["label"]))

            for edge, score in predictions["weights"]:
                source, target = edge
                t_source, label_source = id_time_dictionary[source]
                t_target, label_target = id_time_dictionary[target]

                if t_target == t_source + 1:
                    with open(output_csv_file_name, "a") as file:
                        file.write(
                            f"{sequence} {label_source} {t_source} {label_target} {t_target} {score}\n"
                        )


def main():
    parser = ArgumentParser(description="Run inference using pre-trained model")
    parser.add_argument(
        "--yaml_config_file_name", required=True, help="Path to YAML config file"
    )
    args = parser.parse_args()

    with open(args.yaml_config_file_name, "r") as f:
        config = yaml.safe_load(f)

    infer(
        model_checkpoint=config.get("model_checkpoint"),
        zarr_img_channel=config.get("zarr_img_channel"),
        zarr_img_key=config.get("zarr_img_key"),
        zarr_mask_channel=config.get("zarr_mask_channel"),
        zarr_mask_key=config.get("zarr_mask_key"),
        zarr_path=config.get("zarr_path"),
        test_zarr_sequences=config.get("test_zarr_sequences"),
        output_csv_file_name=config.get("output_csv_file_name"),
        edge_threshold=config.get("edge_threshold", 0.0),
        test_time_augmentation=config.get("test_time_augmentation", True),
    )


if __name__ == "__main__":
    main()
