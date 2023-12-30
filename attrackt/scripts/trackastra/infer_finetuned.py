import os
from argparse import ArgumentParser
from collections import defaultdict
from typing import Dict, List

import numpy as np
import torch
import yaml
import zarr
from peft import PeftModel
from tqdm import tqdm

from attrackt.trackastra.data import build_windows, get_features
from attrackt.trackastra.model import TrackingTransformer
from attrackt.trackastra.model.predict import predict_windows
from attrackt.trackastra.utils import normalize

device = (
    "cuda"
    if torch.cuda.is_available()
    else "mps"
    if torch.backends.mps.is_available()
    else "cpu"
)


class CustomWrapperForPEFT(torch.nn.Module):
    def __init__(self, base_model):
        super().__init__()
        self.base_model = base_model
        self.window = self.base_model.config["window"]

    def normalize_output(self, A, timepoints, coords):
        return self.base_model.normalize_output(A, timepoints, coords)

    def forward(
        self, input_ids=None, coords=None, features=None, padding_mask=None, **kwargs
    ):
        if coords is None or features is None:
            raise ValueError("Coordinates and features must be provided.")

        return self.base_model(coords, features, padding_mask)


def average_weights(predictions_list: List[Dict]) -> Dict[str, List]:
    """
    Averages edge weights and selects nodes from first prediction.

    Args:
        predictions_list: list of predictions from TTA. Each prediction is a dict with
                          "nodes": List[Dict[str, Any]]
                          "weights": List[Tuple[Tuple[int, int], float]]

    Returns:
        Dict with averaged "nodes" and "weights".
    """
    weight_accumulator = defaultdict(list)

    for prediction in predictions_list:
        for (a, b), score in prediction["weights"]:
            key = (a, b)  # Ensure edge symmetry if applicable
            weight_accumulator[key].append(score)

    averaged_weights = [
        (edge, sum(scores) / len(scores)) for edge, scores in weight_accumulator.items()
    ]

    return {
        "nodes": predictions_list[0][
            "nodes"
        ],  # Assume nodes are consistent across TTAs
        "weights": averaged_weights,
    }


def predict(imgs, masks, feature_type, edge_threshold, model):
    # STEP ONE
    imgs = normalize(imgs)
    features = get_features(
        detections=masks,
        imgs=imgs,
        features=feature_type,
        ndim=masks.ndim - 1,
        n_workers=0,  # n_workers,
        progbar_class=tqdm,
    )

    # STEP TWO
    windows = build_windows(
        features,
        window_size=model.window,
        progbar_class=tqdm,
    )

    # STEP THREE
    pred = predict_windows(
        windows=windows,
        features=features,
        model=model,
        edge_threshold=edge_threshold,
        spatial_dim=masks.ndim - 1,
        progbar_class=tqdm,
        intra_window_weight=0,
        delta_t=1,
        window=model.window,
    )
    return pred


def infer(
    unsupervised_model_checkpoint: str,
    transformer: TrackingTransformer,
    finetuned_model_checkpoint_dir: str,
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
    # base_model = Trackastra.from_folder(
    #    Path(unsupervised_model_checkpoint), device=device
    # )
    if unsupervised_model_checkpoint is not None:
        checkpoint = torch.load(
            unsupervised_model_checkpoint,
            map_location=torch.device(device),
            weights_only=False,
        )
        transformer.load_state_dict(checkpoint["model_state_dict"])

    custom_model = CustomWrapperForPEFT(base_model=transformer)
    model = PeftModel.from_pretrained(custom_model, finetuned_model_checkpoint_dir)
    model = model.to(device)
    model.eval()

    if os.path.exists(output_csv_file_name):
        os.remove(output_csv_file_name)

    with open(output_csv_file_name, "w") as file:
        file.write("#sequence id_previous t_previous id_current t_current weight\n")

    def get_tta_variants(images, masks):
        """Return list of (aug_img, aug_mask) pairs"""
        rotations = [np.rot90(images, k, (1, 2)) for k in range(1, 4)]
        flips = [np.flip(images, 1)]
        flips += [np.flip(rot, 1) for rot in rotations]
        aug_imgs = [images] + rotations + flips

        mask_rotations = [np.rot90(masks, k, (1, 2)) for k in range(1, 4)]
        mask_flips = [np.flip(masks, 1)]
        mask_flips += [np.flip(rot, 1) for rot in mask_rotations]
        aug_masks = [masks] + mask_rotations + mask_flips

        return list(zip(aug_imgs, aug_masks))

    def predict_all(variants):
        """Run predictions on all image/mask variants"""
        predictions = []
        for aug_img, aug_mask in variants:
            pred = predict(
                aug_img,
                aug_mask,
                feature_type="wrfeat",
                edge_threshold=edge_threshold,
                model=model,
            )
            predictions.append(pred)
        return predictions

    f = zarr.open_group(zarr_path, "r")
    for sequence in test_zarr_sequences:
        imgs = f[sequence][zarr_img_key][zarr_img_channel]  # T Y X or T Z Y X
        masks = f[sequence][zarr_mask_key][zarr_mask_channel]  # T Y X or T Z Y X

        if test_time_augmentation:
            variants = get_tta_variants(imgs, masks)
            all_predictions = predict_all(variants)
            predictions = average_weights(all_predictions)
        else:
            predictions = predict(
                imgs,
                masks,
                feature_type="wrfeat",
                edge_threshold=edge_threshold,
                model=model,
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
    parser = ArgumentParser(description="Run inference using fine-trained model")
    parser.add_argument(
        "--yaml_config_file_name", required=True, help="Path to YAML config file"
    )
    args = parser.parse_args()

    with open(args.yaml_config_file_name, "r") as f:
        config = yaml.safe_load(f)

    infer(
        unsupervised_model_checkpoint=config.get("unsupervised_model_checkpoint"),
        finetuned_model_checkpoint_dir=config.get("finetuned_model_checkpoint_dir"),
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
