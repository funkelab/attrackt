import logging
from argparse import ArgumentParser
from glob import glob
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union

import numpy as np
import tifffile
import yaml
from skimage.measure import regionprops

# ----------------- Logging Setup -----------------
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
)
logger = logging.getLogger(__name__)
# -------------------------------------------------


def create_csv(
    mask_dir_names: List[str],
    sequence_names: List[str],
    man_track_file_names: Optional[List[str]],
    output_csv_file_name: str,
) -> None:
    """
    Create CSV files from segmentation masks and manual track files.

    This function creates a CSV from available segmentation masks and a corresponding
    `man_track.txt`.

    The CSV file provides a unique ID for each segmentation, with 7 or 8 columns:
    `sequence unique_id time [z] y x parent_id original_id`.

    Unique segmentation IDs begin from 1 by default.

    Parameters
    ----------
    man_track_file_name : List[str]| None
        TXT files with CTC-style 4 columns (track_id time_start time_end parent_track_id).
        If not provided, parent id will be set to -1.
    output_csv_file_name : str
        Output CSV filename with unique segmentation IDs and parent IDs.
        Columns are [sequence id t [z] y x parent_id original_id].

    """

    all_data: List[List[Union[str, int, float]]] = []
    for seq_index in range(len(sequence_names)):
        sequence_name = sequence_names[seq_index]
        mask_dir_name = mask_dir_names[seq_index]
        mask_file_names = sorted(glob(mask_dir_name + "/*.tif"))
        num_frames = len(mask_file_names)
        track_data: Optional[np.ndarray] = None

        if man_track_file_names is not None:
            man_track_file = man_track_file_names[seq_index]
            if Path(man_track_file).exists():
                track_data = np.loadtxt(man_track_file, delimiter=" ")
                logger.info(f"Loaded manual track file: {man_track_file}")
            else:
                logger.warning(f"Manual track file not found: {man_track_file}")
        else:
            logger.info("No manual track file provided; parent IDs will be set to -1.")

        segmentation_id = 1
        mapping: Dict[int, Dict[int, List[float]]] = {}
        reverse_mapping: Dict[int, Dict[int, int]] = {}
        daughter_parent_mapping: Dict[int, Tuple[List[int], int]] = {}

        logger.info("Processing detections...")
        for time in range(num_frames):
            mapping[time] = {}
            reverse_mapping[time] = {}
            mask = tifffile.imread(mask_file_names[time])
            detections = regionprops(mask)
            for detection in detections:
                mapping[time][detection.label] = [segmentation_id, *detection.centroid]
                reverse_mapping[time][segmentation_id] = detection.label
                segmentation_id += 1

        logger.info("Processing manual track TXT...")
        if track_data is not None:
            for row in track_data:
                id_, time_start, time_end, parent_id = row.astype(int)
                daughter_parent_mapping[id_] = ([], parent_id)
                for time in range(time_start, time_end + 1):
                    mask = tifffile.imread(mask_file_names[time])
                    if np.any(mask == id_):
                        daughter_parent_mapping[id_][0].append(time)

        logger.info("Generating CSV output...")
        for time in range(num_frames):
            for key, value in mapping[time].items():
                if key in daughter_parent_mapping:
                    times, parent_track_id = daughter_parent_mapping[key]
                    index = times.index(time)
                    if index == 0:
                        if parent_track_id in daughter_parent_mapping:
                            time_parent = daughter_parent_mapping[parent_track_id][0][
                                -1
                            ]
                            parent_id_updated = mapping[time_parent][parent_track_id][0]
                        else:
                            parent_id_updated = 0
                    else:
                        time_parent = times[index - 1]
                        parent_id_updated = mapping[time_parent][key][0]
                    all_data.append(
                        [
                            sequence_name,
                            value[0],
                            time,
                            *value[1:],
                            parent_id_updated,
                            reverse_mapping[time][int(value[0])],
                        ]
                    )
                else:
                    all_data.append(
                        [
                            sequence_name,
                            value[0],
                            time,
                            *value[1:],
                            -1,
                            reverse_mapping[time][int(value[0])],
                        ]
                    )

    header_2d = "sequence id t y x parent_id original_id"
    header_3d = "sequence id t z y x parent_id original_id"

    logger.info(f"Saving output CSVs to: {output_csv_file_name}")
    if len(mask.shape) == 2:
        np.savetxt(
            output_csv_file_name,
            np.array(all_data, dtype=object),
            delimiter=" ",
            header=header_2d,
            fmt=["%s", "%i", "%i", "%.3f", "%.3f", "%i", "%i"],
        )
    elif len(mask.shape) == 3:
        np.savetxt(
            output_csv_file_name,
            np.array(all_data, dtype=object),
            delimiter=" ",
            header=header_3d,
            fmt=["%s", "%i", "%i", "%.3f", "%.3f", "%i", "%i"],
        )

    logger.info("CSV creation complete.")


# -------------------- Main CLI ------------------------


def main():
    parser = ArgumentParser(description="Create CSV from YAML config file")
    parser.add_argument(
        "--yaml_config_file_name", required=True, help="Path to YAML config file"
    )
    args = parser.parse_args()

    with open(args.yaml_config_file_name, "r") as f:
        config = yaml.safe_load(f)

    create_csv(
        mask_dir_names=config.get("mask_dir_names"),
        man_track_file_names=config.get("man_track_file_names"),
        output_csv_file_name=config["output_csv_file_name"],
    )


if __name__ == "__main__":
    main()
