import logging
from pathlib import Path

import numpy as np
import tifffile

logging.basicConfig(level=logging.INFO)


def correct_gt_with_st(
    silver_truth_dir_name: str, gold_truth_dir_name: str, combined_truth_dir_name: str
) -> None:
    silver_dir = Path(silver_truth_dir_name)
    gold_dir = Path(gold_truth_dir_name)
    output_dir = Path(combined_truth_dir_name)
    output_dir.mkdir(exist_ok=True, parents=True)

    silver_files = sorted(silver_dir.glob("man_seg*.tif"))
    gold_files = sorted(gold_dir.glob("man_track*.tif"))
    assert len(silver_files) == len(gold_files)

    for t in range(len(silver_files)):
        silver_path = silver_files[t]
        silver_mask = tifffile.imread(silver_path)
        output_mask = silver_mask.copy()
        gold_path = gold_files[t]
        gold_mask = tifffile.imread(gold_path)
        output_mask = np.maximum(gold_mask, silver_mask)
        tifffile.imwrite(
            output_dir / gold_path.name, output_mask.astype(silver_mask.dtype)
        )

    logging.info("Corrected gold truth segmentations using silver truth segmentations.")
