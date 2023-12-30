import argparse
import json
import logging
from pathlib import Path
from pprint import pformat
from typing import Optional

import networkx as nx
import numpy as np
from traccuracy import TrackingGraph, run_metrics
from traccuracy.matchers import PointMatcher
from traccuracy.metrics import CTCMetrics, DivisionMetrics

from attrackt.scripts.motile.utils import get_errors

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[logging.StreamHandler()],
)
logger = logging.getLogger(__name__)


def compute_metrics(
    gt_segmentation: Optional[np.ndarray],
    pred_segmentation: Optional[np.ndarray],
    result_dir: Path,
    sequence: str,
    gt_nx_graph: Optional[nx.DiGraph] = None,
    pred_nx_graph: Optional[nx.DiGraph] = None,
) -> dict:
    """
    Computes CTC and Division metrics between ground truth and predicted data.

    Returns:
        dict: Results of metrics computation.
    """
    # Load ground truth data
    if gt_nx_graph:
        logger.info("Using ground truth NetworkX graph...")
        gt_data = TrackingGraph(
            graph=gt_nx_graph,
            segmentation=None,
            name="gt",
            frame_key="time",
            location_keys=("y", "x"),
        )
    else:
        raise ValueError("gt_nx_graph must be provided for ground truth.")

    # Load predicted data
    if pred_nx_graph:
        logger.info("Using predicted NetworkX graph...")
        pred_data = TrackingGraph(
            graph=pred_nx_graph,
            segmentation=None,
            name="pred",
            frame_key="time",
            location_keys=("y", "x"),
        )
    else:
        raise ValueError("pred_nx_graph  must be provided for prediction.")

    # Compute metrics
    logger.info("Running CTC and Division metrics...")
    ctc_results, _ = run_metrics(
        gt_data=gt_data,
        pred_data=pred_data,
        matcher=PointMatcher(threshold=10),
        metrics=[CTCMetrics(), DivisionMetrics()],
    )

    errors = get_errors(pred_data, gt_data)
    # this above can be used if you wish to visualize the kind of errors
    # obtained.
    # logger.info(pformat(errors))

    logger.info("Results:")
    logger.info(pformat(ctc_results))

    # Save results
    results_path = result_dir / sequence / "results.json"
    results_path.parent.mkdir(parents=True, exist_ok=True)
    with open(results_path, "w") as f:
        json.dump(ctc_results, f)
    logger.info(f"Results saved to: {results_path}")

    return ctc_results


def parse_args():
    parser = argparse.ArgumentParser(description="Compute tracking accuracy metrics.")
    parser.add_argument(
        "--gt_segmentation",
        type=Path,
        help="Ground truth segmentation file path (npz/npy)",
    )
    parser.add_argument(
        "--pred_segmentation",
        type=Path,
        help="Predicted segmentation file path (npz/npy)",
    )
    parser.add_argument(
        "--results_dir", type=Path, required=True, help="Directory to store results"
    )
    return parser.parse_args()


def main():
    args = parse_args()

    # Load segmentations
    gt_seg = np.load(args.gt_segmentation) if args.gt_segmentation else None
    pred_seg = np.load(args.pred_segmentation) if args.pred_segmentation else None

    compute_metrics(
        gt_segmentation=gt_seg,
        pred_segmentation=pred_seg,
        results_dir=args.results_dir,
    )


if __name__ == "__main__":
    main()
