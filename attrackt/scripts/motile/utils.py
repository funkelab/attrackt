import json
import logging
from itertools import combinations
from pathlib import Path
from typing import List

import motile
import networkx as nx
import numpy as np
import tifffile
import zarr
from motile.constraints import MaxChildren, MaxParents, Pin
from motile.costs import Appear, Disappear
from motile.variables import EdgeSelected, NodeAppear, NodeDisappear, NodeSelected
from motile_toolbox.candidate_graph import EdgeAttr, NodeAttr
from natsort import natsorted
from traccuracy import EdgeFlag, NodeFlag, TrackingGraph

from attrackt.scripts.motile.costs import (
    EdgeDistanceHyper,
    EdgeDistanceRegular,
    EdgeEmbeddingDistanceHyper,
    EdgeEmbeddingDistanceRegular,
)

# Setup Logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[logging.StreamHandler()],
)
logger = logging.getLogger(__name__)


def create_dir(path):
    Path(path).mkdir(parents=True, exist_ok=True)
    logger.debug(f"Created directory: {path}")


def load_embeddings(file_list, index, results_dir_names):
    if not file_list:
        logger.debug("No embedding files provided.")
        return None
    selected_file = file_list[0] if len(file_list) == 1 else file_list[index]
    logger.info(f"Loaded embedding file: {selected_file}")
    return selected_file


def compute_mean_std(values):
    if values:
        mean_val, std_val = np.mean(values), np.std(values)
        logger.debug(f"Mean: {mean_val}, Std: {std_val}")
        return mean_val, std_val
    return None, None


def get_errors(tracks_predicted: TrackingGraph, tracks_GT: TrackingGraph):
    errors = {}
    errors["FP_DIV"] = []
    errors["FN_DIV"] = []
    errors["CTC_FALSE_POS"] = []
    errors["CTC_FALSE_NEG"] = []
    errors["WRONG_SEMANTIC"] = []

    FP_DIV = NodeFlag.FP_DIV
    FN_DIV = NodeFlag.FN_DIV

    CTC_FALSE_POS = EdgeFlag.CTC_FALSE_POS
    CTC_FALSE_NEG = EdgeFlag.CTC_FALSE_NEG
    WRONG_SEMANTIC = EdgeFlag.WRONG_SEMANTIC

    for node, node_data in tracks_predicted.graph.nodes(data=True):
        if FP_DIV in node_data:
            y = tracks_predicted.graph.nodes[node]["y"]
            x = tracks_predicted.graph.nodes[node]["x"]
            t = tracks_predicted.graph.nodes[node][NodeAttr.TIME.value]
            errors["FP_DIV"].append([node, t, y, x])

    for node, node_data in tracks_GT.graph.nodes(data=True):
        if FN_DIV in node_data:
            y = tracks_GT.graph.nodes[node]["y"]
            x = tracks_GT.graph.nodes[node]["x"]
            t = tracks_GT.graph.nodes[node][NodeAttr.TIME.value]
            errors["FN_DIV"].append([node, t, y, x])

    for edge, edge_data in tracks_predicted.graph.edges.items():
        if CTC_FALSE_POS in edge_data:
            node_in, node_out = edge
            y_in = tracks_predicted.graph.nodes[node_in]["y"]
            x_in = tracks_predicted.graph.nodes[node_in]["x"]
            t_in = tracks_predicted.graph.nodes[node_in][NodeAttr.TIME.value]
            y_out = tracks_predicted.graph.nodes[node_out]["y"]
            x_out = tracks_predicted.graph.nodes[node_out]["x"]

            t_out = tracks_predicted.graph.nodes[node_out][NodeAttr.TIME.value]
            errors["CTC_FALSE_POS"].append(
                [node_in, t_in, y_in, x_in, node_out, t_out, y_out, x_out]
            )

        if WRONG_SEMANTIC in edge_data:
            node_in, node_out = edge
            y_in = tracks_predicted.graph.nodes[node_in]["y"]
            x_in = tracks_predicted.graph.nodes[node_in]["x"]
            t_in = tracks_predicted.graph.nodes[node_in][NodeAttr.TIME.value]
            y_out = tracks_predicted.graph.nodes[node_out]["y"]
            x_out = tracks_predicted.graph.nodes[node_out]["x"]

            t_out = tracks_predicted.graph.nodes[node_out][NodeAttr.TIME.value]
            errors["WRONG_SEMANTIC"].append(
                [node_in, t_in, y_in, x_in, node_out, t_out, y_out, x_out]
            )

    for edge, edge_data in tracks_GT.graph.edges.items():
        if CTC_FALSE_NEG in edge_data:
            node_in, node_out = edge
            y_in = tracks_GT.graph.nodes[node_in]["y"]
            x_in = tracks_GT.graph.nodes[node_in]["x"]
            t_in = tracks_GT.graph.nodes[node_in][NodeAttr.TIME.value]
            y_out = tracks_GT.graph.nodes[node_out]["y"]
            x_out = tracks_GT.graph.nodes[node_out]["x"]
            t_out = tracks_GT.graph.nodes[node_out][NodeAttr.TIME.value]
            errors["CTC_FALSE_NEG"].append(
                [node_in, t_in, y_in, x_in, node_out, t_out, y_out, x_out]
            )

    return errors


def save_json(data, path):
    with open(path, "w") as f:
        json.dump(data, f)
    logger.info(f"Saved JSON: {path}")


def extract_weights(data, use_edge_distance, use_edge_embedding_distance):
    keys = [
        "Edge Distance Regular_weight",
        "Edge Distance Regular_constant",
        "Edge Distance Hyper_weight",
        "Edge Distance Hyper_constant",
        "Edge Embedding Distance Regular_weight",
        "Edge Embedding Distance Regular_constant",
        "Edge Embedding Distance Hyper_weight",
        "Edge Embedding Distance Hyper_constant",
        "Appear_weight",
        "Appear_constant",
        # "Disappear_weight",
        "Disappear_constant",
    ]

    if use_edge_distance and use_edge_embedding_distance:
        selected_keys = keys
    elif use_edge_distance and not use_edge_embedding_distance:
        selected_keys = [
            "Edge Distance Regular_weight",
            "Edge Distance Regular_constant",
            "Edge Distance Hyper_weight",
            "Edge Distance Hyper_constant",
            "Appear_weight",
            "Appear_constant",
            # "Disappear_weight",
            "Disappear_constant",
        ]
    elif not use_edge_distance and use_edge_embedding_distance:
        selected_keys = [
            "Edge Embedding Distance Regular_weight",
            "Edge Embedding Distance Regular_constant",
            "Edge Embedding Distance Hyper_weight",
            "Edge Embedding Distance Hyper_constant",
            "Appear_weight",
            "Appear_constant",
            # "Disappear_weight",
            "Disappear_constant",
        ]

    weights = []
    for k in selected_keys:
        weights.append(float(data[k]))
    weights = np.asarray(weights)
    logger.info(f"Extracted weights: {weights}")
    return weights


def log_stats(msg, **kwargs):
    logger.info(f"{'=' * 20}\n{msg}")
    for k, v in kwargs.items():
        logger.info(f"{k}: {v}")


def check_mutually_exclusive(args, key1, key2):
    key1_specified = args.get(key1)
    key2_specified = args.get(key2)

    if key1_specified and key2_specified:
        raise ValueError(f"Specify only ONE of '{key1}' or '{key2}', not both!")

    if not (key1_specified or key2_specified):
        raise ValueError(f"Specify exactly ONE of '{key1}' or '{key2}'!")


# def save_ilp_result_with_original_ids(solution_graph, results_dir_name, id_mapping):
def save_ilp_result_with_original_ids(solution_graph, sequence, id_mapping):
    results = []

    def extract_parts(node):
        if isinstance(node, tuple):
            return [x for t in node for x in t.split("_")]
        else:
            return node.split("_")

    def map_right(x):
        # Apply mapping to the second part (right of underscore)
        left, right = int(x[0]), int(x[1])
        return left, id_mapping.get(right, right)

    for u, v in solution_graph.edges:
        u_parts = extract_parts(u)
        v_parts = extract_parts(v)

        if isinstance(v, tuple):
            u0, u1 = map_right(u_parts[0:2])
            v0, v1 = map_right(v_parts[0:2])
            v2, v3 = map_right(v_parts[2:4])
            results.append([sequence, u1, u0, v1, v0])
            results.append([sequence, u1, u0, v3, v2])
        else:
            u0, u1 = map_right(u_parts)
            v0, v1 = map_right(v_parts)
            results.append([sequence, u1, u0, v1, v0])

    return results
    # output_path = Path(results_dir_name) / "jsons" / "ilp_original_ids.csv"
    # np.savetxt(
    #    output_path, np.asarray(results), delimiter=" ", fmt=["%i", "%i", "%i", "%i"]
    # )


def save_ilp_result(solution_graph, sequence):
    results = []
    for u, v in solution_graph.edges:
        u_parts = (
            [int(x) for x in u.split("_") if x.isdigit()]
            if not isinstance(u, tuple)
            else [int(x) for t in u for x in t.split("_") if x.isdigit()]
        )
        v_parts = (
            [int(x) for x in v.split("_") if x.isdigit()]
            if not isinstance(v, tuple)
            else [int(x) for t in v for x in t.split("_") if x.isdigit()]
        )
        if isinstance(v, tuple):
            results.append([sequence, u_parts[1], u_parts[0], v_parts[1], v_parts[0]])
            results.append([sequence, u_parts[1], u_parts[0], v_parts[3], v_parts[2]])
        else:
            results.append([sequence, u_parts[1], u_parts[0], v_parts[1], v_parts[0]])
    return results

    # np.savetxt(
    #    Path(results_dir_name) / "jsons" / "ilp.csv",
    #    np.asarray(results),
    #    delimiter=" ",
    #    fmt=["%i", "%i", "%i", "%i"],
    # )


def set_ground_truth_mask(solver: motile.Solver, gt_attribute: str = "gt"):
    """set_ground_truth_mask.
    This function tries to figure out which variables we have gt annotation
    for.

    """

    mask = np.zeros((solver.num_variables), dtype=np.float32)
    ground_truth = np.zeros_like(mask)
    # if nodes have `gt_attribute` specified, set mask and groundtruth for NodeSelected
    # variables.

    for node, index in solver.get_variables(NodeSelected).items():
        gt = solver.graph.nodes[node].get(gt_attribute, None)
        if gt is not None:
            mask[index] = 1.0
            ground_truth[index] = gt

    # if edges have `gt_attribute` specified, set mask and ground truth for
    # `EdgeSelected` variables.
    # IMPORTANT:
    # If groundtruth annotation value is 1.0, then we can also say
    # that mask and groundtruth for NodeDisappear for starting node
    # is known and mask and ground truth for Node Appear for ending
    # nodes is known.

    for edge, index in solver.get_variables(EdgeSelected).items():
        u, v = edge
        if isinstance(v, tuple):
            (u,) = u
            (v1, v2) = v
            index_v1_appear = solver.get_variables(NodeAppear)[v1]
            index_v2_appear = solver.get_variables(NodeAppear)[v2]
            index_u_disappear = solver.get_variables(NodeDisappear)[u]
            gt = solver.graph.edges[edge].get(gt_attribute, None)
            if gt is not None:
                mask[index] = 1.0
                ground_truth[index] = gt
                if gt == 1.0:
                    mask[index_u_disappear] = 1.0
                    ground_truth[index_u_disappear] = 0
                    mask[index_v1_appear] = 1.0
                    ground_truth[index_v1_appear] = 0
                    mask[index_v2_appear] = 1.0
                    ground_truth[index_v2_appear] = 0
        else:
            index_v_appear = solver.get_variables(NodeAppear)[v]
            index_u_disappear = solver.get_variables(NodeDisappear)[u]
            gt = solver.graph.edges[edge].get(gt_attribute, None)
            if gt is not None:
                mask[index] = 1.0
                ground_truth[index] = gt
                if gt == 1.0:
                    mask[index_u_disappear] = 1.0
                    ground_truth[index_u_disappear] = 0
                    mask[index_v_appear] = 1.0
                    ground_truth[index_v_appear] = 0

    return ground_truth, mask


def get_recursion_limit(candidate_graph):
    max_out = max(
        len(candidate_graph.out_edges(node)) for node in candidate_graph.nodes
    )
    max_in = max(len(candidate_graph.in_edges(node)) for node in candidate_graph.nodes)
    logger.info(f"Max out edges: {max_out}, max in edges: {max_in}")
    return max(max_out, max_in) + 500


def load_zarr_data(zarr_container_name, zarr_dataset_name, hypothesis_dim: bool = True):
    f = zarr.open(zarr_container_name)
    ds = f[zarr_dataset_name]  # C T Z Y X
    return np.transpose(ds, (1, 0, 2, 3, 4))


def load_tif_data(segmentation_dir_name: str, add_channel_axis=False):
    filenames = natsorted(Path(segmentation_dir_name).glob("*.tif"))
    segmentation = np.asarray([tifffile.imread(f) for f in filenames])
    return segmentation[:, np.newaxis] if add_channel_axis else segmentation


def add_hyper_edges(candidate_graph: nx.DiGraph):
    graph = candidate_graph.copy()
    for node in list(graph.nodes):
        out_edges = list(graph.out_edges(node))
        for e1, e2 in combinations(out_edges, 2):
            temp_node = f"{e1[0]}_{e1[1]}_{e2[1]}"
            graph.add_node(temp_node)
            graph.add_edge(e1[0], temp_node)
            graph.add_edge(temp_node, e1[1])
            graph.add_edge(temp_node, e2[1])
    return graph


def flip_edges(candidate_graph: nx.DiGraph):
    graph = nx.DiGraph()
    graph.add_nodes_from(candidate_graph.nodes(data=True))
    graph.add_edges_from(
        [(v, u, data) for u, v, data in candidate_graph.edges(data=True)]
    )
    return graph


def add_gt_edges_to_graph_from_json_file(
    groundtruth_graph: nx.DiGraph, json_file_name: str
):
    with open(json_file_name) as f:
        gt_data = json.load(f)
    parent_dict = {}
    for key, value in gt_data.items():
        parent_id = int(value[1])
        parent_dict.setdefault(parent_id, []).append(
            f"{int(np.min(value[0]))}_{int(key)}"
        )
    for key, (frames, parent_id) in gt_data.items():
        for i in range(len(frames) - 1):
            groundtruth_graph.add_edge(f"{frames[i]}_{key}", f"{frames[i + 1]}_{key}")
        if parent_id != 0:
            time_parent = int(np.max(gt_data[str(parent_id)][0]))
            parent_node = f"{time_parent}_{parent_id}"
            daughters = parent_dict[parent_id]
            if len(daughters) == 1:
                groundtruth_graph.add_edge(parent_node, daughters[0])
            else:
                temp_node = parent_node + "_" + "_".join(daughters)
                groundtruth_graph.add_node(temp_node)
                groundtruth_graph.add_edge(parent_node, temp_node)
                for d in daughters:
                    groundtruth_graph.add_edge(temp_node, d)
    return groundtruth_graph


def add_gt_edges_to_graph_from_test_array(
    groundtruth_graph: nx.DiGraph, gt_data: np.ndarray
):
    parent_dict, id_time_dict = {}, {}
    for row in gt_data:
        id_, t, parent_id = int(row[0]), int(row[1]), int(row[-1])
        if parent_id > 0:
            parent_dict.setdefault(parent_id, []).append(id_)
        id_time_dict[id_] = t

    for row in gt_data:
        id_, t, parent_id = int(row[0]), int(row[1]), int(row[-1])
        if parent_id > 0:
            parent_time = id_time_dict[parent_id]
            start_node = f"{parent_time}_{parent_id}"
            if len(parent_dict[parent_id]) == 1:
                end_node = f"{t}_{id_}"
                groundtruth_graph.add_edge(start_node, end_node)
            else:
                temp_node = (
                    start_node
                    + "_"
                    + "_".join([f"{t}_{d}" for d in parent_dict[parent_id]])
                )
                groundtruth_graph.add_node(temp_node)
                groundtruth_graph.add_edge(start_node, temp_node)
                for d in parent_dict[parent_id]:
                    groundtruth_graph.add_edge(temp_node, f"{t}_{d}")
    return groundtruth_graph


def add_costs(
    solver: motile.Solver,
    dT: int,
    use_edge_distance: bool,
    mean_regular_edge_distance=None,
    std_regular_edge_distance=None,
    mean_hyper_edge_distance=None,
    std_hyper_edge_distance=None,
    node_embedding_exists: bool = False,
    mean_regular_node_embedding_distance=None,
    std_regular_node_embedding_distance=None,
    mean_hyper_node_embedding_distance=None,
    std_hyper_node_embedding_distance=None,
    edge_embedding_exists: bool = False,
    mean_regular_edge_embedding_distance=None,
    std_regular_edge_embedding_distance=None,
    mean_hyper_edge_embedding_distance=None,
    std_hyper_edge_embedding_distance=None,
    use_different_weights_hyper=True,
):
    if use_edge_distance:
        solver.add_cost(
            EdgeDistanceRegular(
                weight=1.0,
                constant=0.0,
                attribute=NodeAttr.POS.value,
                mean_edge_distance=mean_regular_edge_distance,
                std_edge_distance=std_regular_edge_distance,
            ),
            name="Edge Distance Regular",
        )
        solver.add_cost(
            EdgeDistanceHyper(
                weight=1.0,
                constant=0.0,
                attribute=NodeAttr.POS.value,
                mean_edge_distance=mean_hyper_edge_distance,
                std_edge_distance=std_hyper_edge_distance,
            ),
            name="Edge Distance Hyper",
        )

    if edge_embedding_exists:
        solver.add_cost(
            EdgeEmbeddingDistanceRegular(
                weight=1.0,
                constant=0.0,
                attribute=EdgeAttr.EDGE_EMBEDDING.value,
                mean_edge_distance=mean_regular_edge_embedding_distance,
                std_edge_distance=std_regular_edge_embedding_distance,
            ),
            name="Edge Embedding Distance Regular",
        )
        solver.add_cost(
            EdgeEmbeddingDistanceHyper(
                weight=1.0,
                constant=0.0,
                attribute=EdgeAttr.EDGE_EMBEDDING.value,
                mean_edge_distance=mean_hyper_edge_embedding_distance,
                std_edge_distance=std_hyper_edge_embedding_distance,
            ),
            name="Edge Embedding Distance Hyper",
        )

    if dT > 1:
        pass
        # solver.add_cost(
        #    TimeGap(weight=1.0, constant=0.0, time_attribute=NodeAttr.TIME.value),
        #    name="Time Gap",
        # )

    solver.add_cost(
        Appear(weight=0.0, constant=1.0, ignore_attribute="ignore_appear_cost")
    )
    solver.add_cost(Disappear(constant=1.0, ignore_attribute="ignore_disappear_cost"))
    return solver


def add_constraints(solver: motile.Solver, pin_nodes: bool, pin_edges: bool):
    solver.add_constraint(MaxParents(1))
    solver.add_constraint(MaxChildren(1))
    if pin_nodes:
        solver.add_constraint(Pin(attribute=NodeAttr.PINNED.value))
    if pin_edges:
        solver.add_constraint(Pin(attribute=EdgeAttr.PINNED.value))
    return solver


def add_app_disapp_attributes(track_graph: motile.TrackGraph, t_min, t_max):
    prev_nodes = {
        t: len(track_graph.nodes_by_frame(t - 1)) if t != t_min else 0
        for t in range(t_min, t_max + 1)
    }
    next_nodes = {
        t: len(track_graph.nodes_by_frame(t + 1)) if t != t_max else 0
        for t in range(t_min, t_max + 1)
    }
    curr_nodes = {
        t: len(track_graph.nodes_by_frame(t)) for t in range(t_min, t_max + 1)
    }

    for node, attrs in track_graph.nodes.items():
        t = attrs[NodeAttr.TIME.value]
        if prev_nodes[t] == 0 and curr_nodes[t] != 0:
            track_graph.nodes[node][NodeAttr.IGNORE_APPEAR_COST.value] = True
        if next_nodes[t] == 0 and curr_nodes[t] != 0:
            track_graph.nodes[node][NodeAttr.IGNORE_DISAPPEAR_COST.value] = True
    return track_graph


def expand_position(
    data: np.ndarray,
    position: List,
    id_: int,
    nhood: int = 1,
):
    outside = True
    if len(position) == 2:
        H, W = data.shape
        y, x = position
        y, x = int(y), int(x)
        while outside:
            data_ = data[
                np.maximum(y - nhood, 0) : np.minimum(y + nhood + 1, H),
                np.maximum(x - nhood, 0) : np.minimum(x + nhood + 1, W),
            ]
            if 0 in data_.shape:
                nhood += 1
            else:
                outside = False
        data[
            np.maximum(y - nhood, 0) : np.minimum(y + nhood + 1, H),
            np.maximum(x - nhood, 0) : np.minimum(x + nhood + 1, W),
        ] = id_
    elif len(position) == 3:
        D, H, W = data.shape
        z, y, x = position
        z, y, x = int(z), int(y), int(x)
        while outside:
            data_ = data[
                np.maximum(z - nhood, 0) : np.minimum(z + nhood + 1, D),
                np.maximum(y - nhood, 0) : np.minimum(y + nhood + 1, H),
                np.maximum(x - nhood, 0) : np.minimum(x + nhood + 1, W),
            ]
            if 0 in data_.shape:
                nhood += 1
            else:
                outside = False
        data[
            np.maximum(z - nhood, 0) : np.minimum(z + nhood + 1, D),
            np.maximum(y - nhood, 0) : np.minimum(y + nhood + 1, H),
            np.maximum(x - nhood, 0) : np.minimum(x + nhood + 1, W),
        ] = id_
    return data
