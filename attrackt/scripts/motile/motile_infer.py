import json
import logging
import sys
from pathlib import Path

import jsonargparse
import networkx as nx
import numpy as np
from attrackt.scripts.motile.run_traccuracy import compute_metrics
from attrackt.scripts.motile.saving_utils import save_result
from attrackt.scripts.motile.utils import (
    add_app_disapp_attributes,
    add_constraints,
    add_costs,
    add_gt_edges_to_graph_from_test_array,
    add_hyper_edges,
    check_mutually_exclusive,
    compute_mean_std,
    create_dir,
    extract_weights,
    flip_edges,
    get_recursion_limit,
    log_stats,
    save_ilp_result,
)
from motile import Solver, TrackGraph
from motile_toolbox.candidate_graph import (
    EdgeAttr,
    NodeAttr,
    get_candidate_graph_from_points_list,
    graph_to_nx,
)
from yaml import safe_load

from attrackt.scripts import (
    load_csv_associations,
    load_csv_data,
    load_csv_embeddings,
    load_csv_ilp_result,
)

# Setup Logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[logging.StreamHandler()],
)
logger = logging.getLogger(__name__)


def motile_infer(args):
    check_mutually_exclusive(args, "max_edge_distance", "num_nearest_neighbours")
    test_csv_file_name = args.get("test_csv_file_name")
    sequence_names = args.get("sequence_names")
    supervised_csv = args.get("supervised_csv")
    result_dir_name = args.get("result_dir_name")
    verbose = args.get("verbose", True)
    results = []
    for index in range(len(sequence_names)):
        sequence = sequence_names[index]
        logger.info(f"Processing sequence: {sequence}")
        result_dir = Path(result_dir_name)
        create_dir(result_dir / sequence)

        # Load CSV and build candidate graph
        test_array, _, test_mapping, test_reverse_mapping = load_csv_data(
            test_csv_file_name,
            voxel_size=args.get("voxel_size"),
            sequences=[sequence],
        )
        t_min, t_max = int(np.min(test_array[:, 1])), int(np.max(test_array[:, 1]))
        candidate_graph, *_ = get_candidate_graph_from_points_list(
            points_list=test_array,
            max_edge_distance=args.get("max_edge_distance"),
            num_nearest_neighbours=args.get("num_nearest_neighbours"),
            direction_candidate_graph=args.get("direction_candidate_graph"),
            dT=args.get("dT", 1),
        )

        if args.get("direction_candidate_graph") == "backward":
            logger.info("Flipping edges (backward direction)")
            candidate_graph = flip_edges(candidate_graph)

        log_stats(
            "Candidate Graph Stats",
            num_nodes=len(candidate_graph.nodes),
            num_edges=len(candidate_graph.edges),
        )

        # Hyper edges
        candidate_graph = add_hyper_edges(candidate_graph)
        track_graph = TrackGraph(nx_graph=candidate_graph, frame_attribute="time")

        log_stats(
            "Track Graph Stats",
            num_nodes=len(track_graph.nodes),
            num_edges=len(track_graph.edges),
        )

        regular_edge_distances = []
        hyper_edge_distances = []
        for edge in track_graph.edges:
            if isinstance(edge[1], tuple):
                (u,) = edge[0]
                (v1, v2) = edge[1]
                u_pos = np.array(track_graph.nodes[u][NodeAttr.POS.value])
                v1_pos = np.array(track_graph.nodes[v1][NodeAttr.POS.value])
                v2_pos = np.array(track_graph.nodes[v2][NodeAttr.POS.value])
                d = np.linalg.norm(u_pos - 0.5 * (v1_pos + v2_pos))
                hyper_edge_distances.append(d)
            else:
                (u, v) = edge
                u_pos = np.array(track_graph.nodes[u][NodeAttr.POS.value])
                v_pos = np.array(track_graph.nodes[v][NodeAttr.POS.value])
                d = np.linalg.norm(u_pos - v_pos)
                regular_edge_distances.append(d)

        mean_regular_edge_dist, std_regular_edge_dist = compute_mean_std(
            regular_edge_distances
        )

        mean_hyper_edge_dist, std_hyper_edge_dist = compute_mean_std(
            hyper_edge_distances
        )

        log_stats(
            "Candidate Graph Stats",
            mean_regular_edge_distance=mean_regular_edge_dist,
            std_regular_edge_distance=std_regular_edge_dist,
            mean_hyper_edge_distance=mean_hyper_edge_dist,
            std_hyper_edge_distance=std_hyper_edge_dist,
        )

        node_embedding_file_name = args.get("node_embedding_file_name")
        edge_embedding_file_name = args.get("edge_embedding_file_name")

        mean_regular_edge_embedding_dist, std_regular_edge_embedding_dist = None, None
        mean_hyper_edge_embedding_dist, std_hyper_edge_embedding_dist = None, None
        regular_edge_embedding_distances, hyper_edge_embedding_distances = [], []

        if node_embedding_file_name is not None:
            node_embedding_data = load_csv_embeddings(
                node_embedding_file_name, sequence
            )

            # make (N, 64) float embeddings
            emb_cols = [f"emb_{i}" for i in range(64)]
            emb_matrix = np.column_stack(
                [node_embedding_data[c] for c in emb_cols]
            ).astype(np.float32)

            for (id_, t), emb in zip(
                zip(node_embedding_data["id"], node_embedding_data["t"]),
                emb_matrix,
            ):
                node_id = f"{t}_{id_}"
                if node_id in track_graph.nodes:
                    track_graph.nodes[node_id][NodeAttr.NODE_EMBEDDING.value] = emb

            for edge in track_graph.edges:
                if isinstance(edge[1], tuple):
                    (u,) = edge[0]
                    (v1, v2) = edge[1]
                    u_embedding = np.array(
                        track_graph.nodes[u][NodeAttr.NODE_EMBEDDING.value]
                    )
                    v1_embedding = np.array(
                        track_graph.nodes[v1][NodeAttr.NODE_EMBEDDING.value]
                    )
                    v2_embedding = np.array(
                        track_graph.nodes[v2][NodeAttr.NODE_EMBEDDING.value]
                    )
                    d = np.linalg.norm(
                        u_embedding - 0.5 * (v1_embedding + v2_embedding)
                    )
                    track_graph.edges[edge][EdgeAttr.EDGE_EMBEDDING.value] = d
                    hyper_edge_embedding_distances.append(d)
                else:
                    (u, v) = edge
                    u_embedding = np.array(
                        track_graph.nodes[u][NodeAttr.NODE_EMBEDDING.value]
                    )
                    v_embedding = np.array(
                        track_graph.nodes[v][NodeAttr.NODE_EMBEDDING.value]
                    )
                    d = np.linalg.norm(u_embedding - v_embedding)
                    track_graph.edges[edge][EdgeAttr.EDGE_EMBEDDING.value] = d
                    regular_edge_embedding_distances.append(d)

            mean_regular_edge_embedding_dist, std_regular_edge_embedding_dist = (
                compute_mean_std(regular_edge_embedding_distances)
            )
            mean_hyper_edge_embedding_dist, std_hyper_edge_embedding_dist = (
                compute_mean_std(hyper_edge_embedding_distances)
            )
            log_stats(
                "Candidate Graph Stats",
                mean_regular_edge_embedding_distance=mean_regular_edge_embedding_dist,
                std_regular_edge_embedding_distance=std_regular_edge_embedding_dist,
                mean_hyper_edge_embedding_distance=mean_hyper_edge_embedding_dist,
                std_hyper_edge_embedding_distance=std_hyper_edge_embedding_dist,
            )

        elif edge_embedding_file_name is not None:
            edge_embedding_data = load_csv_associations(
                edge_embedding_file_name, sequence
            )
            embedding_dictionary = {}
            for id_a, t_a, id_b, t_b, weight in zip(
                edge_embedding_data["id_previous"],
                edge_embedding_data["t_previous"],
                edge_embedding_data["id_current"],
                edge_embedding_data["t_current"],
                edge_embedding_data["weight"],
            ):
                id_a, t_a, id_b, t_b, weight = (
                    int(id_a),
                    int(t_a),
                    int(id_b),
                    int(t_b),
                    float(weight),
                )

                node_a = str(t_a) + "_" + str(id_a)
                node_b = str(t_b) + "_" + str(id_b)
                if node_a in embedding_dictionary:
                    pass
                else:
                    embedding_dictionary[node_a] = {}
                embedding_dictionary[node_a].update({node_b: weight})

            for edge in track_graph.edges:
                if isinstance(edge[1], tuple):
                    (u,) = edge[0]
                    (v1, v2) = edge[1]
                    u_v1 = embedding_dictionary.get(u, {}).get(v1, 0.0)
                    u_v2 = embedding_dictionary.get(u, {}).get(v2, 0.0)
                    d = 0.5 * (u_v1 + u_v2)
                    track_graph.edges[edge][EdgeAttr.EDGE_EMBEDDING.value] = d
                    hyper_edge_embedding_distances.append(d)
                else:
                    (u, v) = edge
                    d = embedding_dictionary.get(u, {}).get(v, 0.0)
                    track_graph.edges[edge][EdgeAttr.EDGE_EMBEDDING.value] = d
                    regular_edge_embedding_distances.append(d)

            mean_regular_edge_embedding_dist, std_regular_edge_embedding_dist = (
                compute_mean_std(regular_edge_embedding_distances)
            )

            mean_hyper_edge_embedding_dist, std_hyper_edge_embedding_dist = (
                compute_mean_std(hyper_edge_embedding_distances)
            )

            log_stats(
                "Candidate Graph Stats",
                mean_regular_edge_embedding_distance=mean_regular_edge_embedding_dist,
                std_regular_edge_embedding_distance=std_regular_edge_embedding_dist,
                mean_hyper_edge_embedding_distance=mean_hyper_edge_embedding_dist,
                std_hyper_edge_embedding_distance=std_hyper_edge_embedding_dist,
            )

        track_graph = add_app_disapp_attributes(track_graph, t_min, t_max)

        # In case, supervision is available:
        if args.get("supervised_csv") is not None:
            supervised_data = load_csv_ilp_result(supervised_csv, sequences=[sequence])
            supervised_dictionary = {}
            for id_previous, t_previous, id_current, t_current in zip(
                supervised_data["id_previous"],
                supervised_data["t_previous"],
                supervised_data["id_current"],
                supervised_data["t_current"],
            ):
                u = str(t_previous) + "_" + str(id_previous)
                v = str(t_current) + "_" + str(id_current)

                if u in supervised_dictionary.keys():
                    pass
                else:
                    supervised_dictionary[u] = []
                supervised_dictionary[u].append(v)
            logger.info(
                f"Number of keys in supervised dictionary is {len(supervised_dictionary)}."
            )
            for u, values in supervised_dictionary.items():
                outgoing_edges_u = track_graph.next_edges[u]
                for outgoing_edge_u in outgoing_edges_u:
                    track_graph.edges[outgoing_edge_u][EdgeAttr.PINNED.value] = False

            for u, values in supervised_dictionary.items():
                if len(values) == 1:
                    v1 = values[0]
                    edge = (u, v1)
                    if edge in track_graph.edges:
                        track_graph.edges[edge][EdgeAttr.PINNED.value] = True
                    else:
                        logger.info(f"++++++edge {edge} not found in graph.++++++")
                elif len(values) == 2:
                    v1, v2 = values[0], values[1]
                    edge1 = ((u,), (v1, v2))
                    edge2 = ((u,), (v2, v1))
                    if edge1 in track_graph.edges:
                        track_graph.edges[edge1][EdgeAttr.PINNED.value] = True
                    elif edge2 in track_graph.edges:
                        track_graph.edges[edge2][EdgeAttr.PINNED.value] = True
                    else:
                        logger.info(
                            f"++++++edge {edge1} or {edge2} not found in graph.++++++"
                        )
        # Constraints and costs
        recursion_limit = get_recursion_limit(candidate_graph)
        if recursion_limit > 1000:
            logger.info(f"Increasing recursion limit to: {recursion_limit}")
            sys.setrecursionlimit(recursion_limit)

        solver = Solver(track_graph)
        solver = add_costs(
            solver,
            dT=args.get("dT", 1),
            use_edge_distance=args.get("use_edge_distance"),
            mean_regular_edge_distance=mean_regular_edge_dist,
            std_regular_edge_distance=std_regular_edge_dist,
            mean_hyper_edge_distance=mean_hyper_edge_dist,
            std_hyper_edge_distance=std_hyper_edge_dist,
            edge_embedding_exists=args.get("edge_embedding_exists"),
            mean_regular_edge_embedding_distance=mean_regular_edge_embedding_dist,
            std_regular_edge_embedding_distance=std_regular_edge_embedding_dist,
            mean_hyper_edge_embedding_distance=mean_hyper_edge_embedding_dist,
            std_hyper_edge_embedding_distance=std_hyper_edge_embedding_dist,
        )
        solver = add_constraints(
            solver, pin_nodes=args.get("pin_nodes"), pin_edges=args.get("pin_edges")
        )
        weights_file_path = args.get("weights_file_path")
        if weights_file_path is None:
            weight_data = {
                "Edge Distance Regular_weight": 1,
                "Edge Distance Regular_constant": 0,
                "Edge Distance Hyper_weight": 1,
                "Edge Distance Hyper_constant": 0,
                "Edge Embedding Distance Regular_weight": -1
                if args.get("embedding_type", "affinity") == "affinity"
                else 1,
                "Edge Embedding Distance Regular_constant": 0,
                "Edge Embedding Distance Hyper_weight": -1
                if args.get("embedding_type", "affinity") == "affinity"
                else 1,
                "Edge Embedding Distance Hyper_constant": 0,
                "Appear_weight": 0,
                "Appear_constant": 1,
                "Disappear_weight": 0,
                "Disappear_constant": 1,
            }
        else:
            with open(weights_file_path, "r") as f:
                weight_data = json.load(f)

        weights = extract_weights(
            data=weight_data,
            use_edge_distance=args.get("use_edge_distance"),
            use_edge_embedding_distance=args.get("edge_embedding_exists"),
        )
        solver.weights.from_ndarray(weights)

        logger.info("Solving ILP optimization...")
        solution = solver.solve(verbose=verbose)
        solution_graph = solver.get_selected_subgraph(solution)

        results.extend(save_ilp_result(solution_graph, sequence))

        log_stats(
            "Optimization Result",
            selected_nodes=len(solution_graph.nodes),
            selected_edges=len(solution_graph.edges),
        )
        segmentation_shape = (
            args.get("segmentation_shapes")[index]
            if args.get("segmentation_shapes") is not None
            else None
        )

        # Save & evaluate
        logger.info("Saving result and computing metrics...")
        _, _, _, tracked_graph = save_result(
            solution_nx_graph=graph_to_nx(solution_graph),
            segmentation_shape=segmentation_shape,
            output_tif_dir_name=result_dir,
            write_tifs=args.get("write_tifs"),
        )

        gt_graph = nx.DiGraph()
        gt_graph.add_nodes_from(candidate_graph.nodes(data=True))

        gt_graph = add_gt_edges_to_graph_from_test_array(
            groundtruth_graph=gt_graph, gt_data=test_array
        )

        gt_track_graph = TrackGraph(nx_graph=gt_graph, frame_attribute="time")

        compute_metrics(
            gt_segmentation=None,
            gt_nx_graph=graph_to_nx(gt_track_graph),
            sequence=sequence,
            pred_segmentation=None,
            pred_nx_graph=tracked_graph,
            result_dir=result_dir,
        )
        logger.info(f"Finished processing: {sequence_names[index]}")

    # save results to csv file with space delimiter using
    output_csv_file_name = Path(result_dir) / "tracking_results.csv"
    if output_csv_file_name.exists():
        output_csv_file_name.unlink()

    header = ["# sequence", "id_previous", "t_previous", "id_current", "t_current"]

    with open(output_csv_file_name, "w") as f:
        f.write(" ".join(header) + "\n")
        for row in results:
            sequence, id_previous, time_previous, id_current, time_current = row
            f.write(
                f"{sequence} {id_previous} {time_previous} {id_current} {time_current}\n"
            )


def main():
    parser = jsonargparse.ArgumentParser()
    parser.add_argument("--yaml_config_file_name", dest="yaml_config_file_name")
    logger.info(f"Loading YAML config: {parser.parse_args().yaml_config_file_name}")
    args = safe_load(open(parser.parse_args().yaml_config_file_name))
    logger.info(f"Parsed config: {json.dumps(args, indent=2)}")
    motile_infer(args)


if __name__ == "__main__":
    main()
