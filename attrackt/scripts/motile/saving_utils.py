from pathlib import Path

import networkx as nx
import numpy as np
import tifffile
from motile_toolbox.candidate_graph import NodeAttr
from tqdm import tqdm

from attrackt.scripts.motile.utils import expand_position


def save_result(
    solution_nx_graph: nx.DiGraph,
    segmentation_shape,
    output_tif_dir_name: str,
    write_tifs: bool = False,
):
    output_path = Path(output_tif_dir_name)
    # output_json_path = output_path / "jsons"
    # output_json_path.mkdir(parents=True, exist_ok=True)

    tracked_masks = (
        np.zeros(segmentation_shape, dtype=np.uint16)
        if segmentation_shape is not None
        else None
    )
    new_mapping = {}
    res_track = {}
    id_counter = 1

    def update_mask(t, pos, track_id):
        if tracked_masks is not None:
            tracked_masks[t] = expand_position(tracked_masks[t], pos, track_id)

    # Process edges
    for in_node, out_node in tqdm(solution_nx_graph.edges()):
        t_in, id_in = map(int, in_node.split("_"))
        t_out, id_out = map(int, out_node.split("_"))
        num_out_edges = len(solution_nx_graph.out_edges(in_node))

        if num_out_edges == 1:  # Continuation
            if in_node in new_mapping:
                track_id = new_mapping[in_node]
                if t_out not in res_track[track_id][0]:
                    res_track[track_id][0].append(t_out)
                update_mask(
                    t_in, solution_nx_graph.nodes[in_node][NodeAttr.POS.value], track_id
                )
                update_mask(
                    t_out,
                    solution_nx_graph.nodes[out_node][NodeAttr.POS.value],
                    track_id,
                )
                new_mapping[out_node] = track_id
            else:  # New track start
                res_track[id_counter] = ([t_in, t_out], 0)
                update_mask(
                    t_in,
                    solution_nx_graph.nodes[in_node][NodeAttr.POS.value],
                    id_counter,
                )
                update_mask(
                    t_out,
                    solution_nx_graph.nodes[out_node][NodeAttr.POS.value],
                    id_counter,
                )
                new_mapping[in_node] = new_mapping[out_node] = id_counter
                id_counter += 1
        elif num_out_edges == 2:  # Split
            out_edges = list(solution_nx_graph.out_edges(in_node))
            track_id = new_mapping.get(in_node, id_counter)
            if in_node not in new_mapping:
                res_track[track_id] = ([t_in], 0)
                update_mask(
                    t_in, solution_nx_graph.nodes[in_node][NodeAttr.POS.value], track_id
                )
                new_mapping[in_node] = track_id
                id_counter += 1

            for edge in out_edges:
                _, out_node = edge
                t_out_n, id_out_n = map(int, out_node.split("_"))
                if out_node not in new_mapping:
                    update_mask(
                        t_out_n,
                        solution_nx_graph.nodes[out_node][NodeAttr.POS.value],
                        id_counter,
                    )
                    res_track[id_counter] = ([t_out_n], new_mapping[in_node])
                    new_mapping[out_node] = id_counter
                    id_counter += 1

    # Handle nodes without edges
    for node in solution_nx_graph.nodes:
        if node not in new_mapping:
            t, _ = map(int, node.split("_"))
            res_track[id_counter] = ([t], 0)
            update_mask(
                t, solution_nx_graph.nodes[node][NodeAttr.POS.value], id_counter
            )
            new_mapping[node] = id_counter
            id_counter += 1

    # Clean tif directory
    for tif_file in output_path.glob("*.tif"):
        tif_file.unlink()

    if write_tifs and tracked_masks is not None:
        for i in range(tracked_masks.shape[0]):
            tifffile.imwrite(
                output_path / f"mask{str(i).zfill(3)}.tif",
                tracked_masks[i].astype(np.uint16),
            )

    # Save tracking results
    # with open(output_json_path / "res_track.json", "w") as f:
    #   json.dump(res_track, f)

    # with open(output_json_path / "tracklet.json", "w") as f:
    #    json.dump(new_mapping, f)

    # Also save as space-delimited TXT
    # txt_path = output_json_path / "res_track.txt"
    # with open(txt_path, "w") as f:
    #    for track_id, (times, parent_id) in res_track.items():
    #        t_start = min(times)
    #        t_end = max(times)
    #        f.write(f"{track_id} {t_start} {t_end} {parent_id}\n")

    # Build tracked graph
    tracked_graph = nx.DiGraph()
    for node, track_id in new_mapping.items():
        t, _ = map(int, node.split("_"))
        pos = solution_nx_graph.nodes[node]["pos"]
        attrs = {"seg_id": track_id, "time": t}
        if len(pos) == 2:
            attrs.update({"y": pos[0], "x": pos[1]})
        elif len(pos) == 3:
            attrs.update({"z": pos[0], "y": pos[1], "x": pos[2]})
        tracked_graph.add_node(node, **attrs)

    tracked_graph.add_edges_from(solution_nx_graph.edges)

    return new_mapping, res_track, tracked_masks, tracked_graph
