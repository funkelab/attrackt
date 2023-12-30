from typing import cast

import networkx as nx
import numpy as np
from motile.costs import Cost, Weight
from motile.solver import Solver
from motile.variables import EdgeSelected


class EdgeDistanceRegular(Cost):
    def __init__(
        self,
        attribute,
        weight,
        constant,
        mean_edge_distance=None,
        std_edge_distance=None,
    ):
        self.attribute = attribute
        self.weight = Weight(weight)
        self.constant = Weight(constant)
        self.mean_edge_distance = mean_edge_distance
        self.std_edge_distance = std_edge_distance

    def apply(self, solver: Solver) -> None:
        edge_variables = solver.get_variables(EdgeSelected)
        for key, index in edge_variables.items():
            if isinstance(key[1], tuple):
                solver.add_variable_cost(index, 0.0, self.weight)
                solver.add_variable_cost(index, 0.0, self.constant)
            else:
                u, v = cast("tuple[int, int]", key)
                pos_u = self.__get_node_position(solver.graph, u)
                pos_v = self.__get_node_position(solver.graph, v)
                feature = np.linalg.norm(pos_u - pos_v)
                if (
                    self.mean_edge_distance is not None
                    and self.std_edge_distance is not None
                ):
                    feature = (
                        feature - self.mean_edge_distance
                    ) / self.std_edge_distance
                solver.add_variable_cost(index, feature, self.weight)
                solver.add_variable_cost(index, 1.0, self.constant)

    def __get_node_position(self, graph: nx.DiGraph, node: int) -> np.ndarray:
        if isinstance(self.attribute, tuple):
            return np.array([graph.nodes[node][p] for p in self.attribute])
        return np.array(graph.nodes[node][self.attribute])


class EdgeDistanceHyper(Cost):
    def __init__(
        self,
        attribute,
        weight,
        constant,
        mean_edge_distance=None,
        std_edge_distance=None,
    ):
        self.attribute = attribute
        self.weight = Weight(weight)
        self.constant = Weight(constant)
        self.mean_edge_distance = mean_edge_distance
        self.std_edge_distance = std_edge_distance

    def apply(self, solver: Solver) -> None:
        edge_variables = solver.get_variables(EdgeSelected)
        for key, index in edge_variables.items():
            if isinstance(key[1], tuple):
                (start,) = key[0]
                end1, end2 = key[1]
                pos_start = self.__get_node_position(solver.graph, start)
                pos_end1 = self.__get_node_position(solver.graph, end1)
                pos_end2 = self.__get_node_position(solver.graph, end2)
                feature = np.linalg.norm(pos_start - 0.5 * (pos_end1 + pos_end2))
                if (
                    self.mean_edge_distance is not None
                    and self.std_edge_distance is not None
                ):
                    feature = (
                        feature - self.mean_edge_distance
                    ) / self.std_edge_distance
                solver.add_variable_cost(index, feature, self.weight)
                solver.add_variable_cost(index, 1.0, self.constant)
            else:
                solver.add_variable_cost(index, 0.0, self.weight)
                solver.add_variable_cost(index, 0.0, self.constant)

    def __get_node_position(self, graph: nx.DiGraph, node: int) -> np.ndarray:
        if isinstance(self.attribute, tuple):
            return np.array([graph.nodes[node][p] for p in self.attribute])
        return np.array(graph.nodes[node][self.attribute])


class EdgeEmbeddingDistanceRegular(Cost):
    def __init__(
        self,
        attribute,
        weight,
        constant,
        mean_edge_distance=None,
        std_edge_distance=None,
    ):
        self.attribute = attribute
        self.weight = Weight(weight)
        self.constant = Weight(constant)
        self.mean_edge_distance = mean_edge_distance
        self.std_edge_distance = std_edge_distance

    def apply(self, solver: Solver) -> None:
        edge_variables = solver.get_variables(EdgeSelected)
        for key, index in edge_variables.items():
            if isinstance(key[1], tuple):
                solver.add_variable_cost(index, 0.0, self.weight)
                solver.add_variable_cost(index, 0.0, self.constant)
            else:
                feature = self.__get_edge_embedding(solver.graph, key)
                if (
                    self.mean_edge_distance is not None
                    and self.std_edge_distance is not None
                ):
                    feature = (
                        feature - self.mean_edge_distance
                    ) / self.std_edge_distance
                solver.add_variable_cost(index, feature, self.weight)
                solver.add_variable_cost(index, 1.0, self.constant)

    def __get_edge_embedding(self, graph, edge):
        if self.attribute in graph.edges[edge]:
            return np.array([graph.edges[edge][self.attribute]])
        print(f"Edge attribute not found for edge {edge}. Setting to 0.001")
        return np.array([0.001])


class EdgeEmbeddingDistanceHyper(Cost):
    def __init__(
        self,
        attribute,
        weight,
        constant,
        mean_edge_distance=None,
        std_edge_distance=None,
    ):
        self.attribute = attribute
        self.weight = Weight(weight)
        self.constant = Weight(constant)
        self.mean_edge_distance = mean_edge_distance
        self.std_edge_distance = std_edge_distance

    def apply(self, solver: Solver) -> None:
        edge_variables = solver.get_variables(EdgeSelected)
        for key, index in edge_variables.items():
            if isinstance(key[1], tuple):
                feature = self.__get_edge_embedding(solver.graph, key)
                if (
                    self.mean_edge_distance is not None
                    and self.std_edge_distance is not None
                ):
                    feature = (
                        feature - self.mean_edge_distance
                    ) / self.std_edge_distance
                solver.add_variable_cost(index, feature, self.weight)
                solver.add_variable_cost(index, 1.0, self.constant)
            else:
                solver.add_variable_cost(index, 0.0, self.weight)
                solver.add_variable_cost(index, 0.0, self.constant)

    def __get_edge_embedding(self, graph, edge):
        if self.attribute in graph.edges[edge]:
            return np.array([graph.edges[edge][self.attribute]])
        print(f"Edge attribute not found for edge {edge}. Setting to 0.001")
        return np.array([0.001])


class NodeEmbeddingDistanceRegular(Cost):
    def __init__(
        self,
        node_embedding_attribute,
        weight,
        constant,
        mean_node_embedding_distance=None,
        std_node_embedding_distance=None,
    ):
        self.node_embedding_attribute = node_embedding_attribute
        self.weight = Weight(weight)
        self.constant = Weight(constant)
        self.mean_node_embedding_distance = mean_node_embedding_distance
        self.std_node_embedding_distance = std_node_embedding_distance

    def apply(self, solver: Solver) -> None:
        edge_variables = solver.get_variables(EdgeSelected)
        for key, index in edge_variables.items():
            if isinstance(key[1], tuple):
                solver.add_variable_cost(index, 0.0, self.weight)
                solver.add_variable_cost(index, 0.0, self.constant)
            else:
                u, v = cast("tuple[int, int]", key)
                embedding_u = self.__get_node_embedding(solver.graph, u)
                embedding_v = self.__get_node_embedding(solver.graph, v)
                feature = np.linalg.norm(embedding_u - embedding_v)
                if (
                    self.mean_node_embedding_distance is not None
                    and self.std_node_embedding_distance
                ):
                    feature = (
                        feature - self.mean_node_embedding_distance
                    ) / self.std_node_embedding_distance
                solver.add_variable_cost(index, feature, self.weight)
                solver.add_variable_cost(index, 1.0, self.constant)

    def __get_node_embedding(self, graph: nx.DiGraph, node: int) -> np.ndarray:
        if isinstance(self.node_embedding_attribute, tuple):
            return np.array(
                [graph.nodes[node][p] for p in self.node_embedding_attribute]
            )
        return np.array(graph.nodes[node][self.node_embedding_attribute])


class NodeEmbeddingDistanceHyper(Cost):
    def __init__(
        self,
        node_embedding_attribute,
        weight,
        constant,
        mean_node_embedding_distance=None,
        std_node_embedding_distance=None,
    ):
        self.node_embedding_attribute = node_embedding_attribute
        self.weight = Weight(weight)
        self.constant = Weight(constant)
        self.mean_node_embedding_distance = mean_node_embedding_distance
        self.std_node_embedding_distance = std_node_embedding_distance

    def apply(self, solver: Solver) -> None:
        edge_variables = solver.get_variables(EdgeSelected)
        for key, index in edge_variables.items():
            if isinstance(key[1], tuple):
                (start,) = key[0]
                end1, end2 = key[1]
                embedding_u = self.__get_node_embedding(solver.graph, start)
                embedding_v1 = self.__get_node_embedding(solver.graph, end1)
                embedding_v2 = self.__get_node_embedding(solver.graph, end2)
                feature = np.linalg.norm(
                    embedding_u - 0.5 * (embedding_v1 + embedding_v2)
                )
                if (
                    self.mean_node_embedding_distance is not None
                    and self.std_node_embedding_distance
                ):
                    feature = (
                        feature - self.mean_node_embedding_distance
                    ) / self.std_node_embedding_distance
                solver.add_variable_cost(index, feature, self.weight)
                solver.add_variable_cost(index, 1.0, self.constant)
            else:
                solver.add_variable_cost(index, 0.0, self.weight)
                solver.add_variable_cost(index, 0.0, self.constant)

    def __get_node_embedding(self, graph: nx.DiGraph, node: int) -> np.ndarray:
        if isinstance(self.node_embedding_attribute, tuple):
            return np.array(
                [graph.nodes[node][p] for p in self.node_embedding_attribute]
            )
        return np.array(graph.nodes[node][self.node_embedding_attribute])


class TimeGap(Cost):
    def __init__(
        self,
        time_attribute: str | tuple[str, ...],
        weight: float = 1.0,
        constant: float = 0.0,
    ) -> None:
        self.time_attribute = time_attribute
        self.weight = Weight(weight)
        self.constant = Weight(constant)

    def apply(self, solver: Solver) -> None:
        edge_variables = solver.get_variables(EdgeSelected)
        for key, index in edge_variables.items():
            if isinstance(key[1], tuple):
                (start,) = key[0]
                end1, end2 = key[1]
                time_start = self.__get_node_frame(solver.graph, start)
                time_end1 = self.__get_node_frame(solver.graph, end1)
                time_end2 = self.__get_node_frame(solver.graph, end2)
                feature = np.linalg.norm(time_start - 0.5 * (time_end1 + time_end2))
                solver.add_variable_cost(index, feature, self.weight)
                solver.add_variable_cost(index, 1.0, self.constant)
            else:
                u, v = cast("tuple[int, int]", key)
                time_u = self.__get_node_frame(solver.graph, u)
                time_v = self.__get_node_frame(solver.graph, v)
                feature = np.linalg.norm(time_u - time_v)
                solver.add_variable_cost(index, feature, self.weight)
                solver.add_variable_cost(index, 1.0, self.constant)

    def __get_node_frame(self, graph: nx.DiGraph, node: int) -> np.ndarray:
        if isinstance(self.time_attribute, tuple):
            return np.array([graph.nodes[node][p] for p in self.time_attribute])
        return np.array(graph.nodes[node][self.time_attribute])
