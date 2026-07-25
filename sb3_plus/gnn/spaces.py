from collections import OrderedDict
from typing import Any

import numpy as np
from gymnasium import spaces
from gymnasium.spaces.utils import flatten_space

GraphSample = dict[str, np.ndarray]


class GraphSpace(spaces.Dict):
    """A space representing graph information that is compatible with `Stable-Baseline3`

    :param max_nodes: maximum number of nodes in the graph
    :param node_space: nodes' features space
    :param edge_space: edges' features space
    :param flatten: whether use flatten representations of graph data or not
    :param seed: random seed used to sample from this space
    """

    NODES = "nodes"
    ADJACENCY_MATRIX = "adj_matrix"
    EDGES = "edges"
    NODES_MASK = "nodes_mask"

    def __init__(
        self,
        max_nodes: int,
        node_space: spaces.Box,
        edge_space: spaces.Box | None = None,
        flatten: bool = True,
        seed: int | np.random.Generator | None = None,
    ):
        assert len(node_space.shape) == 1, "Node space must be flatten"
        assert max_nodes > 0, "Number of nodes must be larger than 0"
        if edge_space is not None:
            assert len(edge_space.shape) == 1, "Edge space must be flatten"

        dict_space = OrderedDict(
            [
                (
                    self.NODES,
                    spaces.Box(
                        shape=(max_nodes,) + node_space.shape,
                        low=np.broadcast_to(
                            node_space.low, (max_nodes,) + node_space.low.shape
                        ),
                        high=np.broadcast_to(
                            node_space.high, (max_nodes,) + node_space.high.shape
                        ),
                        dtype=node_space.dtype,
                    ),
                ),
                (
                    self.NODES_MASK,
                    spaces.MultiBinary(n=max_nodes),
                ),
                (
                    self.ADJACENCY_MATRIX,
                    spaces.Box(
                        shape=(max_nodes, max_nodes), low=0, high=1, dtype=np.float32
                    ),
                ),
            ]
        )
        if edge_space is not None:
            n_edges = max_nodes * max_nodes
            dict_space[self.EDGES] = spaces.Box(
                shape=(n_edges,) + edge_space.shape,
                low=np.broadcast_to(edge_space.low, (n_edges,) + edge_space.low.shape),
                high=np.broadcast_to(
                    edge_space.high, (n_edges,) + edge_space.high.shape
                ),
                dtype=edge_space.dtype,
            )

        if flatten:
            for key, space in dict_space.items():
                dict_space[key] = flatten_space(space)

        super().__init__(spaces=dict_space, seed=seed)

        self.flatten = flatten
        self.max_nodes = max_nodes
        self.node_space = node_space
        self.edge_space = edge_space

    def seed(self, seed: int | None = None) -> int:
        """Seeds the PRNG of this space and node / edge subspace.
        :param seed: optional seed
        :return: seed value
        """
        self.node_space.seed(seed)
        if self.edge_space is not None:
            self.edge_space.seed(seed)
        return super().seed(seed)

    def __repr__(self) -> str:
        """A string representation of this space.
        :return: space representation as a str
        """
        return (
            f"Graph({self.max_nodes}, node={self.node_space}, edge={self.edge_space})"
        )

    def __eq__(self, other: Any) -> bool:
        """Check whether `other` is equivalent to this instance.
        :param: other object
        :return: whether it is equal or not"""
        return (
            isinstance(other, GraphSpace)
            and (self.node_space == other.node_space)
            and (self.edge_space == other.edge_space)
            and (self.max_nodes == other.max_nodes)
        )

    def build_sample(
        self, x: np.ndarray, adj_matrix: np.ndarray, edge_attr: np.ndarray | None = None
    ) -> GraphSample:
        """Build a sample of the graph space given data of a graph
        :param x: nodes' features with shape (N, D) where D is the feature dimension
        :param adj_matrix: adjacency matrix with shape (N, N)
        :param edge_attr: edge's features with shape (N, N, D) where D is the feature dimension
        :return: built sample
        """
        if x.ndim == 1:
            x = x.reshape(-1, *self.node_space.shape)
        n_nodes = x.shape[0]
        if n_nodes > self.max_nodes:
            raise ValueError(
                f"Sample has {n_nodes} nodes, which is greater than {self.max_nodes}, the maximum number of nodes in the graph"
            )
        x = x.reshape(n_nodes, -1)
        adj_matrix = adj_matrix.reshape(n_nodes, n_nodes)
        mask = np.ones((self.max_nodes,), dtype=np.int8)
        pad_width = 0
        if n_nodes < self.max_nodes:
            pad_width = self.max_nodes - n_nodes
            x = np.pad(
                x,
                pad_width=[(0, pad_width), (0, 0)],
                mode="mean",
            )
            adj_matrix = np.pad(
                adj_matrix,
                pad_width=[(0, pad_width), (0, pad_width)],
                mode="constant",
                constant_values=0,
            )
            mask[n_nodes:] = 0
        sample = OrderedDict(
            [
                (self.NODES, x),
                (self.NODES_MASK, mask),
                (self.ADJACENCY_MATRIX, adj_matrix),
            ]
        )
        if edge_attr is not None:
            edge_attr = edge_attr.reshape(n_nodes, n_nodes, -1)
            if pad_width > 0:
                edge_attr = np.pad(
                    edge_attr,
                    pad_width=[(0, pad_width), (0, pad_width), (0, 0)],
                    mode="mean",
                )
        elif self.edge_space is not None:
            edge_shape = (self.max_nodes * self.max_nodes,) + self.edge_space.shape
            edge_attr = np.full(
                edge_shape, fill_value=self.edge_space.low, dtype=self.edge_space.dtype
            )

        if edge_attr is not None:
            edge_attr = edge_attr.reshape(self.max_nodes * self.max_nodes, -1)
            sample[self.EDGES] = edge_attr

        if self.flatten:
            for key, value in sample.items():
                sample[key] = value.flatten()
        return sample

    def build_sample_from_gym(self, sample: spaces.GraphInstance) -> GraphSample:
        """
        Build a sample of the graph space given a `gymnasium.spaces.GraphInstance` graph sample
        :param sample: gym graph sample
        :return: built sample
        """
        n_nodes = sample.nodes.shape[0]
        adj_matrix = np.zeros((n_nodes, n_nodes), dtype=np.float32)
        if sample.edge_links is not None:
            src = sample.edge_links[:, 0]
            dst = sample.edge_links[:, 1]
            adj_matrix[src, dst] = 1

        edge_attr = None
        if sample.edges is not None and sample.edge_links is not None:
            edge_shape = (n_nodes, n_nodes) + sample.edges.shape[1:]
            fill_value = sample.edges.mean(axis=0)
            edge_attr = np.full(
                edge_shape, fill_value=fill_value, dtype=sample.edges.dtype
            )
            edge_attr[src, dst] = sample.edges

        return self.build_sample(
            x=sample.nodes, adj_matrix=adj_matrix, edge_attr=edge_attr
        )

    @staticmethod
    def from_gym(
        space: spaces.Graph,
        max_nodes: int,
        flatten: bool = False,
        seed: int | np.random.Generator | None = None,
    ) -> "GraphSpace":
        """Build the space from the `gymnasium.spaces.Graph` space

        :param space: gymnasium space
        :param max_nodes: maximum number of nodes in the graph
        :param flatten: whether use flatten representations of graph data or not
        :param seed: random seed used to sample from this space
        :return: built space
        """
        assert isinstance(
            space.node_space, spaces.Box
        ), "Only Box space is supported for a node space"
        assert space.edge_space is None or isinstance(
            space.edge_space, spaces.Box
        ), "Only Box space is supported for an edge space"
        if seed is None and space.np_random is not None:
            seed = space.np_random
        return GraphSpace(
            max_nodes=max_nodes,
            node_space=space.node_space,
            edge_space=space.edge_space,
            flatten=flatten,
            seed=seed,
        )
