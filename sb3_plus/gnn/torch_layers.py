import warnings
from typing import Any, Callable

import torch
from gymnasium.spaces.utils import flatdim
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor
from stable_baselines3.common.type_aliases import TensorDict
from torch_geometric.data import Batch
from torch_geometric.nn.models import GAT
from torch_geometric.nn.models.basic_gnn import BasicGNN
from torch_geometric.nn.pool import global_add_pool, global_max_pool, global_mean_pool
from torch_geometric.resolver import normalize_string
from torch_geometric.utils import to_dense_batch

from sb3_plus.gnn.spaces import GraphSpace

_GNN_POOL: dict[str, Callable] = {
    "add": global_add_pool,
    "max": global_max_pool,
    "mean": global_mean_pool,
}


def global_pool_resolver(pool: str | None) -> Callable | None:
    """
    Resolve the name of a global pooling to a function
    :param pool: name of the pooling function
         Options are: (`None`, `"add"`, `"max"`, `"mean"`).
    :return: pooling function or `None` if no name was specified
    """
    if pool is None:
        return None
    pool = normalize_string(pool)
    if pool not in _GNN_POOL:
        raise ValueError(
            f"{pool} is an invalid pool function, options are: {list(_GNN_POOL.keys())}"
        )
    return _GNN_POOL[pool]


class GNNExtractor(BaseFeaturesExtractor):
    """
    Extract features using a GNN model

    :param observation_space: the observation space of the environment
    :param gnn_class: class of the GNN model
    :param hidden_channels: number of hidden channels in the GNN model
    :param n_layers: number of message passing layers in the GNN model
    :param out_channels: number of output channels in the GNN model
    :param dropout: dropout probability
    :param act_fn: the non-linear activation function to use in the GNN model
    :param act_first: whether the activation is applied before normalization or not
    :param act_kwargs: arguments passed to the activation function
    :param norm_fn: the normalization function to use in the GNN model
    :param norm_kwargs: arguments passed to the normalization function
    :param jk: The Jumping Knowledge mode.
        If specified, the model will additionally apply a final linear transformation to transform
        node embeddings to the expected output feature dimensionality.
        Options are: (`None`, `"last"`, `"cat"`, `"max"`, `"lstm"`).
    :param pool_fn: global pooling function to be used.
        Options are: (`None`, `"add"`, `"max"`, `"mean"`).
    """

    _observation_space: GraphSpace

    def __init__(
        self,
        observation_space: GraphSpace,
        gnn_class: type[BasicGNN] = GAT,
        hidden_channels: int = 32,
        n_layers: int = 1,
        out_channels: int | None = None,
        dropout: float = 0.0,
        act_fn: str | Callable | None = "relu",
        act_first: bool = False,
        act_kwargs: dict[str, Any] | None = None,
        norm_fn: str | Callable | None = None,
        norm_kwargs: dict[str, Any] | None = None,
        jk: str | None = None,
        pool_fn: str | None = None,
        **kwargs,
    ):
        # Calculate total number of features that the extractor outputs
        features_dim = out_channels or hidden_channels
        if pool_fn is None:
            features_dim *= observation_space.max_nodes
        super().__init__(observation_space, features_dim)

        # Get input and edge dimensions
        edge_dim: int | None = None
        if observation_space.edge_space is not None:
            edge_dim = flatdim(observation_space.edge_space)
        in_channels = flatdim(observation_space.node_space)

        # Obtain arguments using the name conventions from torch-geometric
        if "num_layers" in kwargs:
            n_layers = kwargs.pop("num_layers")
        if "act" in kwargs:
            act_fn = kwargs.pop("act")
        if "norm" in kwargs:
            norm_fn = kwargs.pop("norm")

        self.pool = global_pool_resolver(pool_fn)
        gnn_kwargs = dict(
            in_channels=in_channels,
            hidden_channels=hidden_channels,
            num_layers=n_layers,
            out_channels=out_channels,
            dropout=dropout,
            act=act_fn,
            act_first=act_first,
            act_kwargs=act_kwargs,
            norm=norm_fn,
            norm_kwargs=norm_kwargs,
            jk=jk,
            **kwargs,
        )
        if edge_dim is not None:
            gnn_kwargs["edge_dim"] = edge_dim

        self.gnn = gnn_class(**gnn_kwargs)

    @torch.no_grad()
    def _obs_to_batch(self, observations: TensorDict) -> Batch:
        """Return batch of graphs from observations
        :param observations: observations
        :return: batched graphs
        """
        assert isinstance(
            observations, dict
        ), "Observations must follows the sample format specified by `sb3_plus.gnn.GraphSpace` space"
        assert (
            GraphSpace.NODES in observations
        ), "No node features found in the observation"
        assert (
            GraphSpace.ADJACENCY_MATRIX in observations
        ), "No graph's adjacency matrix found in the observation"

        x = observations[GraphSpace.NODES]
        adj_matrix = observations[GraphSpace.ADJACENCY_MATRIX]
        edge_attr = observations.get(GraphSpace.EDGES, None)
        mask = observations.get(GraphSpace.NODES_MASK, None)

        # Unflatten the node features
        if x.ndim == 2:
            x = x.reshape(x.size(dim=0), self._observation_space.max_nodes, -1)

        B, N, _ = x.shape
        device = x.device
        x = x.reshape(B, N, -1)
        adj_matrix = adj_matrix.reshape(B, N, N)

        if mask is None:
            mask = torch.ones(B, N, dtype=torch.bool, device=device)
        else:
            mask = mask.reshape(B, N).bool()

        # Number of nodes in each graph
        n_nodes = mask.sum(dim=1)
        # Concatenate all valid node features
        x = x[mask]

        # Mapping (graph,node) -> global node index
        local_to_global = torch.full((B, N), -1, dtype=torch.long, device=device)
        local_to_global[mask] = torch.arange(x.size(0), device=device)

        # Obtaining the edge indices
        valid_edge_mask = (adj_matrix != 0) & mask.unsqueeze(2) & mask.unsqueeze(1)
        graph_id, src_local, dst_local = valid_edge_mask.nonzero(as_tuple=True)
        edge_index = torch.stack(
            (
                local_to_global[graph_id, src_local],
                local_to_global[graph_id, dst_local],
            ),
            dim=0,
        )

        # Obtaining the edge features
        if edge_attr is not None:
            edge_attr = edge_attr.reshape(B, N, N, -1)
            edge_attr = edge_attr[graph_id, src_local, dst_local]

        # Obtaining the edge weights
        edge_weight = None
        is_binary_adj = torch.all((adj_matrix == 0) | (adj_matrix == 1))
        n_edge_feat = edge_attr.size(dim=-1) == 1 if edge_attr is not None else 0
        # edge weights are in the adj. matrix if the matrix is not binary
        if not is_binary_adj:
            edge_weight = adj_matrix[graph_id, src_local, dst_local]
        # weights are the edge features if there is only one feature
        elif n_edge_feat == 1:
            edge_weight = edge_attr

        if not torch.all(mask) and self.pool is None:
            warnings.warn(
                "A graph pooling function should be used in the policy when an environment has a variable number of nodes in the graph-based observations"
            )

        # Building the batch data
        batch = torch.repeat_interleave(torch.arange(B, device=device), n_nodes)
        ptr = (
            torch.cat(
                (
                    torch.zeros(
                        1,
                        dtype=torch.long,
                        device=device,
                    ),
                    n_nodes.cumsum(0),
                )
            ),
        )
        data = Batch(
            x=x,
            edge_index=edge_index,
            edge_weight=edge_weight,
            edge_attr=edge_attr,
            batch=batch,
            ptr=ptr,
        )
        return data

    def forward(self, observations: TensorDict) -> torch.Tensor:
        """Extract features of observations using a GNN model
        :param observations: observations
        :return: extracted features with shape (B, F) if the global pool is specified or (B, N * F) otherwise,
            where B is the batch size, N is the maximum number of nodes, and F is the out channel dimension
        """
        data = self._obs_to_batch(observations)
        output: torch.Tensor = self.gnn(
            x=data.x,
            edge_index=data.edge_index,
            edge_weight=data.edge_weight,
            edge_attr=data.edge_attr,
            batch=data.batch,
        )
        if self.pool is not None:
            output = self.pool(output, data.batch)
        else:
            output, _ = to_dense_batch(
                output, data.batch, max_num_nodes=self._observation_space.max_nodes
            )
            output = output.flatten(start_dim=1)
        return output
