from typing import Any

import torch
from gymnasium import spaces
from stable_baselines3.common.policies import ActorCriticPolicy
from stable_baselines3.common.type_aliases import Schedule
from torch_geometric.nn.models import GAT
from torch_geometric.nn.models.basic_gnn import BasicGNN

from sb3_plus.gnn.spaces import GraphSpace
from sb3_plus.gnn.torch_layers import GNNExtractor


class ActorCriticGnnPolicy(ActorCriticPolicy):
    """
    GNN policy class for actor-critic algorithms (has both policy and value prediction).
    Used by A2C, PPO and the likes.

    The policy uses a GNN model to embed nodes from graph-based observations,
    then a MLP-based readout model to extract features of the entire node embeddings

    :param observation_space: Observation space
    :param action_space: Action space
    :param lr_schedule: Learning rate schedule (could be constant)
    :param gnn_class: Class of the GNN model.
        Options are:
        (``torch_geometric.nn.models.GCN``, ``torch_geometric.nn.models.GraphSAGE``,
        ``torch_geometric.nn.models.GIN``, ``torch_geometric.nn.models.GAT``,
        ``torch_geometric.nn.models.PNA``,``torch_geometric.nn.models.EdgeCNN``)
    :param gnn_kwargs: Initialization arguments of the GNN model
        See the torch-geometric ``documentation <https://pytorch-geometric.readthedocs.io/en/latest/modules/nn.html#models>`` for details.
    :param gnn_pool_fn: Name of the global graph pooling function to be used.
        Options are: (``None``, ``"add"``, ``"max"``, ``"mean"``)
    :param readout_net_arch: The specification of the graph readout model in the policy and value networks.
    :param readout_act_fn: Activation function of the graph readout model
    :param ortho_init: Whether to use or not orthogonal initialization
    :param use_sde: Whether to use State Dependent Exploration or not
    :param log_std_init: Initial value for the log standard deviation
    :param full_std: Whether to use (n_features x n_actions) parameters
        for the std instead of only (n_features,) when using gSDE
    :param use_expln: Use ``expln()`` function instead of ``exp()`` to ensure
        a positive standard deviation (cf paper). It allows to keep variance
        above zero and prevent it from growing too fast. In practice, ``exp()`` is usually enough.
    :param squash_output: Whether to squash the output using a tanh function,
        this allows to ensure boundaries when using gSDE.
    :param features_extractor_class: Features extractor to use.
    :param features_extractor_kwargs: Keyword arguments
        to pass to the features extractor.
    :param share_features_extractor: If True, the features extractor is shared between the policy and value networks.
    :param normalize_images: Whether to normalize images or not,
         dividing by 255.0 (True by default)
    :param optimizer_class: The optimizer to use,
        ``th.optim.Adam`` by default
    :param optimizer_kwargs: Additional keyword arguments,
        excluding the learning rate, to pass to the optimizer
    """

    def __init__(
        self,
        observation_space: GraphSpace,
        action_space: spaces.Space,
        lr_schedule: Schedule,
        gnn_class: type[BasicGNN] = GAT,
        gnn_kwargs: dict[str, Any] | None = None,
        gnn_pool_fn: str | None = None,
        readout_net_arch: list[int] | dict[str, list[int]] | None = None,
        readout_act_fn: type[torch.nn.Module] = torch.nn.Tanh,
        ortho_init: bool = True,
        use_sde: bool = False,
        log_std_init: float = 0.0,
        full_std: bool = True,
        use_expln: bool = False,
        squash_output: bool = False,
        share_features_extractor: bool = True,
        normalize_images: bool = True,
        optimizer_class: type[torch.optim.Optimizer] = torch.optim.Adam,
        optimizer_kwargs: dict[str, Any] | None = None,
    ):
        assert isinstance(
            observation_space, GraphSpace
        ), f"{type(observation_space)} is incompatible with the policy, a `sb3_plus.gnn.GraphSpace` space must be used instead"
        features_extractor_class = GNNExtractor
        features_extractor_kwargs = dict(gnn_class=gnn_class, pool_fn=gnn_pool_fn)
        features_extractor_kwargs.update(gnn_kwargs or {})
        super().__init__(
            observation_space,
            action_space,
            lr_schedule,
            readout_net_arch,
            readout_act_fn,
            ortho_init,
            use_sde,
            log_std_init,
            full_std,
            use_expln,
            squash_output,
            features_extractor_class,
            features_extractor_kwargs,
            share_features_extractor,
            normalize_images,
            optimizer_class,
            optimizer_kwargs,
        )
