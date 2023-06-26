from stable_baselines3.common.torch_layers import MlpExtractor
from typing import Dict, List, Tuple, Type, Union
from torch import nn
import torch as th


class LagMlpExtractor(MlpExtractor):
    """
    Constructs an MLP that receives the output from a previous features extractor (i.e. a CNN) or directly
    the observations (if no features extractor is applied) as an input and outputs a latent representation
    for the policy and a value network.

    The ``net_arch`` parameter allows to specify the amount and size of the hidden layers.
    It can be in either of the following forms:
    1. ``dict(vf=[<list of layer sizes>], cvf=[<value layer sizes>], pi=[<list of layer sizes>])``:
        to specify the amount and size of the layers in the
        policy, penalty/cost and value nets individually. If it is missing any of the keys (pi, pvf or vf),
        zero layers will be considered for that key.
    2. ``[<list of layer sizes>]``: "shortcut" in case the amount and size of the layers
        in the policy and value nets are the same. Same as ``dict(vf=int_list, pvf=int_list, pi=int_list)``
        where int_list is the same for the actor and critic.

    .. note::
        If a key is not specified or an empty list is passed ``[]``, a linear network will be used.

    :param feature_dim: Dimension of the feature vector (can be the output of a CNN)
    :param net_arch: The specification of the policy and value networks.
        See above for details on its formatting.
    :param activation_fn: The activation function to use for the networks.
    :param device:
    """

    def __init__(
            self,
            feature_dim: int,
            net_arch: Union[List[int], Dict[str, List[int]]],
            activation_fn: Type[nn.Module],
            device: Union[th.device, str] = "auto",
    ) -> None:
        super().__init__(feature_dim, net_arch, activation_fn, device)
        penalty_value_net: List[nn.Module] = []
        last_layer_dim_pvf = feature_dim

        # save dimensions of layers in penalty net
        if isinstance(net_arch, dict):
            vf_layers_dims = net_arch.get("vf", [])
            pvf_layers_dims = net_arch.get("pvf", vf_layers_dims)
        else:
            pvf_layers_dims = net_arch
        # Iterate through the penalty layers and build the policy net
        for curr_layer_dim in pvf_layers_dims:
            penalty_value_net.append(nn.Linear(last_layer_dim_pvf, curr_layer_dim))
            penalty_value_net.append(activation_fn())
            last_layer_dim_pvf = curr_layer_dim

        # Save dim, used to create the distributions
        self.latent_dim_pvf = last_layer_dim_pvf
        # Create networks
        # If the list of layers is empty, the network will just act as an Identity module
        self.penalty_value_net = nn.Sequential(*penalty_value_net).to(device)

    def forward(self, features: th.Tensor) -> Tuple[th.Tensor, th.Tensor, th.Tensor]:
        """
        :return: latent_policy, latent_value, latent_cost_value of the specified network.
            If all layers are shared, then ``latent_policy == latent_value == latent_cost_value``
        """
        return self.policy_net(features), self.value_net(features), self.penalty_value_net(features)

    def forward_penalty(self, features: th.Tensor) -> th.Tensor:
        return self.penalty_value_net(features)
