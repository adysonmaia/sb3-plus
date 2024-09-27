from stable_baselines3.common.torch_layers import MlpExtractor
from typing import Dict, List, Tuple, Type, Union
from torch import nn
import torch as th


class SafeMlpExtractor(MlpExtractor):
    """
    MLP Extractor for Safe Reinforcement Learning

    The ``net_arch`` parameter allows to specify the amount and size of the hidden layers.
    It can be in either of the following forms:
    1. ``dict(vf=[<list of layer sizes>], cvf=[<value layer sizes>], pi=[<list of layer sizes>])``:
        to specify the amount and size of the layers in the
        policy, cost and value nets individually. If it is missing any of the keys (pi, cvf or vf),
        zero layers will be considered for that key.
    2. ``[<list of layer sizes>]``: "shortcut" in case the amount and size of the layers
        in the policy and value nets are the same. Same as ``dict(vf=int_list, cvf=int_list, pi=int_list)``
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
        cost_value_net: List[nn.Module] = []
        last_layer_dim_cvf = feature_dim

        # save dimensions of layers in cost net
        if isinstance(net_arch, dict):
            vf_layers_dims = net_arch.get("vf", [])
            cvf_layers_dims = net_arch.get("cvf", vf_layers_dims)
        else:
            cvf_layers_dims = net_arch
        # Iterate through the cost layers and build the policy net
        for curr_layer_dim in cvf_layers_dims:
            cost_value_net.append(nn.Linear(last_layer_dim_cvf, curr_layer_dim))
            cost_value_net.append(activation_fn())
            last_layer_dim_cvf = curr_layer_dim

        # Save dim, used to create the distributions
        self.latent_dim_cvf = last_layer_dim_cvf
        # Create networks
        # If the list of layers is empty, the network will just act as an Identity module
        self.cost_value_net = nn.Sequential(*cost_value_net).to(device)

    def forward(self, features: th.Tensor) -> Tuple[th.Tensor, th.Tensor, th.Tensor]:
        """
        :return: latent_policy, latent_value, latent_cost_value of the specified network.
            If all layers are shared, then ``latent_policy == latent_value == latent_cost_value``
        """
        return self.policy_net(features), self.value_net(features), self.cost_value_net(features)

    def forward_cost(self, features: th.Tensor) -> th.Tensor:
        return self.cost_value_net(features)
