from stable_baselines3.common.torch_layers import MlpExtractor
from typing import Dict, List, Tuple, Type, Union
from torch import nn
import torch as th


class LagMlpExtractor(MlpExtractor):
    """
    Constructs an MLP that receives the output from a previous features extractor (i.e. a CNN) or directly
    the observations (if no features extractor is applied) as an input and outputs a latent representation
    for the policy and a value network.
    The ``net_arch`` parameter allows to specify the amount and size of the hidden layers and how many
    of them are shared between the policy network and the value network. It is assumed to be a list with the following
    structure:

    1. An arbitrary length (zero allowed) number of integers each specifying the number of units in a shared layer.
       If the number of ints is zero, there will be no shared layers.
    2. An optional dict, to specify the following non-shared layers for the value network, the cost value network
       and the policy network.
       It is formatted like ``dict(vf=[<value layer sizes>], cvf=[<value layer sizes>], pi=[<policy layer sizes>])``.
       If it is missing any of the keys (pi, cvf or vf), no non-shared layers (empty list) is assumed.

    Deprecation note: shared layers in ``net_arch`` are deprecated, please use separate
    pi, vf, and pvf networks (e.g. net_arch=dict(pi=[...], vf=[...], pvf=[...]))

    Adapted from Stable Baselines.

    :param feature_dim: Dimension of the feature vector (can be the output of a CNN)
    :param net_arch: The specification of the policy and value networks.
        See above for details on its formatting.
    :param activation_fn: The activation function to use for the networks.
    :param device:
    """

    def __init__(
            self,
            feature_dim: int,
            net_arch: Union[Dict[str, List[int]], List[Union[int, Dict[str, List[int]]]]],
            activation_fn: Type[nn.Module],
            device: Union[th.device, str] = "auto",
    ) -> None:
        super().__init__(feature_dim, net_arch, activation_fn, device)
        penalty_value_net: List[nn.Module] = []
        penalty_value_only_layers: List[int] = []  # Layer sizes of the network that only belongs to the value network

        last_layer_dim_shared = feature_dim
        if isinstance(net_arch, dict):
            penalty_value_only_layers = net_arch.get("pvf", net_arch["vf"])
        else:
            for layer in net_arch:
                if not isinstance(layer, dict):
                    last_layer_dim_shared = layer
                    continue

                if "pvf" in layer:
                    assert isinstance(layer["pvf"], list), "Error: net_arch[-1]['pvf'] must contain a list of integers."
                    penalty_value_only_layers = layer["pvf"]
                break  # From here on the network splits up in policy and value network
        last_layer_dim_pvf = last_layer_dim_shared

        # Build the non-shared part of the network
        for pvf_layer_size in penalty_value_only_layers:
            assert isinstance(pvf_layer_size, int), "Error: net_arch[-1]['pvf'] must only contain integers."
            penalty_value_net.append(nn.Linear(last_layer_dim_pvf, pvf_layer_size))
            penalty_value_net.append(activation_fn())
            last_layer_dim_pvf = pvf_layer_size

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
        shared_latent = self.shared_net(features)
        return self.policy_net(shared_latent), self.value_net(shared_latent), self.penalty_value_net(shared_latent)

    def forward_penalty(self, features: th.Tensor) -> th.Tensor:
        shared_latent = self.shared_net(features)
        return self.penalty_value_net(shared_latent)

