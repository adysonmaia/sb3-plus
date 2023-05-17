from stable_baselines3.common import distributions as sb3
from .preprocessing import get_net_action_dim, get_action_dim
from typing import Dict, Tuple, Any, Union, Optional
from torch import nn
import torch as th
import gym


class ParametrizedDistribution(sb3.Distribution):
    """
    Distribution to a Dict action space

    """

    def __init__(self, action_space: Union[gym.spaces.Dict, gym.spaces.Tuple]):
        super().__init__()
        self.action_space = action_space
        list_spaces = action_space.spaces.values() if isinstance(action_space, gym.spaces.Dict) else action_space.spaces
        self.distribution = [sb3.make_proba_distribution(s) for s in list_spaces]
        self.action_dims = [get_action_dim(s) for s in list_spaces]
        self._flatten_action_dim = sum(self.action_dims)
        self._net_action_dims = [get_net_action_dim(s) for s in list_spaces]
        self._net_flatten_action_dim = sum(self._net_action_dims)

    def proba_distribution_net(self, latent_dim: int, log_std_init: float = 0.0) -> Tuple[nn.Module, nn.Parameter]:
        """Create the layers and parameters that represent the distribution.
        Subclasses must define this, but the arguments and return type vary between
        concrete classes."""
        flatten_actions = nn.Linear(latent_dim, self._net_flatten_action_dim)
        flatten_log_std = nn.Parameter(th.ones(self._net_flatten_action_dim) * log_std_init, requires_grad=True)
        return flatten_actions, flatten_log_std

    def proba_distribution(self, mean_actions: th.Tensor, log_std: th.Tensor) -> 'ParametrizedDistribution':
        """Set parameters of the distribution.
        :return: self
        """
        split_mean_actions = th.split(mean_actions, self._net_action_dims, dim=1)
        split_log_std = th.split(log_std, self._net_action_dims, dim=-1)
        # print(log_std.shape, [t.shape for t in split_log_std])
        # print(mean_actions.shape, [t.shape for t in split_mean_actions])
        for dist, dist_mean_actions, dist_log_std in zip(self.distribution, split_mean_actions, split_log_std):
            if isinstance(dist, sb3.DiagGaussianDistribution):
                dist.proba_distribution(dist_mean_actions, dist_log_std)
            else:
                # For categorical and bernoulli distributions, mean actions are actually action logits
                dist.proba_distribution(dist_mean_actions)

        return self

    def log_prob(self, actions: th.Tensor) -> th.Tensor:
        """
        Returns the log likelihood
        :param actions: the taken action
        :return: The log likelihood of the distribution
        """
        split_actions = th.split(actions, self.action_dims, dim=1)
        log_prob = th.stack([dist.log_prob(action) for dist, action in zip(self.distribution, split_actions)], dim=1)
        log_prob = log_prob.sum(dim=1)
        return log_prob

    def entropy(self) -> Optional[th.Tensor]:
        """
        Returns Shannon's entropy of the probability
        :return: the entropy, or None if no analytical form is known
        """
        return th.stack([dist.entropy() for dist in self.distribution], dim=1).sum(dim=1)

    def sample(self) -> th.Tensor:
        """
        Returns a sample from the probability distribution
        :return: the stochastic action
        """
        return th.cat([dist.sample() for dist in self.distribution], dim=1)

    def mode(self) -> th.Tensor:
        """
        Returns the most likely action (deterministic output)
        from the probability distribution
        :return: the stochastic action
        """
        return th.cat([dist.mode() for dist in self.distribution], dim=1)

    def actions_from_params(self, mean_actions: th.Tensor, log_std: th.Tensor,
                            deterministic: bool = False) -> th.Tensor:
        """
        Returns samples from the probability distribution
        given its parameters.
        :return: actions
        """
        split_mean_actions = th.split(mean_actions, self._net_action_dims, dim=1)
        split_log_std = th.split(log_std, self._net_action_dims, dim=1)
        list_actions = []
        for dist, dist_mean_actions, dist_log_std in zip(self.distribution, split_mean_actions, split_log_std):
            actions = None
            if isinstance(dist, sb3.DiagGaussianDistribution):
                actions = dist.actions_from_params(dist_mean_actions, dist_log_std, deterministic=deterministic)
            else:
                # For categorical and bernoulli distributions, mean actions are actually action logits
                actions = dist.actions_from_params(dist_mean_actions, deterministic=deterministic)
            list_actions.append(actions)
        actions = th.cat(list_actions, dim=1)
        return actions

    def log_prob_from_params(self, mean_actions: th.Tensor, log_std: th.Tensor) -> Tuple[th.Tensor, th.Tensor]:
        """
        Returns samples and the associated log probabilities
        from the probability distribution given its parameters.
        :return: actions and log prob
        """
        split_mean_actions = th.split(mean_actions, self._net_action_dims, dim=1)
        split_log_std = th.split(log_std, self._net_action_dims, dim=1)
        list_actions = []
        list_log_prob = []
        for dist, dist_mean_actions, dist_log_std in zip(self.distribution, split_mean_actions, split_log_std):
            actions = None
            log_prob = None
            if isinstance(dist, sb3.DiagGaussianDistribution):
                actions, log_prob = dist.log_prob_from_params(dist_mean_actions, dist_log_std)
            else:
                # For categorical and bernoulli distributions, mean actions are actually action logits
                actions, log_prob = dist.log_prob_from_params(dist_mean_actions)
            list_actions.append(actions)
            list_log_prob.append(log_prob)

        actions = th.cat(list_actions, dim=1)
        log_prob = th.cat(list_log_prob, dim=1)
        return actions, log_prob


def make_proba_distribution(action_space: gym.spaces.Space, use_sde: bool = False,
                            dist_kwargs: Optional[Dict[str, Any]] = None) -> sb3.Distribution:
    """
    Return an instance of Distribution for the correct type of action space
    :param action_space: the input action space
    :param use_sde: Force the use of StateDependentNoiseDistribution
        instead of DiagGaussianDistribution
    :param dist_kwargs: Keyword arguments to pass to the probability distribution
    :return: the appropriate Distribution object
    """
    if isinstance(action_space, (gym.spaces.Dict, gym.spaces.Tuple)):
        return ParametrizedDistribution(action_space)
    else:
        return sb3.make_proba_distribution(action_space, use_sde, dist_kwargs)


def kl_divergence(dist_true: sb3.Distribution, dist_pred: sb3.Distribution) -> th.Tensor:
    """
    Wrapper for the PyTorch implementation of the full form KL Divergence
    :param dist_true: the p distribution
    :param dist_pred: the q distribution
    :return: KL(dist_true||dist_pred)
    """
    # KL Divergence for different distribution types is out of scope
    assert dist_true.__class__ == dist_pred.__class__, "Error: input distributions should be the same type"
    if isinstance(dist_pred, ParametrizedDistribution) and isinstance(dist_true, ParametrizedDistribution):
        assert dist_pred.action_dims == dist_true.action_dims, "Error: distributions must have the same input space"
        return th.cat([sb3.kl_divergence(p, q) for p, q in zip(dist_true.distribution, dist_pred.distribution)],
                      dim=1
                      ).sum(dim=1)
    else:
        return sb3.kl_divergence(dist_true, dist_pred)
