from .preprocessing import get_net_action_dim, get_action_dim
from stable_baselines3.common import distributions as sb3
from typing import Dict, Tuple, Any, TypeVar, Union, Optional
from torch import nn
from gymnasium import spaces
import torch as th


SelfMultiOutputDistribution = TypeVar("SelfMultiOutputDistribution", bound="MultiOutputDistribution")


class MultiOutputDistribution(sb3.Distribution):
    """
    Distribution to a multi outputs represented as a Dict or Tuple action space

    """

    def __init__(self, action_space: Union[spaces.Dict, spaces.Tuple]):
        super().__init__()
        self.action_space = action_space
        list_spaces = action_space.spaces.values() if isinstance(action_space, spaces.Dict) else action_space.spaces
        # TODO: Add support to nested distributions with initialization arguments
        self.distribution = [make_proba_distribution(s) for s in list_spaces]
        self.action_dims = [get_action_dim(s) for s in list_spaces]
        self._flatten_action_dim = sum(self.action_dims)
        self._net_action_dims = [get_net_action_dim(s) for s in list_spaces]
        self._net_flatten_action_dim = sum(self._net_action_dims)

    def proba_distribution_net(self, latent_dim: int, log_std_init: float = 0.0) -> Tuple[nn.Module, nn.Parameter]:
        """
        Create the layers and parameter that represent the distribution:
        one output will be the mean or logits, the other parameter will be the
        standard deviation (log std in fact to allow negative values) for Gaussian distributions

        :param latent_dim: Dimension of the last layer of the policy (before the action layer)
        :param log_std_init: Initial value for the log standard deviation
        :return:
        """
        flatten_actions = nn.Linear(latent_dim, self._net_flatten_action_dim)
        flatten_log_std = nn.Parameter(th.ones(self._net_flatten_action_dim) * log_std_init, requires_grad=True)
        return flatten_actions, flatten_log_std

    def proba_distribution(
            self: SelfMultiOutputDistribution, mean_actions: th.Tensor, log_std: th.Tensor
    ) -> SelfMultiOutputDistribution:
        """
        Create the distribution given its parameters (mean, std)

        :param mean_actions:
        :param log_std:
        :return:
        """
        split_mean_actions = th.split(mean_actions, self._net_action_dims, dim=1)
        split_log_std = th.split(log_std, self._net_action_dims, dim=-1)
        for dist, dist_mean_actions, dist_log_std in zip(self.distribution, split_mean_actions, split_log_std):
            if isinstance(dist, (sb3.DiagGaussianDistribution, MultiOutputDistribution)):
                dist.proba_distribution(dist_mean_actions, dist_log_std)
            else:
                # For categorical and bernoulli distributions, mean actions are actually action logits
                dist.proba_distribution(dist_mean_actions)

        return self

    def log_prob(self, actions: th.Tensor) -> th.Tensor:
        """
        Returns the log probabilities of actions according to the distribution.
        Note that you must first call the ``proba_distribution()`` method.

        :param actions: the taken action
        :return: The log likelihood of the distribution
        """
        split_actions = th.split(actions, self.action_dims, dim=1)
        list_log_prob = [dist.log_prob(action) for dist, action in zip(self.distribution, split_actions)]
        log_prob = th.stack(list_log_prob, dim=1)
        log_prob = log_prob.sum(dim=1)
        return log_prob

    def entropy(self) -> Optional[th.Tensor]:
        """
        Returns Shannon's entropy of the probability
        :return: the entropy, or None if no analytical form is known
        """
        list_entropies = [dist.entropy() for dist in self.distribution]
        if None in list_entropies:
            return None
        return th.stack(list_entropies, dim=1).sum(dim=1)

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

    def actions_from_params(
            self, mean_actions: th.Tensor, log_std: th.Tensor, deterministic: bool = False
    ) -> th.Tensor:
        """
        Returns samples from the probability distribution
        given its parameters.

        :param mean_actions:
        :param log_std:
        :param deterministic:
        :return: actions
        """
        split_mean_actions = th.split(mean_actions, self._net_action_dims, dim=1)
        split_log_std = th.split(log_std, self._net_action_dims, dim=1)
        list_actions = []
        for dist, dist_mean_actions, dist_log_std in zip(self.distribution, split_mean_actions, split_log_std):
            actions = None
            if isinstance(dist, (sb3.DiagGaussianDistribution, MultiOutputDistribution)):
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

        :param mean_actions:
        :param log_std:
        :return: actions and log prob
        """
        split_mean_actions = th.split(mean_actions, self._net_action_dims, dim=1)
        split_log_std = th.split(log_std, self._net_action_dims, dim=1)
        list_actions = []
        list_log_prob = []
        for dist, dist_mean_actions, dist_log_std in zip(self.distribution, split_mean_actions, split_log_std):
            actions = None
            log_prob = None
            if isinstance(dist, (sb3.DiagGaussianDistribution, MultiOutputDistribution)):
                actions, log_prob = dist.log_prob_from_params(dist_mean_actions, dist_log_std)
            else:
                # For categorical and bernoulli distributions, mean actions are actually action logits
                actions, log_prob = dist.log_prob_from_params(dist_mean_actions)
            list_actions.append(actions)
            list_log_prob.append(log_prob)

        actions = th.cat(list_actions, dim=1)
        log_prob = th.cat(list_log_prob, dim=1)
        return actions, log_prob


class FlattenCategoricalDistribution(sb3.CategoricalDistribution):
    """
    Distribution to categorical actions that consider the dimension of flatten action space
    """
    def __init__(self, action_space: spaces.Discrete):
        super().__init__(action_space.n)
        self.action_flatdim = get_action_dim(action_space)

    def sample(self) -> th.Tensor:
        return super().sample().reshape((-1, self.action_flatdim))

    def mode(self) -> th.Tensor:
        return super().mode().reshape((-1, self.action_flatdim))

    def log_prob(self, actions: th.Tensor) -> th.Tensor:
        return super().log_prob(actions.squeeze(-1))


def make_proba_distribution(action_space: spaces.Space, use_sde: bool = False,
                            dist_kwargs: Optional[Dict[str, Any]] = None) -> sb3.Distribution:
    """
    Return an instance of Distribution for the correct type of action space
    :param action_space: the input action space
    :param use_sde: Force the use of StateDependentNoiseDistribution
        instead of DiagGaussianDistribution
    :param dist_kwargs: Keyword arguments to pass to the probability distribution
    :return: the appropriate Distribution object
    """
    if isinstance(action_space, (spaces.Dict, spaces.Tuple)):
        assert not use_sde, "Error: StateDependentNoiseDistribution not supported for multi action"
        return MultiOutputDistribution(action_space)
    elif isinstance(action_space, spaces.Discrete):
        return FlattenCategoricalDistribution(action_space)
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
    if isinstance(dist_pred, MultiOutputDistribution) and isinstance(dist_true, MultiOutputDistribution):
        assert dist_pred.action_dims == dist_true.action_dims, "Error: distributions must have the same input space"
        return th.cat([kl_divergence(p, q) for p, q in zip(dist_true.distribution, dist_pred.distribution)],
                      dim=1
                      ).sum(dim=1)
    else:
        return sb3.kl_divergence(dist_true, dist_pred)
