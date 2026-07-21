from typing import NamedTuple

import torch as th

TensorDict = dict[str | int, th.Tensor]

PENALTY_COST_INFO_KEY = "cost"


class SafeRolloutBufferSamples(NamedTuple):
    observations: th.Tensor
    actions: th.Tensor
    old_values: th.Tensor
    old_log_prob: th.Tensor
    advantages: th.Tensor
    returns: th.Tensor
    old_cost_values: th.Tensor
    cost_returns: th.Tensor
    cost_advantages: th.Tensor


class SafeDictRolloutBufferSamples(NamedTuple):
    observations: TensorDict
    actions: th.Tensor
    old_values: th.Tensor
    old_log_prob: th.Tensor
    advantages: th.Tensor
    returns: th.Tensor
    old_cost_values: th.Tensor
    cost_returns: th.Tensor
    cost_advantages: th.Tensor
