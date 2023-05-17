from typing import NamedTuple, Dict, Union
import torch as th

TensorDict = Dict[Union[str, int], th.Tensor]


class LagRolloutBufferSamples(NamedTuple):
    observations: th.Tensor
    actions: th.Tensor
    old_values: th.Tensor
    old_log_prob: th.Tensor
    advantages: th.Tensor
    returns: th.Tensor
    old_penalty_values: th.Tensor
    penalty_returns: th.Tensor
    penalty_advantages: th.Tensor


class LagDictRolloutBufferSamples(NamedTuple):
    observations: TensorDict
    actions: th.Tensor
    old_values: th.Tensor
    old_log_prob: th.Tensor
    advantages: th.Tensor
    returns: th.Tensor
    old_penalty_values: th.Tensor
    penalty_returns: th.Tensor
    penalty_advantages: th.Tensor
