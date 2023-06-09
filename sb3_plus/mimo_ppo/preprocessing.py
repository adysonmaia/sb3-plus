from sb3_plus.common.space import flatdim
from gym import spaces
from typing import Any
import numpy as np


def get_action_dtype(action_space: spaces.Space) -> Any:
    if isinstance(action_space, spaces.Dict):
        return np.result_type(*[s.dtype for s in action_space.spaces.values()])
    elif isinstance(action_space, spaces.Tuple):
        return np.result_type(*[s.dtype for s in action_space.spaces])
    else:
        return action_space.dtype


def get_action_dim(action_space: spaces.Space) -> int:
    """
    Get the dimension of the action space.
    :param action_space:
    :return:
    """
    return flatdim(action_space, use_onehot=False)


def get_net_action_dim(action_space: spaces.Space) -> int:
    """
    Get the dimension of the neural network layer to select an action based on the action space
    :param action_space: action space
    :return: dimension
    """
    if isinstance(action_space, spaces.Box):
        return int(np.prod(action_space.shape))
    elif isinstance(action_space, spaces.Discrete):
        return int(action_space.n)
    elif isinstance(action_space, spaces.MultiDiscrete):
        return int(sum(action_space.nvec))
    elif isinstance(action_space, spaces.MultiBinary):
        return int(action_space.n)
    elif isinstance(action_space, spaces.Dict):
        return sum([get_net_action_dim(s) for s in action_space.spaces.values()])
    else:
        raise NotImplementedError(f"{action_space} action space is not supported")


def scale_actions(actions: np.ndarray, action_space: spaces.Space) -> np.ndarray:
    """
    Rescale the action from [low, high] to [-1, 1]

    :param actions: action
    :param action_space: action space
    :return: rescaled action
    """
    if isinstance(action_space, spaces.Box):
        low, high = action_space.low, action_space.high
        return 2.0 * ((actions - low) / (high - low)) - 1.0
    elif isinstance(action_space, (spaces.Dict, spaces.Tuple)):
        list_spaces = action_space.spaces.values() if isinstance(action_space, spaces.Dict) else action_space.spaces
        dims = [get_action_dim(s) for s in list_spaces]
        split_actions = np.split(actions, np.cumsum(dims)[:-1], axis=-1)
        list_unscaled = [scale_actions(a, s) for a, s in zip(split_actions, list_spaces)]
        return np.concatenate(list_unscaled, axis=-1)
    else:
        # No scaling for discrete actions
        return actions


def unscale_actions(scaled_actions: np.ndarray, action_space: spaces.Space) -> np.ndarray:
    """
    Rescale the action from [-1, 1] to [low, high]
    (no need for symmetric action space)
    :param scaled_actions: Action to un-scale
    :param action_space: action space
    """
    if isinstance(action_space, spaces.Box):
        low, high = action_space.low, action_space.high
        return low + (0.5 * (scaled_actions + 1.0) * (high - low))
    elif isinstance(action_space, (spaces.Dict, spaces.Tuple)):
        list_spaces = action_space.spaces.values() if isinstance(action_space, spaces.Dict) else action_space.spaces
        dims = [get_action_dim(s) for s in list_spaces]
        split_actions = np.split(scaled_actions, np.cumsum(dims)[:-1], axis=-1)
        list_unscaled = [unscale_actions(a, s) for a, s in zip(split_actions, list_spaces)]
        return np.concatenate(list_unscaled, axis=-1)
    else:
        # No scaling for discrete actions
        return scaled_actions


def clip_actions(actions: np.ndarray, action_space: spaces.Space) -> np.ndarray:
    """
    Clip the action to value between [low, high]

    :param actions: action
    :param action_space: action space
    :return: clipped action
    """
    if isinstance(action_space, spaces.Box):
        return np.clip(actions, action_space.low, action_space.high)
    elif isinstance(action_space, (spaces.Dict, spaces.Tuple)):
        list_spaces = action_space.spaces.values() if isinstance(action_space, spaces.Dict) else action_space.spaces
        dims = [get_action_dim(s) for s in list_spaces]
        split_actions = np.split(actions, np.cumsum(dims)[:-1], axis=-1)
        list_clipped = [clip_actions(a, s) for a, s in zip(split_actions, list_spaces)]
        return np.concatenate(list_clipped, axis=-1)
    else:
        # No clipping for discrete actions
        return actions





