import gymnasium.spaces
from gymnasium import spaces
from collections import OrderedDict
from functools import singledispatch
from typing import Union
import numpy as np

# TODO: check if flatdim and unflatten are necessary for gymnasium package


@singledispatch
def flatdim(space: spaces.Space, use_onehot: bool = False) -> int:
    """
    Return the number of dimensions a flattened equivalent of this space
    would have.

    :param space: space
    :param use_onehot: if one-hot encoding is used or not for representing the space's dimension.
                       ``gym.spaces.flatdim`` uses one-hot encoding
    :return: flatten dimension
    :raise NotImplementedError: if the space is not defined in ``gym.spaces``
    """
    raise NotImplementedError(f"{space} space is not supported")


@flatdim.register
def flatdim_box(space: spaces.Box, use_onehot: bool = False) -> int:
    return spaces.flatdim(space) if use_onehot else int(np.prod(space.shape))


@flatdim.register
def flatdim_discrete(space: spaces.Discrete, use_onehot: bool = False) -> int:
    return spaces.flatdim(space) if use_onehot else 1  # A data point is an int


@flatdim.register
def flatdim_multibinary(space: spaces.MultiBinary, use_onehot: bool = False) -> int:
    return spaces.flatdim(space) if use_onehot else int(space.n)  # Number of binary data points


@flatdim.register
def flatdim_multidiscrete(space: spaces.MultiDiscrete, use_onehot: bool = False) -> int:
    return spaces.flatdim(space) if use_onehot else int(len(space.nvec))  # Number of discrete data points


@flatdim.register
def flatdim_dict(space: spaces.Dict, use_onehot: bool = False) -> int:
    return spaces.flatdim(space) if use_onehot else int(sum([flatdim(s) for s in space.spaces.values()]))


@flatdim.register
def flatdim_tuple(space: spaces.Tuple, use_onehot: bool = False) -> int:
    return spaces.flatdim(space) if use_onehot else int(sum([flatdim(s) for s in space.spaces]))


UnflattenSpacePoint = Union[np.ndarray, int, tuple, dict]


@singledispatch
def flatten(space: spaces.Space, x: UnflattenSpacePoint, use_onehot: bool = False) -> np.ndarray:
    """
    Flatten a data point from a space.

    :param space: space
    :param x: a point
    :param use_onehot: if one-hot encoding is used or not for representing the data point.
                       ``gym.spaces.flatten`` uses one-hot encoding
    :return: 1D array
    :raise NotImplementedError: if the space is not defined in ``gym.spaces``
    """
    raise NotImplementedError(f"{space} space is not supported")


@flatten.register(spaces.Box)
@flatten.register(spaces.Discrete)
@flatten.register(spaces.MultiBinary)
@flatten.register(spaces.MultiDiscrete)
def flatten_box(space, x: UnflattenSpacePoint, use_onehot: bool = False) -> np.ndarray:
    return spaces.flatten(space, x) if use_onehot else np.asarray(x, dtype=space.dtype).flatten()


@flatten.register
def flatten_dict(space: spaces.Dict, x: UnflattenSpacePoint, use_onehot: bool = False) -> np.ndarray:
    if use_onehot:
        return spaces.flatten(space, x)
    else:
        return np.concatenate(
            [flatten(s, x_part, use_onehot=False) for x_part, s in zip(x, space.spaces)]
        )


@flatten.register
def flatten_tuple(space: spaces.Tuple, x: UnflattenSpacePoint, use_onehot: bool = False) -> np.ndarray:
    if use_onehot:
        return spaces.flatten(space, x)
    else:
        return np.concatenate(
            [flatten(s, x_part, use_onehot=False) for x_part, s in zip(x, space.spaces)]
        )


@singledispatch
def unflatten(space: spaces.Space, x: np.ndarray, use_onehot: bool = False) -> Union[np.ndarray, int, tuple, dict]:
    """
    Unflatten a data point from a space.

    This reverses the transformation applied by ``flatten()``. You must ensure
    that the ``space`` argument is the same as for the ``flatten()`` call.

    :param space: space
    :param x: a flattened point
    :param use_onehot: if one-hot encoding is used or not for representing the data point.
                       ``gym.spaces.unflatten`` uses one-hot encoding
    :return: returns a point with a structure that matches the space
    """
    raise NotImplementedError(f"{space} space is not supported")


@unflatten.register(spaces.Box)
@unflatten.register(spaces.MultiDiscrete)
@unflatten.register(spaces.MultiBinary)
def unflatten_box(space, x: np.ndarray, use_onehot: bool = False) -> np.ndarray:
    return spaces.unflatten(space, x) if use_onehot else np.asarray(x, dtype=space.dtype).reshape(space.shape)


@unflatten.register
def unflatten_discrete(space: spaces.Discrete, x: np.ndarray, use_onehot: bool = False) -> int:
    return spaces.unflatten(space, x) if use_onehot else int(x.flat[0])


@unflatten.register
def unflatten_dict(space: spaces.Dict, x: np.ndarray, use_onehot: bool = False) -> dict:
    if use_onehot:
        return gymnasium.spaces.unflatten(space, x)
    else:
        dims = [flatdim(s, use_onehot=False) for s in space.spaces.values()]
        list_flattened = np.split(x, np.cumsum(dims)[:-1])
        list_unflattened = [
            (key, unflatten(s, flattened, use_onehot=False))
            for flattened, (key, s) in zip(list_flattened, space.spaces.items())
        ]
        return OrderedDict(list_unflattened)


@unflatten.register
def unflatten_tuple(space: spaces.Tuple, x: np.ndarray, use_onehot: bool = False) -> tuple:
    if use_onehot:
        return gymnasium.spaces.unflatten(space, x)
    else:
        dims = [flatdim(s, use_onehot=False) for s in space.spaces]
        list_flattened = np.split(x, np.cumsum(dims)[:-1])
        list_unflattened = [
            unflatten(s, flattened, use_onehot=False)
            for flattened, s in zip(list_flattened, space.spaces)
        ]
        return tuple(list_unflattened)
