from gymnasium import spaces
from collections import OrderedDict
from functools import singledispatch
from typing import Union
import numpy as np


@singledispatch
def action_flatdim(space: spaces.Space) -> int:
    """
    Return the number of dimensions a flattened equivalent of this space
    would have.

    :param space: space
    :return: flatten dimension
    :raise NotImplementedError: if the space is not defined in ``gymnasium.spaces``
    """
    raise NotImplementedError(f"{space} space is not supported")


@action_flatdim.register
def _action_flatdim_box(space: spaces.Box) -> int:
    return int(np.prod(space.shape))


@action_flatdim.register
def _action_flatdim_discrete(space: spaces.Discrete) -> int:
    return 1  # A data point is an int


@action_flatdim.register
def _action_flatdim_multibinary(space: spaces.MultiBinary) -> int:
    # TODO: handle np.ndarray
    return int(space.n)  # Number of binary data points


@action_flatdim.register
def _action_flatdim_multidiscrete(space: spaces.MultiDiscrete) -> int:
    return int(len(space.nvec))  # Number of discrete data points


@action_flatdim.register
def _action_flatdim_dict(space: spaces.Dict) -> int:
    return int(sum([action_flatdim(s) for s in space.spaces.values()]))


@action_flatdim.register
def _action_flatdim_tuple(space: spaces.Tuple) -> int:
    return int(sum([action_flatdim(s) for s in space.spaces]))


UnflattenSpacePoint = Union[np.ndarray, int, tuple, dict]


@singledispatch
def action_flatten(space: spaces.Space, x: UnflattenSpacePoint) -> np.ndarray:
    """
    Flatten a data point from a space.

    :param space: space
    :param x: a point
    :return: 1D array
    :raise NotImplementedError: if the space is not defined in ``gym.spaces``
    """
    raise NotImplementedError(f"{space} space is not supported")


@action_flatten.register(spaces.Box)
@action_flatten.register(spaces.Discrete)
@action_flatten.register(spaces.MultiBinary)
@action_flatten.register(spaces.MultiDiscrete)
def _action_flatten_box(space, x: UnflattenSpacePoint) -> np.ndarray:
    return np.asarray(x, dtype=space.dtype).flatten()


@action_flatten.register
def _action_flatten_dict(space: spaces.Dict, x: UnflattenSpacePoint) -> np.ndarray:
    return np.concatenate(
        [action_flatten(s, x_part) for x_part, s in zip(x, space.spaces)]
    )


@action_flatten.register
def _action_flatten_tuple(space: spaces.Tuple, x: UnflattenSpacePoint) -> np.ndarray:
    return np.concatenate(
        [action_flatten(s, x_part) for x_part, s in zip(x, space.spaces)]
    )


@singledispatch
def action_unflatten(space: spaces.Space, x: np.ndarray) -> Union[np.ndarray, int, tuple, dict]:
    """
    Unflatten a data point from a space.

    This reverses the transformation applied by ``action_flatten()``. You must ensure
    that the ``space`` argument is the same as for the ``action_flatten()`` call.

    :param space: space
    :param x: a flattened point
    :return: returns a point with a structure that matches the space
    """
    raise NotImplementedError(f"{space} space is not supported")


@action_unflatten.register(spaces.Box)
@action_unflatten.register(spaces.MultiDiscrete)
@action_unflatten.register(spaces.MultiBinary)
def _action_unflatten_box(space, x: np.ndarray) -> np.ndarray:
    return np.asarray(x, dtype=space.dtype).reshape(space.shape)


@action_unflatten.register
def _action_unflatten_discrete(space: spaces.Discrete, x: np.ndarray) -> int:
    return int(x.flat[0])


@action_unflatten.register
def _action_unflatten_dict(space: spaces.Dict, x: np.ndarray) -> dict:
    dims = [action_flatdim(s) for s in space.spaces.values()]
    list_flattened = np.split(x, np.cumsum(dims)[:-1])
    list_unflattened = [
        (key, action_unflatten(s, flattened))
        for flattened, (key, s) in zip(list_flattened, space.spaces.items())
    ]
    return OrderedDict(list_unflattened)


@action_unflatten.register
def _action_unflatten_tuple(space: spaces.Tuple, x: np.ndarray) -> tuple:
    dims = [action_flatdim(s) for s in space.spaces]
    list_flattened = np.split(x, np.cumsum(dims)[:-1])
    list_unflattened = [
        action_unflatten(s, flattened)
        for flattened, s in zip(list_flattened, space.spaces)
    ]
    return tuple(list_unflattened)


@singledispatch
def action_flatten_space(space: spaces.Space) -> spaces.Space:
    raise NotImplementedError(f"{space} space is not supported")


@action_flatten_space.register(spaces.Box)
def _action_flatten_space_box(space: spaces.Box) -> spaces.Box:
    return spaces.Box(space.low.flatten(), space.high.flatten(), dtype=space.dtype)


@action_flatten_space.register(spaces.Discrete)
def _action_flatten_space_discrete(space: spaces.Discrete) -> spaces.Discrete:
    return space


@action_flatten_space.register(spaces.MultiBinary)
def _action_flatten_space_multibinary(space: spaces.MultiBinary) -> spaces.MultiBinary:
    return spaces.MultiBinary(action_flatdim(space))


@action_flatten_space.register(spaces.MultiDiscrete)
def _action_flatten_space_multidiscrete(space: spaces.MultiDiscrete) -> spaces.MultiDiscrete:
    return spaces.MultiDiscrete(action_flatdim(space), dtype=space.dtype)


@action_flatten_space.register(spaces.Tuple)
def _action_flatten_space_tuple(space: spaces.Tuple) -> spaces.Tuple:
    return spaces.Tuple(spaces=[action_flatten_space(s) for s in space.spaces])


@action_flatten_space.register(spaces.Dict)
def _action_flatten_space_dict(space: spaces.Dict) -> spaces.Dict:
    return spaces.Dict(
        spaces=OrderedDict(
            (key, action_flatten_space(space)) for key, space in space.spaces.items()
        )
    )
