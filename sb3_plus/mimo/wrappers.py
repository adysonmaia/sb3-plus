from typing import Any, Callable

import gymnasium as gym
import numpy as np
from gymnasium import spaces
from gymnasium.core import ActType, ObsType
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.vec_env import DummyVecEnv, SubprocVecEnv, VecEnv

from sb3_plus.common.spaces import action_unflatten

WrapperActType = np.ndarray | dict | tuple


class MultiOutputEnv(gym.ActionWrapper[ObsType, WrapperActType, ActType]):
    """
    Wraps an environment to transform multi-output actions represented as a :class:`numpy.ndarray` into a dict or tuple
    """

    def action(self, action: WrapperActType) -> ActType:
        if isinstance(action, np.ndarray) and isinstance(
            self.env.action_space, (spaces.Dict, spaces.Tuple)
        ):
            return action_unflatten(self.env.action_space, action)
        else:
            return action


def make_multioutput_env(
    env_id: str | Callable[..., gym.Env],
    n_envs: int = 1,
    seed: int | None = None,
    start_index: int = 0,
    monitor_dir: str | None = None,
    env_kwargs: dict[str, Any] | None = None,
    vec_env_cls: type[DummyVecEnv | SubprocVecEnv] | None = None,
    vec_env_kwargs: dict[str, Any] | None = None,
    monitor_kwargs: dict[str, Any] | None = None,
) -> VecEnv:
    """
    Create a wrapped vectorized environment with multiple outputs

    :param env_id: either the env ID, the env class or a callable returning an env
    :param n_envs: the number of environments you wish to have in parallel
    :param seed: the initial seed for the random number generator
    :param start_index: start rank index
    :param monitor_dir: Path to a folder where the monitor files will be saved.
        If None, no file will be written, however, the env will still be wrapped
        in a Monitor wrapper to provide additional information about training.
    :param env_kwargs: Optional keyword argument to pass to the env constructor
    :param vec_env_cls: A custom ``VecEnv`` class constructor. Default: None.
    :param vec_env_kwargs: Keyword arguments to pass to the ``VecEnv`` class constructor.
    :param monitor_kwargs: Keyword arguments to pass to the ``Monitor`` class constructor.
    :return: The wrapped environment
    """
    return make_vec_env(
        env_id=env_id,
        n_envs=n_envs,
        seed=seed,
        start_index=start_index,
        monitor_dir=monitor_dir,
        wrapper_class=MultiOutputEnv,
        env_kwargs=env_kwargs,
        vec_env_cls=vec_env_cls,
        vec_env_kwargs=vec_env_kwargs,
        monitor_kwargs=monitor_kwargs,
        wrapper_kwargs=None,
    )
