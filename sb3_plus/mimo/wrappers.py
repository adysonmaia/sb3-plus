from sb3_plus.common.spaces import action_unflatten
from gymnasium import spaces
from gymnasium.core import ActType, ObsType
from stable_baselines3.common.vec_env import DummyVecEnv, SubprocVecEnv, VecEnv
from stable_baselines3.common.env_util import make_vec_env
from typing import Any, Callable, Dict, Optional, Type, Union
import gymnasium as gym
import numpy as np


WrapperActType = Union[np.ndarray, dict, tuple]


class MultiOutputEnv(gym.ActionWrapper[ObsType, WrapperActType, ActType]):
    """
    Wraps an environment to transform multi-output actions represented as a :class:`numpy.ndarray` into a dict or tuple
    """
    def action(self, action: WrapperActType) -> ActType:
        if isinstance(action, np.ndarray) and isinstance(self.env.action_space, (spaces.Dict, spaces.Tuple)):
            return action_unflatten(self.env.action_space, action)
        else:
            return action


def make_multioutput_env(
    env_id: Union[str, Callable[..., gym.Env]],
    n_envs: int = 1,
    seed: Optional[int] = None,
    start_index: int = 0,
    monitor_dir: Optional[str] = None,
    env_kwargs: Optional[Dict[str, Any]] = None,
    vec_env_cls: Optional[Type[Union[DummyVecEnv, SubprocVecEnv]]] = None,
    vec_env_kwargs: Optional[Dict[str, Any]] = None,
    monitor_kwargs: Optional[Dict[str, Any]] = None,
) -> VecEnv:
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
        wrapper_kwargs=None
    )
