from sb3_plus.common.spaces import action_unflatten
from gymnasium import spaces
from gymnasium.core import ActType, ObsType, WrapperActType
import gymnasium as gym
import numpy as np


class MultiOutputEnv(gym.ActionWrapper[ObsType, np.ndarray, ActType]):
    """
    Wraps an environment to transform multi-output actions represented as a :class:`numpy.ndarray` into a dict or tuple
    """
    def action(self, action: np.ndarray) -> ActType:
        if isinstance(self.env.action_space, (spaces.Dict, spaces.Tuple)):
            return action_unflatten(self.env.action_space, action)
        else:
            return action
