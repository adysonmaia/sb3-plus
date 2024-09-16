from sb3_plus import MultiOutputPPO, make_multioutput_env
from sb3_plus.common.spaces import action_flatdim
from stable_baselines3.common.evaluation import evaluate_policy
from stable_baselines3.common.vec_env import DummyVecEnv, SubprocVecEnv
from gymnasium import spaces
from collections import OrderedDict
from typing import Dict, Tuple, Any, Optional
import gymnasium as gym
import numpy as np
import pytest


class DummyDictEnv(gym.Env):
    """Custom Environment for testing purposes only"""

    metadata = {"render.modes": ["human"]}

    def __init__(self, multi_input: bool = True, max_step: int = 10):
        super().__init__()
        self.max_step = max_step
        self._step_count = 0
        obs_space = spaces.Box(low=-1, high=1, shape=(2,), dtype=np.float32)
        if multi_input:
            self.observation_space = spaces.Dict({
                'obs': obs_space
            })
        else:
            self.observation_space = obs_space

        self.action_space = spaces.Dict({
            'vec': spaces.Box(low=-1, high=1, shape=(2,), dtype=np.float32),
            'discrete': spaces.Discrete(4),
            'binary': spaces.MultiBinary(4),
            'multi_discrete': spaces.MultiDiscrete([2, 3, 4])
        })

    def step(self, action: Dict) -> Tuple[Any, float, bool, bool, Dict]:
        self._step_count += 1
        reward = 0.0
        done = truncated = (self._step_count > self.max_step)
        return self.observation_space.sample(), reward, done, truncated, {}

    def reset(self, *, seed: Optional[int] = None, options: Optional[Dict] = None):
        if seed is not None:
            self.observation_space.seed(seed)
        self._step_count = 0
        return self.observation_space.sample(), {}


class ToTupleEnvWrapper(gym.ActionWrapper):
    def __init__(self, env: gym.Env):
        super().__init__(env)
        assert isinstance(env.action_space, spaces.Dict)
        self.action_space = spaces.Tuple(env.action_space.spaces.values())

    def action(self, action: Tuple) -> Dict:
        action_space_keys = self.env.action_space.spaces.keys()
        return OrderedDict(zip(action_space_keys, action))


@pytest.mark.parametrize("multi_input", [True, False])
def test_dict_env(multi_input: bool):
    env = DummyDictEnv(multi_input=multi_input)
    action_space = env.action_space
    assert isinstance(action_space, spaces.Dict)
    action_space_keys = action_space.spaces.keys()
    policy = 'MIMOPolicy' if multi_input else 'MultiOutputPolicy'
    model = MultiOutputPPO(policy, env, n_steps=64, seed=8)
    obs, _ = env.reset(seed=8)
    action, _ = model.predict(obs)
    assert isinstance(action, OrderedDict)
    assert all(k in action_space_keys for k in action.keys())

    model.learn(64)
    evaluate_policy(model, model.get_env(), n_eval_episodes=10, warn=False)


@pytest.mark.parametrize("multi_input", [True, False])
def test_tuple_env(multi_input: bool):
    env = DummyDictEnv(multi_input=multi_input)
    env = ToTupleEnvWrapper(env)
    action_space = env.action_space
    assert isinstance(action_space, spaces.Tuple)
    policy = 'MIMOPolicy' if multi_input else 'MultiOutputPolicy'
    model = MultiOutputPPO(policy, env, n_steps=64, seed=8)
    obs, _ = env.reset(seed=8)
    action, _ = model.predict(obs)
    assert isinstance(action, Tuple)
    assert len(action) == len(env.action_space.spaces)

    model.learn(64)
    evaluate_policy(model, model.get_env(), n_eval_episodes=10, warn=False)


@pytest.mark.parametrize("multi_input", [True, False])
@pytest.mark.parametrize("vec_env_class", [DummyVecEnv, SubprocVecEnv])
def test_vec_env(multi_input: bool, vec_env_class):
    n_envs = 4
    vec_env = make_multioutput_env(lambda: DummyDictEnv(multi_input=multi_input), n_envs=n_envs, vec_env_cls=vec_env_class)
    action_space = vec_env.action_space
    assert isinstance(action_space, spaces.Dict)

    policy = 'MIMOPolicy' if multi_input else 'MultiOutputPolicy'
    model = MultiOutputPPO(policy, vec_env, n_steps=64, seed=8)

    obs = vec_env.reset()
    action, _ = model.predict(obs)
    assert isinstance(action, np.ndarray)
    assert action.shape[0] == n_envs
    assert action.shape[0] == action_flatdim(action_space)

    model.learn(64)
    evaluate_policy(model, model.get_env(), n_eval_episodes=5, warn=False)
