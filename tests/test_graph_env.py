import warnings

import numpy as np
import pytest
from gymnasium import spaces
from stable_baselines3 import A2C, PPO
from stable_baselines3.common.env_checker import check_env
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.evaluation import evaluate_policy
from stable_baselines3.common.on_policy_algorithm import OnPolicyAlgorithm
from stable_baselines3.common.vec_env import DummyVecEnv, SubprocVecEnv, VecEnv

from sb3_plus.gnn import ActorCriticGnnPolicy, GraphObservationWrapper, GraphSpace
from sb3_plus.gnn.common.envs import DummyGraphEnv


@pytest.mark.parametrize("min_nodes,max_nodes", [(5, 5), (5, 10)])
@pytest.mark.parametrize("flatten", [True, False])
def test_graph_env(min_nodes: int, max_nodes: int, flatten: bool):
    env = DummyGraphEnv(min_nodes=min_nodes, max_nodes=max_nodes)
    assert isinstance(env.observation_space, spaces.Graph)
    env = GraphObservationWrapper(env, max_nodes=max_nodes, flatten=flatten)
    assert isinstance(env.observation_space, GraphSpace)
    check_env(env, warn=flatten)

    obs, _ = env.reset(seed=100)
    assert isinstance(obs, dict)
    x: np.ndarray = obs[GraphSpace.NODES]
    if flatten:
        assert x.ndim == 1
    else:
        assert x.ndim > 1
    if x.ndim == 1:
        x = x.reshape(env.observation_space.max_nodes, -1)
    n_nodes = x.shape[0]
    assert min_nodes <= n_nodes <= max_nodes


@pytest.mark.parametrize("min_nodes,max_nodes", [(5, 5), (5, 10)])
@pytest.mark.parametrize("vec_env_class", [DummyVecEnv, SubprocVecEnv])
@pytest.mark.filterwarnings("ignore::UserWarning")
def test_vec_env(min_nodes: int, max_nodes: int, vec_env_class: type[VecEnv]):
    n_envs = 4
    vec_env = make_vec_env(
        lambda: DummyGraphEnv(min_nodes=min_nodes, max_nodes=max_nodes),
        n_envs=n_envs,
        vec_env_cls=vec_env_class,
        wrapper_class=GraphObservationWrapper,
        wrapper_kwargs=dict(max_nodes=max_nodes),
    )
    assert isinstance(vec_env.observation_space, GraphSpace)

    policy = ActorCriticGnnPolicy
    model = PPO(policy, vec_env, n_steps=64, seed=8)

    obs = vec_env.reset()
    assert isinstance(obs, dict)
    x = obs[GraphSpace.NODES]
    assert x.ndim >= 2
    batch_size = x.shape[0]
    assert batch_size == n_envs

    model.learn(64)
    evaluate_policy(model, model.get_env(), n_eval_episodes=5, warn=False)


@pytest.mark.parametrize("min_nodes,max_nodes", [(5, 5), (5, 10)])
@pytest.mark.parametrize("model_class", [PPO, A2C])
@pytest.mark.parametrize("pool_fn", [None, "mean", "add", "max"])
def test_gnn_policy(
    min_nodes: int,
    max_nodes: int,
    model_class: type[OnPolicyAlgorithm],
    pool_fn: str | None,
):
    env = DummyGraphEnv(min_nodes=min_nodes, max_nodes=max_nodes)
    env = GraphObservationWrapper(env, max_nodes=max_nodes)
    policy = ActorCriticGnnPolicy
    policy_kwargs = dict(
        gnn_pool_fn=pool_fn,
    )

    def train_model():
        model = model_class(
            policy, env, n_steps=64, seed=8, policy_kwargs=policy_kwargs
        )
        model.learn(128)
        evaluate_policy(model, model.get_env(), n_eval_episodes=5, warn=False)

    if min_nodes < max_nodes and pool_fn is None:
        with pytest.warns(UserWarning, match="A graph pooling function should be used"):
            train_model()
    else:
        with warnings.catch_warnings():
            warnings.simplefilter("error")
            train_model()
