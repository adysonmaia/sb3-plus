# SB3-Plus

Repository containing additional RL algorithms to [Stable-Baselines3](https://github.com/DLR-RM/stable-baselines3) library

## Installation

To install SB3-Plus with pip, execute:

```
pip install git+https://github.com/adysonmaia/sb3-plus#egg=sb3-plus
```

## Table of Contents

- [Multi-Input Multi-Output (MIMO) Environments](#multi-input-multi-output-mimo-environments)
- [Graph Neural Network (GNN) for RL Algorithms](#graph-neural-network-gnn-for-rl-algorithms)


## Documentation

### Multi-Input Multi-Output (MIMO) Environments

SB3-Plus supports [gymnasium](https://gymnasium.farama.org/) environments with multiple inputs and multiple outputs. Single input and single output is also supported by SB3-Plus.
That is:
- Multi-Input means that the observation space of an environment is represented as a [Dict](https://gymnasium.farama.org/api/spaces/composite/#gymnasium.spaces.Dict) or [Tuple](https://gymnasium.farama.org/api/spaces/composite/#gymnasium.spaces.Tuple) space.
- Single-Input refers to observation space as a [Box](https://gymnasium.farama.org/api/spaces/fundamental/#gymnasium.spaces.Box), [Discrete](https://gymnasium.farama.org/api/spaces/fundamental/#gymnasium.spaces.Discrete), [MultiBinary](https://gymnasium.farama.org/api/spaces/fundamental/#gymnasium.spaces.MultiBinary), or [MultiDiscrete](https://gymnasium.farama.org/api/spaces/fundamental/#gymnasium.spaces.MultiDiscrete) space. 
- Multi-Output refers to the action space modeled as a [Dict](https://gymnasium.farama.org/api/spaces/composite/#gymnasium.spaces.Dict) or [Tuple](https://gymnasium.farama.org/api/spaces/composite/#gymnasium.spaces.Tuple) space class. In this way, an environment can have a hybrid action space composed of continuous and discrete actions. However, SB3-Plus assumes that the multiple actions are mutually independent.
- Single-Output means that the action space is represented as a [Box](https://gymnasium.farama.org/api/spaces/fundamental/#gymnasium.spaces.Box), [Discrete](https://gymnasium.farama.org/api/spaces/fundamental/#gymnasium.spaces.Discrete), [MultiBinary](https://gymnasium.farama.org/api/spaces/fundamental/#gymnasium.spaces.MultiBinary), or [MultiDiscrete](https://gymnasium.farama.org/api/spaces/fundamental/#gymnasium.spaces.MultiDiscrete) space.

#### MIMO Policies
The ``policy`` argument of a RL algorithm is then used to specify input and output types. The following options are possible for a policy:
- 'MultiInputPolicy': multiple input and single output.
- 'MultiOutputPolicy': single input and multiple output.
- 'MIMOPolicy': multiple input and multiple output.
- 'MlpPolicy' or 'CnnPolicy' for single input and single output.

#### MIMO RL algorithms

The following RL algorithms work with MIMO environments 
- ``MultiOutputPPO`` is an extension of [PPO](https://stable-baselines3.readthedocs.io/en/master/modules/ppo.html) algorithm for multi-output environments.

#### Examples

In the following examples, we use a [gymnasium-hybrid](https://github.com/adysonmaia/gymnasium-hybrid) environment composed of hybrid actions.

To install SB3, SB3-Plus, and gymnasium-hybrid with pip, execute:

```
pip install 'stable-baselines3[extra]' \
    'git+https://github.com/adysonmaia/sb3-plus#egg=sb3-plus' \
    'git+https://github.com/adysonmaia/gymnasium-hybrid#egg=gymnasium-hybrid'
```

**Basic Usage: Training, Saving, Loading**
```python
from sb3_plus import MultiOutputPPO
from stable_baselines3.common.evaluation import evaluate_policy
import gymnasium as gym
import gymnasium_hybrid
import time

# Create environment
env = gym.make('Moving-v0', render_mode='rgb_array')

# Instantiate the agent
model = MultiOutputPPO(
    policy='MultiOutputPolicy',
    env=env,
    verbose=1,
    policy_kwargs=dict(
        net_arch=dict(pi=[252] * 4, vf=[252] * 4)
    )
)

# Train the agent and display a progress bar
model.learn(
    total_timesteps=int(2e5),
    progress_bar=True
)

# Save the agent
model.save("ppo_moving")
del model  # delete trained model to demonstrate loading

# Load the trained agent
model = MultiOutputPPO.load("ppo_moving", env=env)

# Evaluate the agent
mean_reward, std_reward = evaluate_policy(model, model.get_env(), n_eval_episodes=10)
print(f'Mean Reward {mean_reward} | Std Reward {std_reward}')

# Enjoy trained agent
vec_env = model.get_env()
obs = vec_env.reset()
for i in range(1000):
    action, _states = model.predict(obs, deterministic=True)
    obs, rewards, dones, info = vec_env.step(action)
    vec_env.render('human')
```

**Multiprocessing**
```python
from sb3_plus import MultiOutputPPO, make_multioutput_env
from stable_baselines3.common.vec_env import SubprocVecEnv
import gymnasium_hybrid


def main():
    # Create vectorized environment
    vec_env = make_multioutput_env('Moving-v0', n_envs=4, vec_env_cls=SubprocVecEnv)

    # Instantiate the agent
    model = MultiOutputPPO(
        policy='MultiOutputPolicy',
        env=vec_env,
        verbose=1,
        policy_kwargs=dict(
            net_arch=dict(pi=[252] * 4, vf=[252] * 4)
        )
    )

    # Train the agent and display a progress bar
    model.learn(
        total_timesteps=int(2e5),
        progress_bar=True
    )

    # Enjoy trained agent
    obs = vec_env.reset()
    for i in range(1000):
        action, _states = model.predict(obs, deterministic=True)
        obs, rewards, dones, info = vec_env.step(action)
        vec_env.render('human')
        
    vec_env.close()


if __name__ == '__main__':
    main()
```

**Multiprocessing with custom environment making**
```python
from sb3_plus import MultiOutputPPO, MultiOutputEnv
from stable_baselines3.common.vec_env import SubprocVecEnv
from stable_baselines3.common.utils import set_random_seed
import gymnasium_hybrid
import gymnasium as gym


def make_env(env_id: str, rank: int, seed: int = 0):
    def _init():
        env = gym.make(env_id, render_mode='rgb_array')
        # Wrapping env to transform multi-output actions from flatten numpy.ndarray into dict or tuple
        env = MultiOutputEnv(env)
        env.reset(seed=seed + rank)
        return env
    set_random_seed(seed)
    return _init


def main():
    env_id = 'Moving-v0'
    num_cpu = 4  # Number of processes to use
    # Create the vectorized environment
    vec_env = SubprocVecEnv([make_env(env_id, i) for i in range(num_cpu)])

    # Instantiate the agent
    model = MultiOutputPPO(
        policy='MultiOutputPolicy',
        env=vec_env,
        verbose=1,
        policy_kwargs=dict(
            net_arch=dict(pi=[252] * 4, vf=[252] * 4)
        )
    )

    # Train the agent and display a progress bar
    model.learn(
        total_timesteps=int(2e5),
        progress_bar=True
    )

    # Enjoy trained agent
    obs = vec_env.reset()
    for i in range(1000):
        action, _states = model.predict(obs, deterministic=True)
        obs, rewards, dones, info = vec_env.step(action)
        vec_env.render('human')
    vec_env.close()


if __name__ == '__main__':
    main()
```

---

### Graph Neural Network (GNN) for RL Algorithms

#### Environments with Graph-based Observations

SB3-Plus adds support of [gymnasium](https://gymnasium.farama.org/) environments with graph-based environments to [Stable-Baselines3](https://github.com/DLR-RM/stable-baselines3). However, the [Graph](https://gymnasium.farama.org/api/spaces/composite/#gymnasium.spaces.Graph) space of gymnasium is not directly supported by Stable-Baselines3, as its size varies with the number of nodes in a graph. Therefore, SB3-Plus offers the ``GraphObservationWrapper`` observation wrapper to transform the dynamic space into a fixed-size space. This transformed space uses dense data format and padding to fix the size of the graph representation. For this, it is necessary to specify the maximum number of nodes in a graph via the ``max_nodes`` argument during the initialization of the ``sb3_plus.gnn.GraphObservationWrapper`` class.

**Example:**

To install SB3 and SB3-Plus with GNN support, execute:

```
pip install pip install 'stable-baselines3[extra]' \
    'sb3-plus[gnn] @ git+https://github.com/adysonmaia/sb3-plus#egg=sb3-plus'
```

**Environment wrapping for graph-based observations**
```python
from stable_baselines3.common.env_checker import check_env

from sb3_plus.gnn import GraphObservationWrapper
from sb3_plus.gnn.common.envs import SimpleGraphEnv


def main():
    # Create the environment
    min_nodes = 5
    max_nodes = 10
    env = SimpleGraphEnv(min_nodes, max_nodes)
    print("Observation Space:", env.observation_space)
    
    # A warning should appear telling that Graph space is not supported
    check_env(env, warn=True)

    # Wrapping the environment to have an observation space with fixed size
    env = GraphObservationWrapper(env, max_nodes=max_nodes)
    print("Fixed-size Observation Space:", env.observation_space)
    # No warning should appear
    check_env(env, warn=True)

    # Running N episodes with random actions
    total_steps = 0
    total_nodes = 0
    n_episodes = 10
    for i in range(n_episodes):
        obs, _ = env.reset(seed=100 + i)
        done = False
        while not done:
            # a mask is returned in the transformed observations 
            # to distinguish real from fake nodes used for padding
            n_nodes = int(obs["nodes_mask"].sum())
            total_nodes += n_nodes
            action = env.action_space.sample()
            obs, reward, terminated, truncated, info = env.step(action)
            done = terminated or truncated
            total_steps += 1
    print(
        f"{n_episodes} episodes finished with {total_nodes/total_steps:.1f} graph nodes per step"
    )


if __name__ == "__main__":
    main()

```


### Graph Neural Network (GNN) Policy

SB3-Plus provides a graph-based policy designed for Actor-Critic RL algorithms via the `sb3_plus.gnn.ActorCriticGnnPolicy` class. The network in this class can be divided into tree parts:

1. A **GNN model** to extract node features from the observations. SB3-Plus supports the GNN models from the [PyTorch Geometric (PyG)](https://pytorch-geometric.readthedocs.io/en/latest/) library. To setting the model, two initialization parameters are available in the ``ActorCriticGnnPolicy`` class:
    - ``gnn_class``: the class of the GNN model. It should be a sub-class of the ``BasicGNN`` class of PyG, such as [GCN](https://pytorch-geometric.readthedocs.io/en/latest/generated/torch_geometric.nn.models.GCN.html#torch_geometric.nn.models.GCN), [GraphSAGE](https://pytorch-geometric.readthedocs.io/en/latest/generated/torch_geometric.nn.models.GraphSAGE.html#torch_geometric.nn.models.GraphSAGE), [GIN](https://pytorch-geometric.readthedocs.io/en/latest/generated/torch_geometric.nn.models.GIN.html#torch_geometric.nn.models.GIN), [GAT](https://pytorch-geometric.readthedocs.io/en/latest/generated/torch_geometric.nn.models.GAT.html#torch_geometric.nn.models.GAT), and [EdgeCNN](https://pytorch-geometric.readthedocs.io/en/latest/generated/torch_geometric.nn.models.EdgeCNN.html#torch_geometric.nn.models.EdgeCNN).
    - ``gnn_kwargs``: dict with initialization parameters of the GNN class specified by ``gnn_class``.
2. An optional **pooling function** to aggregate the features of all nodes extracted by the GNN model. The name of the function can be specified in the ``gnn_pool_fn`` parameter of the ``ActorCriticGnnPolicy`` class, and the following options are possible:
    - ``'add'``: it uses the [global_add_pool](https://pytorch-geometric.readthedocs.io/en/latest/generated/torch_geometric.nn.pool.global_add_pool.html#torch_geometric.nn.pool.global_add_pool) function to output a one-dimensional vector by summing the features of all real nodes across the feature dimensions.
    - ``'mean'``: it uses the [global_mean_pool](https://pytorch-geometric.readthedocs.io/en/latest/generated/torch_geometric.nn.pool.global_mean_pool.html#torch_geometric.nn.pool.global_mean_pool) function to output a one-dimensional vector by averaging the features of all real nodes across the feature dimensions.
    - ``'max'``: it uses the [global_max_pool](https://pytorch-geometric.readthedocs.io/en/latest/generated/torch_geometric.nn.pool.global_max_pool.html#torch_geometric.nn.pool.global_max_pool) function to output a one-dimensional vector by taking the features-wise maximum across all real nodes.
    - `None`: no pooling function is used. A one-dimensional vector is output by flattening the features of all nodes, including real and fake nodes used for padding. Therefore, this option **should be avoided** if the environment has an observation space with a variable number of nodes.
3. A **Readout Model** that maps the extracted and aggregated features to actions/value. Its architecture is similar to the fully-connected network of the standard ``ActorCriticPolicy`` class and is specified by two initialization parameters:
    - ``readout_net_arch``: the specification of the policy and value networks. See the [net_arch](https://stable-baselines3.readthedocs.io/en/master/guide/custom_policy.html) argument of ``ActorCriticPolicy`` for more details.
    - ``readout_act_fn``: the class of the activation function.


**Example: Training, Saving, Loading**
```python
import torch
from stable_baselines3 import PPO
from stable_baselines3.common.evaluation import evaluate_policy
from torch_geometric.nn.models import GAT

from sb3_plus.gnn import ActorCriticGnnPolicy, GraphObservationWrapper
from sb3_plus.gnn.common.envs import SimpleGraphEnv


def main():
    # Create the environment
    min_nodes = 5
    max_nodes = 8
    env = SimpleGraphEnv(min_nodes, max_nodes)
    env = GraphObservationWrapper(env, max_nodes=max_nodes)

    # Instantiate the agent
    policy_kwargs = dict(
        gnn_class=GAT,
        gnn_kwargs=dict(
            hidden_channels=16,
            num_layers=1,
            out_channels=16,
            residual=True,
            v2=True,
        ),
        gnn_pool_fn="mean",
        readout_net_arch=dict(pi=[32] * 2, vf=[32] * 2),
        readout_act_fn=torch.nn.ReLU,
        share_features_extractor=False,
    )
    model = PPO(
        policy=ActorCriticGnnPolicy,
        env=env,
        verbose=1,
        policy_kwargs=policy_kwargs,
    )

    # Train the agent and display a progress bar
    model.learn(total_timesteps=int(5e5), progress_bar=True)

    # Save the agent
    model.save("ppo_gnn")
    del model  # delete trained model to demonstrate loading

    # Load the trained agent
    model = PPO.load("ppo_gnn", env=env)

    # Evaluate the agent
    mean_reward, std_reward = evaluate_policy(
        model, model.get_env(), n_eval_episodes=10
    )
    print(f"Mean Reward {mean_reward} | Std Reward {std_reward}")

    # Enjoy trained agent
    vec_env = model.get_env()
    obs = vec_env.reset()
    for i in range(10):
        action, _states = model.predict(obs, deterministic=True)
        obs, rewards, dones, info = vec_env.step(action)
        print(f"step: {i}, rewards: {rewards}, dones: {dones}")


if __name__ == "__main__":
    main()
```

---