# SB3-Plus

Repository containing additional RL algorithms to [StableBaselines3](https://github.com/DLR-RM/stable-baselines3) library

## Installation

To install SB3-Plus with pip, execute:

```
pip install git+https://github.com/adysonmaia/sb3-plus#egg=sb3-plus
```

## Features

**RL Algorithms**:
- Multi-Output PPO (MultiOutputPPO)


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