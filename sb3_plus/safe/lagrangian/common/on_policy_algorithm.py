import sys
import time
from collections import deque
from typing import Any, Dict, List, Optional, Tuple, Type, TypeVar, Union

import numpy as np
import torch as th
from gymnasium import spaces
from stable_baselines3.common.base_class import BaseAlgorithm
from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.type_aliases import GymEnv, MaybeCallback, Schedule
from stable_baselines3.common.utils import obs_as_tensor, safe_mean
from stable_baselines3.common.vec_env import VecEnv

from sb3_plus.safe.buffers import SafeRolloutBuffer, SafeDictRolloutBuffer
from sb3_plus.safe.lagrangian.common.lagrange import BaseLagrange
from sb3_plus.safe.lagrangian.naive.lagrange import Lagrange
from sb3_plus.safe.policies import SafeActorCriticPolicy
from sb3_plus.safe.type_aliases import PENALTY_COST_INFO_KEY

SelfLagOnPolicyAlgorithm = TypeVar("SelfLagOnPolicyAlgorithm", bound="LagOnPolicyAlgorithm")


class LagOnPolicyAlgorithm(BaseAlgorithm):
    """
    The base for On-Policy algorithms (ex: A2C/PPO).

    :param policy: The policy model to use (MlpPolicy, CnnPolicy, ...)
    :param env: The environment to learn from (if registered in Gym, can be str)
    :param learning_rate: The learning rate, it can be a function
        of the current progress remaining (from 1 to 0)
    :param n_steps: The number of steps to run for each environment per update
        (i.e. batch size is n_steps * n_env where n_env is number of environment copies running in parallel)
    :param gamma: Discount factor
    :param gae_lambda: Factor for trade-off of bias vs variance for Generalized Advantage Estimator.
        Equivalent to classic advantage when set to 1.
    :param ent_coef: Entropy coefficient for the loss calculation
    :param vf_coef: Value function coefficient for the loss calculation
    :param max_grad_norm: The maximum value for the gradient clipping
    :param use_sde: Whether to use generalized State Dependent Exploration (gSDE)
        instead of action noise exploration (default: False)
    :param sde_sample_freq: Sample a new noise matrix every n steps when using gSDE
        Default: -1 (only sample at the beginning of the rollout)
    :param stats_window_size: Window size for the rollout logging, specifying the number of episodes to average
        the reported success rate, mean episode length, and mean reward over
    :param tensorboard_log: the log location for tensorboard (if None, no logging)
    :param monitor_wrapper: When creating an environment, whether to wrap it
        or not in a Monitor wrapper.
    :param policy_kwargs: additional arguments to be passed to the policy on creation
    :param verbose: Verbosity level: 0 for no output, 1 for info messages (such as device or wrappers used), 2 for
        debug messages
    :param seed: Seed for the pseudo random generators
    :param device: Device (cpu, cuda, ...) on which the code should be run.
        Setting it to auto, the code will be run on the GPU if possible.
    :param _init_setup_model: Whether or not to build the network at the creation of the instance
    :param supported_action_spaces: The action spaces supported by the algorithm.

    :param lagrange_class: class implementing a lagrange-base algorithm
    :param lagrange_kwargs: additional arguments to be passed to the lagrange on creation
    :param cost_gae_lambda: GAE lambda for cost advantage estimations
    :param cost_gamma: Discount factor for cost returns
    """

    lagrange: BaseLagrange

    def __init__(
        self,
        policy: Union[str, Type[SafeActorCriticPolicy]],
        env: Union[GymEnv, str],
        learning_rate: Union[float, Schedule],
        n_steps: int,
        gamma: float,
        gae_lambda: float,
        ent_coef: float,
        vf_coef: float,
        max_grad_norm: float,
        use_sde: bool,
        sde_sample_freq: int,
        stats_window_size: int = 100,
        tensorboard_log: Optional[str] = None,
        monitor_wrapper: bool = True,
        policy_kwargs: Optional[Dict[str, Any]] = None,
        verbose: int = 0,
        seed: Optional[int] = None,
        device: Union[th.device, str] = "auto",
        _init_setup_model: bool = True,
        supported_action_spaces: Optional[Tuple[Type[spaces.Space], ...]] = None,

        lagrange_class: Type[BaseLagrange] = Lagrange,
        lagrange_kwargs: Optional[Dict[str, Any]] = None,
        cost_gae_lambda: Optional[float] = None,
        cost_gamma: Optional[float] = None,
    ):

        super().__init__(
            policy=policy,
            env=env,
            learning_rate=learning_rate,
            policy_kwargs=policy_kwargs,
            verbose=verbose,
            device=device,
            use_sde=use_sde,
            sde_sample_freq=sde_sample_freq,
            support_multi_env=True,
            seed=seed,
            stats_window_size=stats_window_size,
            tensorboard_log=tensorboard_log,
            supported_action_spaces=supported_action_spaces,
            monitor_wrapper=monitor_wrapper
        )

        self.n_steps = n_steps
        self.gamma = gamma
        self.gae_lambda = gae_lambda
        self.ent_coef = ent_coef
        self.vf_coef = vf_coef
        self.max_grad_norm = max_grad_norm
        self.rollout_buffer = None

        self.lagrange_class = lagrange_class
        self.lagrange_kwargs = lagrange_kwargs
        self.cost_gae_lambda = cost_gae_lambda if cost_gae_lambda is not None else self.gae_lambda
        self.cost_gamma = cost_gamma if cost_gamma is not None else self.gamma

        self._ep_costs: deque = deque(maxlen=stats_window_size)
        self._ep_cost_returns: deque = deque(maxlen=stats_window_size)
        self._current_costs: Optional[np.ndarray] = None
        self._current_cost_returns: Optional[np.ndarray] = None
        self._current_env_steps: Optional[np.ndarray] = None

        if _init_setup_model:
            self._setup_model()

    def _setup_model(self) -> None:
        self._setup_lr_schedule()
        self.set_random_seed(self.seed)

        buffer_cls = SafeDictRolloutBuffer if isinstance(self.observation_space, spaces.Dict) else SafeRolloutBuffer

        self.rollout_buffer = buffer_cls(
            self.n_steps,
            self.observation_space,
            self.action_space,
            device=self.device,
            gamma=self.gamma,
            gae_lambda=self.gae_lambda,
            n_envs=self.n_envs,
            cost_gamma=self.cost_gamma,
            cost_gae_lambda=self.cost_gae_lambda
        )
        # pytype:disable=not-instantiable
        self.policy = self.policy_class(  # type: ignore[assignment]
            self.observation_space,
            self.action_space,
            self.lr_schedule,
            use_sde=self.use_sde,
            **self.policy_kwargs
        )
        # pytype:enable=not-instantiable
        self.policy = self.policy.to(self.device)

        if self.lagrange_kwargs is None:
            self.lagrange_kwargs = {}
        self.lagrange = self.lagrange_class(**self.lagrange_kwargs)

    def collect_rollouts(
        self,
        env: VecEnv,
        callback: BaseCallback,
        rollout_buffer: SafeRolloutBuffer,
        n_rollout_steps: int,
    ) -> bool:
        """
        Collect experiences using the current policy and fill a ``RolloutBuffer``.
        The term rollout here refers to the model-free notion and should not
        be used with the concept of rollout used in model-based RL or planning.

        :param env: The training environment
        :param callback: Callback that will be called at each step
            (and at the beginning and end of the rollout)
        :param rollout_buffer: Buffer to fill with rollouts
        :param n_rollout_steps: Number of experiences to collect per environment
        :return: True if function returned with at least `n_rollout_steps`
            collected, False if callback terminated rollout prematurely.
        """
        assert self._last_obs is not None, "No previous observation was provided"
        # Switch to eval mode (this affects batch norm / dropout)
        self.policy.set_training_mode(False)

        n_steps = 0
        rollout_buffer.reset()
        self._ep_costs.clear()
        self._ep_cost_returns.clear()
        # Sample new weights for the state dependent exploration
        if self.use_sde:
            self.policy.reset_noise(env.num_envs)

        callback.on_rollout_start()

        while n_steps < n_rollout_steps:
            if self.use_sde and self.sde_sample_freq > 0 and n_steps % self.sde_sample_freq == 0:
                # Sample a new noise matrix
                self.policy.reset_noise(env.num_envs)

            with th.no_grad():
                # Convert to pytorch tensor or to TensorDict
                obs_tensor = obs_as_tensor(self._last_obs, self.device)
                actions, values, cost_values, log_probs = self.policy(obs_tensor)
            actions = actions.cpu().numpy()

            # Rescale and perform action
            clipped_actions = actions
            # Clip the actions to avoid out of bound error
            if isinstance(self.action_space, spaces.Box):
                clipped_actions = np.clip(actions, self.action_space.low, self.action_space.high)

            new_obs, rewards, dones, infos = env.step(clipped_actions)

            self.num_timesteps += env.num_envs

            # Give access to local variables
            callback.update_locals(locals())
            if callback.on_step() is False:
                return False

            self._update_info_buffer(infos)
            n_steps += 1

            if isinstance(self.action_space, spaces.Discrete):
                # Reshape in case of discrete action
                actions = actions.reshape(-1, 1)

            costs = np.zeros(shape=rewards.shape, dtype=np.float32)

            # Handle timeout by bootstraping with value function
            # see GitHub issue #633
            for idx, done in enumerate(dones):
                costs[idx] = infos[idx].get(PENALTY_COST_INFO_KEY, 0.0)

                if (
                    done
                    and infos[idx].get("terminal_observation") is not None
                    and infos[idx].get("TimeLimit.truncated", False)
                ):
                    terminal_obs = self.policy.obs_to_tensor(infos[idx]["terminal_observation"])[0]
                    with th.no_grad():
                        terminal_value = self.policy.predict_values(terminal_obs)[0]
                        terminal_cost_value = self.policy.predict_cost_values(terminal_obs)[0]
                    rewards[idx] += self.gamma * terminal_value
                    costs[idx] += self.cost_gamma * terminal_cost_value

            self._current_costs += costs
            self._current_cost_returns += costs * np.power(self.cost_gamma, self._current_env_steps)
            self._current_env_steps += 1
            dones_mask = dones.astype(bool)
            if np.any(dones_mask):
                self._ep_costs.extend(self._current_costs[dones_mask])
                self._ep_cost_returns.extend(self._current_cost_returns[dones_mask])
                # reset done environments
                not_dones = 1 - dones
                self._current_costs *= not_dones
                self._current_cost_returns *= not_dones
                self._current_env_steps *= not_dones

            rollout_buffer.add(
                self._last_obs,  # type: ignore[arg-type]
                actions,
                rewards,
                costs,
                self._last_episode_starts,  # type: ignore[arg-type]
                values,
                cost_values,
                log_probs)
            self._last_obs = new_obs  # type: ignore[assignment]
            self._last_episode_starts = dones

        with th.no_grad():
            # Compute value for the last timestep
            values = self.policy.predict_values(obs_as_tensor(new_obs, self.device))  # type: ignore[arg-type]
            cost_values = self.policy.predict_cost_values(obs_as_tensor(new_obs, self.device))

        rollout_buffer.compute_returns_and_advantage(
            last_values=values,
            last_cost_values=cost_values,
            dones=dones
        )

        callback.on_rollout_end()

        return True

    def train(self) -> None:
        """
        Consume current rollout data and update policy parameters.
        Implemented by individual algorithms.
        """
        raise NotImplementedError

    def train_penalty(self) -> None:
        """
        Update penalty policy parameters based on mean episode cost of current rollout data

        """
        mean_ep_cost = 0.0
        if len(self._ep_cost_returns) > 0:
            mean_ep_cost = float(safe_mean(self._ep_costs))
        loss = self.lagrange.update_multiplier(mean_ep_cost, self._current_progress_remaining)

        # Logs
        self.logger.record("train_penalty/lag_multiplier", self.lagrange.multiplier().item())
        self.logger.record("train_penalty/lag_multiplier_loss", loss.item())
        self.logger.record("train_penalty/cost_threshold", self.lagrange.cost_threshold)

    def learn(
        self: SelfLagOnPolicyAlgorithm,
        total_timesteps: int,
        callback: MaybeCallback = None,
        log_interval: int = 1,
        tb_log_name: str = "LagOnPolicyAlgorithm",
        reset_num_timesteps: bool = True,
        progress_bar: bool = False,
    ) -> SelfLagOnPolicyAlgorithm:
        iteration = 0

        total_timesteps, callback = self._setup_learn(
            total_timesteps,
            callback,
            reset_num_timesteps,
            tb_log_name,
            progress_bar,
        )

        callback.on_training_start(locals(), globals())

        assert self.env is not None

        num_envs = 1
        if isinstance(self.env, VecEnv):
            num_envs = self.env.num_envs
        self._current_costs = np.zeros(num_envs, dtype=np.float32)
        self._current_cost_returns = np.zeros(num_envs, dtype=np.float32)
        self._current_env_steps = np.zeros(num_envs, dtype=np.float32)

        while self.num_timesteps < total_timesteps:
            continue_training = self.collect_rollouts(self.env, callback, self.rollout_buffer, n_rollout_steps=self.n_steps)

            if continue_training is False:
                break

            iteration += 1
            self._update_current_progress_remaining(self.num_timesteps, total_timesteps)

            # Display training infos
            if log_interval is not None and iteration % log_interval == 0:
                assert self.ep_info_buffer is not None
                time_elapsed = max((time.time_ns() - self.start_time) / 1e9, sys.float_info.epsilon)
                fps = int((self.num_timesteps - self._num_timesteps_at_start) / time_elapsed)
                self.logger.record("time/iterations", iteration, exclude="tensorboard")
                if len(self.ep_info_buffer) > 0 and len(self.ep_info_buffer[0]) > 0:
                    self.logger.record("rollout/ep_rew_mean", safe_mean([ep_info["r"] for ep_info in self.ep_info_buffer]))
                    self.logger.record("rollout/ep_cost_mean", safe_mean(self._ep_costs))
                    self.logger.record("rollout/ep_len_mean", safe_mean([ep_info["l"] for ep_info in self.ep_info_buffer]))
                self.logger.record("time/fps", fps)
                self.logger.record("time/time_elapsed", int(time_elapsed), exclude="tensorboard")
                self.logger.record("time/total_timesteps", self.num_timesteps, exclude="tensorboard")
                self.logger.dump(step=self.num_timesteps)

            self.train_penalty()
            self.train()

        callback.on_training_end()

        return self

    def _get_torch_save_params(self) -> Tuple[List[str], List[str]]:
        state_dicts = ["policy", "policy.optimizer"]

        return state_dicts, []
