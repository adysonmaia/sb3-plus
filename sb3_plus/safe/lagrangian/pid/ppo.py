from typing import Any, Dict, Optional, Type, Union, TypeVar

import torch as th
from stable_baselines3.common.type_aliases import GymEnv, Schedule, MaybeCallback

from sb3_plus.safe.lagrangian.common.ppo import BaseLagPPO
from sb3_plus.safe.lagrangian.pid.lagrange import PIDLagrange
from sb3_plus.safe.policies import SafeActorCriticPolicy

SelfCPPOPID = TypeVar("SelfCPPOPID", bound="CPPOPID")


class CPPOPID(BaseLagPPO):
    """
    PID version of Lagrangian Proximal Policy Optimization algorithm (PPO-Lag) (clip version)

    Paper: https://arxiv.org/abs/2007.03964

    :param policy: The policy model to use (MlpPolicy, CnnPolicy, ...)
    :param env: The environment to learn from (if registered in Gym, can be str)
    :param learning_rate: The learning rate, it can be a function
        of the current progress remaining (from 1 to 0)
    :param n_steps: The number of steps to run for each environment per update
        (i.e. rollout buffer size is n_steps * n_envs where n_envs is number of environment copies running in parallel)
        NOTE: n_steps * n_envs must be greater than 1 (because of the advantage normalization)
        See https://github.com/pytorch/pytorch/issues/29372
    :param batch_size: Minibatch size
    :param n_epochs: Number of epoch when optimizing the surrogate loss
    :param gamma: Discount factor
    :param gae_lambda: Factor for trade-off of bias vs variance for Generalized Advantage Estimator
    :param clip_range: Clipping parameter, it can be a function of the current progress
        remaining (from 1 to 0).
    :param clip_range_vf: Clipping parameter for the value function,
        it can be a function of the current progress remaining (from 1 to 0).
        This is a parameter specific to the OpenAI implementation. If None is passed (default),
        no clipping will be done on the value function.
        IMPORTANT: this clipping depends on the reward scaling.
    :param normalize_advantage: Whether to normalize or not the advantage
    :param ent_coef: Entropy coefficient for the loss calculation
    :param vf_coef: Value function coefficient for the loss calculation
    :param max_grad_norm: The maximum value for the gradient clipping
    :param use_sde: Whether to use generalized State Dependent Exploration (gSDE)
        instead of action noise exploration (default: False)
    :param sde_sample_freq: Sample a new noise matrix every n steps when using gSDE
        Default: -1 (only sample at the beginning of the rollout)
    :param target_kl: Limit the KL divergence between updates,
        because the clipping is not enough to prevent large update
        see issue #213 (cf https://github.com/hill-a/stable-baselines/issues/213)
        By default, there is no limit on the kl div.
    :param stats_window_size: Window size for the rollout logging, specifying the number of episodes to average
        the reported success rate, mean episode length, and mean reward over
    :param tensorboard_log: the log location for tensorboard (if None, no logging)
    :param policy_kwargs: additional arguments to be passed to the policy on creation
    :param verbose: Verbosity level: 0 for no output, 1 for info messages (such as device or wrappers used), 2 for
        debug messages
    :param seed: Seed for the pseudo random generators
    :param device: Device (cpu, cuda, ...) on which the code should be run.
        Setting it to auto, the code will be run on the GPU if possible.
    :param _init_setup_model: Whether or not to build the network at the creation of the instance

    :param cost_threshold: Cost return threshold
    :param lag_multiplier_init: Lagrange multiplier initial value
    :param clip_range_cvf: Clipping parameter for the cost value function,
        it can be a function of the current progress remaining (from 1 to 0).
        This is a parameter specific to the OpenAI implementation. If None is passed (default),
        no clipping will be done on the cost value function.
        IMPORTANT: this clipping depends on the reward scaling.
    :param cvf_coef: Cost value function coefficient for the loss calculation
    :param cost_gae_lambda: GAE lambda for cost advantage estimations
    :param cost_gamma: Discount factor for cost returns
    :param pid_kp: The proportional gain of the PID controller
    :param pid_ki: The integral gain of the PID controller
    :param pid_kd: The derivative gain of the PID controller
    :param pid_d_delay: The delay of the derivative term of the PID controller
    :param pid_delta_p_ema_alpha: The exponential moving average alpha of the delta_p in the PID controller
    :param pid_delta_d_ema_alpha: The exponential moving average alpha of the delta_d in the PID controller
    """

    def __init__(
            self,
            policy: Union[str, Type[SafeActorCriticPolicy]],
            env: Union[GymEnv, str],
            learning_rate: Union[float, Schedule] = 3e-4,
            n_steps: int = 2048,
            batch_size: int = 64,
            n_epochs: int = 10,
            gamma: float = 0.99,
            gae_lambda: float = 0.95,
            clip_range: Union[float, Schedule] = 0.2,
            clip_range_vf: Union[None, float, Schedule] = None,
            normalize_advantage: bool = True,
            ent_coef: float = 0.0,
            vf_coef: float = 0.5,
            max_grad_norm: float = 0.5,
            use_sde: bool = False,
            sde_sample_freq: int = -1,
            target_kl: Optional[float] = None,
            stats_window_size: int = 100,
            tensorboard_log: Optional[str] = None,
            policy_kwargs: Optional[Dict[str, Any]] = None,
            verbose: int = 0,
            seed: Optional[int] = None,
            device: Union[th.device, str] = "auto",
            _init_setup_model: bool = True,

            cost_threshold: Union[float, Schedule] = 0.0,
            lag_multiplier_init: float = 0.001,
            clip_range_cvf: Union[None, float, Schedule] = None,
            cvf_coef: float = 0.1,
            cost_gae_lambda: Optional[float] = None,
            cost_gamma: Optional[float] = None,
            pid_kp: float = 0.1,
            pid_ki: float = 0.01,
            pid_kd: float = 0.01,
            pid_d_delay: int = 10,
            pid_delta_p_ema_alpha: float = 0.95,
            pid_delta_d_ema_alpha: float = 0.95,
    ):
        lagrange_kwargs = dict(
            cost_threshold=cost_threshold,
            multiplier_init=lag_multiplier_init,
            pid_kp=pid_kp,
            pid_ki=pid_ki,
            pid_kd=pid_kd,
            pid_d_delay=pid_d_delay,
            pid_delta_p_ema_alpha=pid_delta_p_ema_alpha,
            pid_delta_d_ema_alpha=pid_delta_d_ema_alpha
        )

        super().__init__(
            policy,
            env,
            learning_rate=learning_rate,
            n_steps=n_steps,
            batch_size=batch_size,
            n_epochs=n_epochs,
            gamma=gamma,
            gae_lambda=gae_lambda,
            clip_range=clip_range,
            clip_range_vf=clip_range_vf,
            normalize_advantage=normalize_advantage,
            ent_coef=ent_coef,
            vf_coef=vf_coef,
            max_grad_norm=max_grad_norm,
            use_sde=use_sde,
            sde_sample_freq=sde_sample_freq,
            target_kl=target_kl,
            stats_window_size=stats_window_size,
            tensorboard_log=tensorboard_log,
            policy_kwargs=policy_kwargs,
            verbose=verbose,
            seed=seed,
            device=device,
            _init_setup_model=_init_setup_model,
            lagrange_class=PIDLagrange,
            lagrange_kwargs=lagrange_kwargs,
            cost_gae_lambda=cost_gae_lambda,
            cost_gamma=cost_gamma,
            clip_range_cvf=clip_range_cvf,
            cvf_coef=cvf_coef,
        )

    def learn(
        self: SelfCPPOPID,
        total_timesteps: int,
        callback: MaybeCallback = None,
        log_interval: int = 1,
        tb_log_name: str = "CPPOPID",
        reset_num_timesteps: bool = True,
        progress_bar: bool = False,
    ) -> SelfCPPOPID:
        super().learn(
            total_timesteps=total_timesteps,
            callback=callback,
            log_interval=log_interval,
            tb_log_name=tb_log_name,
            reset_num_timesteps=reset_num_timesteps,
            progress_bar=progress_bar,
        )
        return self
