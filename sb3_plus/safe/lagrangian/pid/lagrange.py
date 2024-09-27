from collections import deque
from typing import Union

import torch as th
from stable_baselines3.common.type_aliases import Schedule

from sb3_plus.safe.lagrangian.common.lagrange import BaseLagrange


class PIDLagrange(BaseLagrange):
    """
    PID controller to control the lagrangian multiplier

    :param pid_kp: The proportional gain of the PID controller.
    :param pid_ki: The integral gain of the PID controller.
    :param pid_kd: The derivative gain of the PID controller.
    :param pid_d_delay: The delay of the derivative term.
    :param pid_delta_p_ema_alpha: The exponential moving average alpha of the delta_p.
    :param pid_delta_d_ema_alpha: The exponential moving average alpha of the delta_d.

    """
    def __init__(
            self,
            pid_kp: float = 0.1,
            pid_ki: float = 0.01,
            pid_kd: float = 0.01,
            pid_d_delay: int = 10,
            pid_delta_p_ema_alpha: float = 0.95,
            pid_delta_d_ema_alpha: float = 0.95,
            cost_threshold: Union[float, Schedule] = 0.0,
            multiplier_init: float = 0.001,
    ):
        super().__init__(cost_threshold, multiplier_init)
        self.pid_kp = pid_kp
        self.pid_ki = pid_ki
        self.pid_kd = pid_kd
        self.pid_d_delay = pid_d_delay
        self.pid_delta_p_ema_alpha = pid_delta_p_ema_alpha
        self.pid_delta_d_ema_alpha = pid_delta_d_ema_alpha

        self._cost_penalty: float = 0.0
        self._pid_i: float = self.multiplier_init
        self._cost_ds: deque[float] = deque(maxlen=self.pid_d_delay)
        self._cost_ds.append(0.0)
        self._delta_p: float = 0.0
        self._cost_d: float = 0.0

    def multiplier(self) -> th.Tensor:
        return th.tensor(self._cost_penalty)

    def update_multiplier(self, mean_ep_cost: float, current_progress_remaining: float, **kwargs) -> th.Tensor:
        self.cost_threshold = self.cost_threshold_scheduler(current_progress_remaining)

        delta = float(mean_ep_cost - self.cost_threshold)
        self._pid_i = max(0.0, self._pid_i + delta * self.pid_ki)

        a_p = self.pid_delta_p_ema_alpha
        self._delta_p *= a_p
        self._delta_p += (1 - a_p) * delta

        a_d = self.pid_delta_d_ema_alpha
        self._cost_d *= a_d
        self._cost_d += (1 - a_d) * float(mean_ep_cost)
        pid_d = max(0.0, self._cost_d - self._cost_ds[0])

        pid_o = self.pid_kp * self._delta_p + self._pid_i + self.pid_kd * pid_d
        self._cost_penalty = max(0.0, pid_o)
        self._cost_ds.append(self._cost_d)

        return th.tensor(pid_o)
