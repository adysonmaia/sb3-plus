from abc import ABC, abstractmethod
from typing import Union

import torch as th
from stable_baselines3.common.type_aliases import Schedule
from stable_baselines3.common.utils import get_schedule_fn


class BaseLagrange(ABC):
    """
    Base class to implement methods to obtain and update the lagrange multiplier

    :param cost_threshold: cost threshold
    :param multiplier_init: multiplier initial value
    """
    def __init__(
            self,
            cost_threshold: Union[float, Schedule] = 0.0,
            multiplier_init: float = 0.0,
    ):
        self.cost_threshold = cost_threshold
        self.cost_threshold_scheduler = get_schedule_fn(self.cost_threshold)
        self.multiplier_init = max(0.0, multiplier_init)

    @abstractmethod
    def multiplier(self) -> th.Tensor:
        """
        Get Lagrange multiplier
        :return: penalty multiplier
        """
        pass

    @abstractmethod
    def update_multiplier(self, mean_ep_cost: float, current_progress_remaining: float, **kwargs) -> th.Tensor:
        """
        Update the lagrange multiplier (lambda) based on episode cost
        :param mean_ep_cost: mean episode cost
        :param current_progress_remaining:
        :return: penalty loss
        """
        pass
