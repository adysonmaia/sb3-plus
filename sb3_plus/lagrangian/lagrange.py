
from stable_baselines3.common.type_aliases import Schedule
from stable_baselines3.common.utils import get_schedule_fn, update_learning_rate
from typing import Dict, Type, Union, Optional, Any
from torch import nn
import torch as th
import numpy as np


class Lagrange:
    """
    This class implements methods to obtain and update the lagrange multiplier

    :param cost_threshold: cost threshold
    :param multiplier_init: multiplier initial value
    :param learning_rate: learning rate
    :param max_grad_norm: maximum value for the gradient clipping
    :param optimizer_class: optimizer class
    :param optimizer_kwargs: optimizer parameters
    """

    def __init__(
            self,
            cost_threshold: Union[float, Schedule] = 0.0,
            multiplier_init: float = 0.0,
            learning_rate: Union[float, Schedule] = 3e-4,
            max_grad_norm: float = 20.0,
            optimizer_class: Type[th.optim.Optimizer] = th.optim.Adam,
            optimizer_kwargs: Optional[Dict[str, Any]] = None,
    ):
        self.cost_threshold = cost_threshold
        self.cost_threshold_scheduler = get_schedule_fn(self.cost_threshold)
        self.multiplier_init = max(0.0, multiplier_init)
        self.max_grad_norm = max_grad_norm
        self.optimizer_class = optimizer_class
        self.learning_rate = learning_rate
        self.lr_schedule = get_schedule_fn(self.learning_rate)

        if optimizer_kwargs is None:
            optimizer_kwargs = {}
            # Small values to avoid NaN in Adam optimizer
            if optimizer_class == th.optim.Adam:
                optimizer_kwargs["eps"] = 1e-5
        self.optimizer_kwargs = optimizer_kwargs

        # Setup lagrange multiplier optimizer
        self.multiplier_net = LagMultiplier(init_value=multiplier_init)
        self.optimizer = self.optimizer_class(self.multiplier_net.parameters(),
                                              lr=self.lr_schedule(1),
                                              **self.optimizer_kwargs)

    def multiplier(self) -> th.Tensor:
        """
        Get Lagrange multiplier
        :return: penalty multiplier
        """
        return self.multiplier_net()

    def update_multiplier(self, mean_ep_cost: float, current_progress_remaining: float) -> th.Tensor:
        """
        Update the lagrange multiplier (lambda) based on episode cost
        :param mean_ep_cost: mean episode cost
        :param current_progress_remaining:
        :return: penalty loss
        """
        self.multiplier_net.train(True)
        # Update optimizer learning rate
        update_learning_rate(self.optimizer, self.lr_schedule(current_progress_remaining))
        self.cost_threshold = self.cost_threshold_scheduler(current_progress_remaining)

        multiplier = self.multiplier()
        delta = mean_ep_cost - self.cost_threshold
        loss = - multiplier * delta

        # Optimization step
        self.optimizer.zero_grad()
        loss.backward()
        th.nn.utils.clip_grad_norm_(self.multiplier_net.parameters(), self.max_grad_norm)
        self.optimizer.step()
        return loss


class LagMultiplier(nn.Module):
    """
    Lagrange Multiplier Module

    This class implements the neural net to obtain the lagrange multiplier parameter

    :param init_value: initial value of the multiplier
    """

    def __init__(self, init_value: float = 0.0):
        super().__init__()
        init_value = max(0.0, init_value)
        init_value = np.log(max(np.exp(init_value) - 1, 1e-8))
        self.multiplier_parameter = nn.Parameter(th.tensor(init_value), requires_grad=True)
        self.multiplier_net = nn.Softplus()

    def forward(self) -> th.Tensor:
        return self.multiplier_net(self.multiplier_parameter)
