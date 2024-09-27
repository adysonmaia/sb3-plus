from stable_baselines3.common.monitor import Monitor

import time
from typing import Any, Dict, List, Optional, SupportsFloat, Tuple, Union

import gymnasium as gym
from gymnasium.core import ActType, ObsType
from .type_aliases import PENALTY_COST_INFO_KEY


class SafeMonitor(Monitor):
    """
    A monitor wrapper for Gym environments, it is used to know the episode reward, length, cost, time and other data.

    """

    def __init__(
            self,
            env: gym.Env,
            filename: Optional[str] = None,
            allow_early_resets: bool = True,
            reset_keywords: Tuple[str, ...] = (),
            info_keywords: Tuple[str, ...] = (),
            override_existing: bool = True,
    ):
        info_keywords = tuple(["c"]) + info_keywords
        super().__init__(
            env=env,
            filename=filename,
            allow_early_resets=allow_early_resets,
            reset_keywords=reset_keywords,
            info_keywords=info_keywords,
            override_existing=override_existing
        )
        self.costs: List[float] = []
        self.episode_costs: List[float] = []

    def reset(self, **kwargs) -> Tuple[ObsType, Dict[str, Any]]:
        result = super().reset(**kwargs)
        self.costs = []
        return result

    def step(self, action: ActType) -> Tuple[ObsType, SupportsFloat, bool, bool, Dict[str, Any]]:
        """
        Step the environment with the given action

        :param action: the action
        :return: observation, reward, terminated, truncated, information
        """
        if self.needs_reset:
            raise RuntimeError("Tried to step environment that needs reset")
        observation, reward, terminated, truncated, info = self.env.step(action)
        cost = info.get(PENALTY_COST_INFO_KEY, 0.0)
        self.rewards.append(float(reward))
        self.costs.append(float(cost))
        if terminated or truncated:
            self.needs_reset = True
            ep_rew = sum(self.rewards)
            ep_cost = sum(self.costs)
            ep_len = len(self.rewards)
            ep_info = {
                "r": round(ep_rew, 6),
                "l": ep_len,
                "t": round(time.time() - self.t_start, 6),
                "c": round(ep_cost, 6)
            }
            for key in self.info_keywords:
                ep_info[key] = info[key]
            self.episode_returns.append(ep_rew)
            self.episode_lengths.append(ep_len)
            self.episode_times.append(time.time() - self.t_start)
            ep_info.update(self.current_reset_info)
            if self.results_writer:
                self.results_writer.write_row(ep_info)
            info["episode"] = ep_info
        self.total_steps += 1
        return observation, reward, terminated, truncated, info

    def get_episode_cost(self) -> List[float]:
        """
        Returns the cost of all the episodes

        :return:
        """
        return self.episode_costs
