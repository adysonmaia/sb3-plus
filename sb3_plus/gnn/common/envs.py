from typing import Any, SupportsFloat

import gymnasium as gym
import numpy as np
from gymnasium import spaces


class DummyGraphEnv(gym.Env):
    """
    Dummy environment with graph-based states used for tests

    :param min_nodes: min number of nodes in the graph
    :param max_nodes: max number of nodes in the graph. If `None`, graph always have `min_nodes` nodes
    """

    observation_space: spaces.Graph
    metadata = {"render_modes": ["human"]}

    def __init__(self, min_nodes: int = 5, max_nodes: int | None = None):
        super().__init__()
        self.render_mode = "human"
        self.min_nodes = min_nodes
        self.max_nodes = (
            max(max_nodes, min_nodes) if max_nodes is not None else min_nodes
        )

        # Each node has:
        #   feature[0] = random value
        #   feature[1] = activated flag
        #   feature[2] = normalized node index
        self.node_space = spaces.Box(
            low=0.0,
            high=1.0,
            shape=(3,),
            dtype=np.float32,
        )
        self.edge_space = spaces.Box(
            low=0.0,
            high=1.0,
            shape=(2,),
            dtype=np.float32,
        )
        self.observation_space = spaces.Graph(
            node_space=self.node_space,
            edge_space=self.edge_space,
        )

        self.action_space = spaces.Discrete(self.max_nodes)
        self.num_nodes = 0
        self.step_count = 0

    def _obs(self) -> spaces.GraphInstance:
        return self.observation_space.sample(num_nodes=self.num_nodes)

    def reset(
        self, *, seed: int | None = None, options: dict[str, Any] | None = None
    ) -> tuple[spaces.GraphInstance, dict[str, Any]]:
        super().reset(seed=seed, options=options)
        self.step_count = 0
        self.num_nodes = int(
            self.np_random.integers(self.min_nodes, self.max_nodes + 1)
        )
        return self._obs(), {}

    def step(
        self, action: int
    ) -> tuple[spaces.GraphInstance, SupportsFloat, bool, bool, dict[str, Any]]:
        reward = 0.0
        terminated = False
        truncated = self.step_count > 2 * self.num_nodes
        self.step_count += 1
        return self._obs(), reward, terminated, truncated, {}

    def render(self) -> None:
        pass


class SimpleGraphEnv(gym.Env):
    """
    Simple graph-based RL environment using gymnasium.spaces.Graph.

    Observation:
        Graph(
            node_space = Box(shape=(3,), dtype=float32),
            edge_space = Box(shape=(1,), dtype=float32)
        )

    Action:
        Discrete(num_nodes)
            Select a node to activate.

    :param min_nodes: min number of nodes in the graph
    :param max_nodes: max number of nodes in the graph. If `None`, graph always have `min_nodes` nodes
    """

    metadata = {"render_modes": ["human"]}

    def __init__(self, min_nodes: int = 5, max_nodes: int | None = None):
        super().__init__()
        self.render_mode = "human"
        self.min_nodes = min_nodes
        self.max_nodes = (
            max(max_nodes, min_nodes) if max_nodes is not None else min_nodes
        )

        # Each node has:
        #   feature[0] = random value
        #   feature[1] = activated flag
        #   feature[2] = normalized node index
        self.node_space = spaces.Box(
            low=0.0,
            high=1.0,
            shape=(3,),
            dtype=np.float32,
        )
        # Each edge stores a single weight
        self.edge_space = spaces.Box(
            low=0.0,
            high=1.0,
            shape=(1,),
            dtype=np.float32,
        )
        self.observation_space = spaces.Graph(
            node_space=self.node_space,
            edge_space=self.edge_space,
        )

        self.action_space = spaces.Discrete(self.max_nodes)
        self.step_count = 0
        self.num_nodes = 0
        self.reset()

    def _build_observation(self) -> spaces.GraphInstance:
        return spaces.GraphInstance(
            nodes=self.nodes,
            edges=self.edges,
            edge_links=self.edge_links,
        )

    def reset(
        self, *, seed: int | None = None, options: dict[str, Any] | None = None
    ) -> tuple[spaces.GraphInstance, dict[str, Any]]:
        super().reset(seed=seed, options=options)

        rng = self.np_random

        # ------------------------
        # Node features
        # ------------------------
        self.num_nodes = int(rng.integers(self.min_nodes, self.max_nodes + 1))
        self.nodes = np.zeros((self.num_nodes, 3), dtype=np.float32)

        self.nodes[:, 0] = rng.random(self.num_nodes)
        self.nodes[:, 1] = 0.0
        self.nodes[:, 2] = np.arange(self.num_nodes) / max(1, self.num_nodes - 1)

        # ------------------------
        # Chain graph
        # 0--1--2--3--...
        # ------------------------
        edge_links = []
        edge_features = []

        for i in range(self.num_nodes - 1):
            edge_links.append([i, i + 1])
            edge_links.append([i + 1, i])

            w = rng.random()

            edge_features.append([w])
            edge_features.append([w])

        self.edge_links = np.asarray(edge_links, dtype=np.int64)
        self.edges = np.asarray(edge_features, dtype=np.float32)
        self.step_count = 0

        return self._build_observation(), {}

    def step(
        self, action: int
    ) -> tuple[spaces.GraphInstance, SupportsFloat, bool, bool, dict[str, Any]]:
        reward = -0.5

        if action < self.num_nodes and self.nodes[action, 1] == 0:
            self.nodes[action, 1] = 1.0
            reward = 1.0

        terminated = bool(np.all(self.nodes[:, 1] == 1))
        truncated = self.step_count > 3 * self.num_nodes
        self.step_count += 1

        return (
            self._build_observation(),
            reward,
            terminated,
            truncated,
            {},
        )

    def render(self):
        print("Node features:")
        print(self.nodes)
        print("\nEdges:")
        for (u, v), w in zip(self.edge_links, self.edges):
            print(f"{u} -> {v}  weight={w[0]:.3f}")
