import gymnasium as gym
from gymnasium import Env, spaces
from gymnasium.core import ActType, ObsType, WrapperObsType

from sb3_plus.gnn.spaces import GraphSample, GraphSpace


class GraphObservationWrapper(
    gym.ObservationWrapper[WrapperObsType, ActType, ObsType],
    gym.utils.RecordConstructorArgs,
):
    """
    Wrapper that transform observations from a ``gymnasium.spaces.Graph`` space into a format supported by Stable-Baselines3

    :param env: environment
    :param max_nodes: maximum number of nodes in a graph-based observation
    :param flatten: whether use flatten representations of graph data or not
    """

    observation_space: GraphSpace

    def __init__(self, env: Env, max_nodes: int, flatten: bool = True):
        gym.utils.RecordConstructorArgs.__init__(
            self, max_nodes=max_nodes, flatten=flatten
        )
        gym.ObservationWrapper.__init__(self, env)
        if isinstance(env.observation_space, GraphSpace):
            if env.observation_space.flatten == flatten:
                self.observation_space = env.observation_space
            else:
                self.observation_space = GraphSpace(
                    max_nodes=env.observation_space.max_nodes,
                    node_space=env.observation_space.node_space,
                    edge_space=env.observation_space.edge_space,
                    flatten=flatten,
                    seed=env.observation_space.np_random,
                )
        elif isinstance(env.observation_space, spaces.Graph):
            self.observation_space = GraphSpace.from_gym(
                env.observation_space, max_nodes=max_nodes, flatten=flatten
            )
        else:
            raise ValueError(
                f"{type(env.observation_space)} is an invalid observation space"
            )

    def observation(self, observation: ObsType) -> GraphSample:
        if isinstance(observation, spaces.GraphInstance):
            return self.observation_space.build_sample_from_gym(observation)
        elif isinstance(observation, dict):
            return observation
        else:
            raise ValueError(f"{type(observation)} is an invalid type of observation")
