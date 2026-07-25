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
