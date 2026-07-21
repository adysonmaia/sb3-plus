import gymnasium as gym
import gymnasium_hybrid
from stable_baselines3.common.utils import set_random_seed
from stable_baselines3.common.vec_env import SubprocVecEnv

from sb3_plus import MultiOutputEnv, MultiOutputPPO


def make_env(env_id: str, rank: int, seed: int = 0):
    def _init():
        env = gym.make(env_id, render_mode="rgb_array")
        # Wrapping env to transform multi-output actions from flatten numpy.ndarray into dict or tuple
        env = MultiOutputEnv(env)
        env.reset(seed=seed + rank)
        return env

    set_random_seed(seed)
    return _init


def main():
    env_id = "Moving-v0"
    num_cpu = 4  # Number of processes to use
    # Create the vectorized environment
    vec_env = SubprocVecEnv([make_env(env_id, i) for i in range(num_cpu)])

    # Instantiate the agent
    model = MultiOutputPPO(
        policy="MultiOutputPolicy",
        env=vec_env,
        verbose=1,
        policy_kwargs=dict(net_arch=dict(pi=[252] * 4, vf=[252] * 4)),
    )

    # Train the agent and display a progress bar
    model.learn(total_timesteps=int(2e5), progress_bar=True)

    # Enjoy trained agent
    obs = vec_env.reset()
    for i in range(1000):
        action, _states = model.predict(obs, deterministic=True)
        obs, rewards, dones, info = vec_env.step(action)
        vec_env.render("human")
    vec_env.close()


if __name__ == "__main__":
    main()
