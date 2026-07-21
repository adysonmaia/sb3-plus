import gymnasium_hybrid
from stable_baselines3.common.vec_env import SubprocVecEnv

from sb3_plus import MultiOutputPPO, make_multioutput_env


def main():
    # Create vectorized environment
    vec_env = make_multioutput_env("Moving-v0", n_envs=4, vec_env_cls=SubprocVecEnv)

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
