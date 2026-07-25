import torch
from stable_baselines3 import PPO
from stable_baselines3.common.evaluation import evaluate_policy
from torch_geometric.nn.models import GAT

from sb3_plus.gnn import ActorCriticGnnPolicy, GraphObservationWrapper
from sb3_plus.gnn.common.envs import SimpleGraphEnv


def main():
    # Create the environment
    min_nodes = 5
    max_nodes = 8
    env = SimpleGraphEnv(min_nodes, max_nodes)
    env = GraphObservationWrapper(env, max_nodes=max_nodes)

    # Instantiate the agent
    policy_kwargs = dict(
        gnn_class=GAT,
        gnn_kwargs=dict(
            hidden_channels=16,
            num_layers=1,
            out_channels=16,
            residual=True,
            v2=True,
        ),
        gnn_pool_fn="mean",
        readout_net_arch=dict(pi=[32] * 2, vf=[32] * 2),
        readout_act_fn=torch.nn.ReLU,
        share_features_extractor=False,
    )
    model = PPO(
        policy=ActorCriticGnnPolicy,
        env=env,
        verbose=1,
        policy_kwargs=policy_kwargs,
    )

    # Train the agent and display a progress bar
    model.learn(total_timesteps=int(5e5), progress_bar=True)

    # Save the agent
    model.save("ppo_gnn")
    del model  # delete trained model to demonstrate loading

    # Load the trained agent
    model = PPO.load("ppo_gnn", env=env)

    # Evaluate the agent
    mean_reward, std_reward = evaluate_policy(
        model, model.get_env(), n_eval_episodes=10
    )
    print(f"Mean Reward {mean_reward} | Std Reward {std_reward}")

    # Enjoy trained agent
    vec_env = model.get_env()
    obs = vec_env.reset()
    for i in range(10):
        action, _states = model.predict(obs, deterministic=True)
        obs, rewards, dones, info = vec_env.step(action)
        print(f"step: {i}, rewards: {rewards}, dones: {dones}")


if __name__ == "__main__":
    main()
