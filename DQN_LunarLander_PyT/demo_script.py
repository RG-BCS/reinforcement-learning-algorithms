import torch
import gym
import numpy as np
from utils import QNetwork, evaluate_agent

def load_model(path, input_dim, output_dim):
    model = QNetwork(input_dim, output_dim)
    model.load_state_dict(torch.load(path))
    model.eval()
    return model

if __name__ == '__main__':
    np.bool8 = np.bool_

    env = gym.make("LunarLander-v2")
    result = env.reset()
    obs_dim = result[0].shape[0] if isinstance(result, tuple) else result.shape[0]
    action_dim = env.action_space.n

    model_path = "dqn_lunarlander.pth"  # Update if you're demoing Double DQN
    model = load_model(model_path, obs_dim, action_dim)

    print("Evaluating trained DQN model...")
    avg_reward = evaluate_agent(env, model, episodes=5, render=False)
    print(f"Average reward over 5 demo episodes: {avg_reward:.2f}")
