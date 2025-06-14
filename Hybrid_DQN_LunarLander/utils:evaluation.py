# utils/evaluation.py

import torch
import numpy as np

def evaluate_agent(env, model, episodes=10, max_steps=1000, render=False):
    """
    Evaluates a trained RL agent.

    Args:
        env: Gym environment instance
        model: Trained PyTorch model (DuelingNetwork)
        episodes (int): Number of episodes to run evaluation
        max_steps (int): Max steps per episode
        render (bool): Whether to render environment during evaluation

    Returns:
        avg_reward (float): Average total reward over episodes
    """
    model.eval()
    total_rewards = []

    for episode in range(episodes):
        state, _ = env.reset()
        episode_reward = 0

        for _ in range(max_steps):
            if render:
                env.render()

            state_tensor = torch.FloatTensor(state).unsqueeze(0)
            with torch.no_grad():
                action = torch.argmax(model(state_tensor)).item()

            next_state, reward, terminated, truncated, _ = env.step(action)
            done = terminated or truncated
            episode_reward += reward
            state = next_state
            if done:
                break
        
        total_rewards.append(episode_reward)

    avg_reward = np.mean(total_rewards)
    print(f"Average reward over {episodes} episodes: {avg_reward:.2f}")
    return avg_reward
