import torch
import numpy as np
import matplotlib.pyplot as plt


def set_seed(seed: int = 43):
    import random
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def evaluate_agent(env, model, episodes=10, max_steps=1000, render=False):
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


def plot_rewards(reward_list, title="Episode Rewards"):
    plt.figure(figsize=(10, 5))
    plt.plot(reward_list)
    plt.title(title)
    plt.xlabel("Episode")
    plt.ylabel("Total Reward")
    plt.grid(True)
    plt.tight_layout()
    plt.show()
