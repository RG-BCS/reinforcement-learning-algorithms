import random
import numpy as np
import torch
import matplotlib.pyplot as plt


def set_seed(seed: int = 42):
    """Set random seeds for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)


def plot_rewards(rewards, save_path=None):
    """Plot episode rewards and optionally save to disk."""
    plt.figure(figsize=(10, 5))
    plt.plot(rewards, label="Episode Reward")
    plt.xlabel("Episode")
    plt.ylabel("Total Reward")
    plt.title("Training Progress")
    plt.grid(True)
    plt.legend()
    if save_path:
        plt.savefig(save_path)
        print(f"Reward plot saved to {save_path}")
    else:
        plt.show()


def save_model(model, path):
    """Save model state_dict to file."""
    torch.save(model.state_dict(), path)
    print(f"Model saved to {path}")


def load_model(model, path):
    """Load model weights from file into provided model."""
    model.load_state_dict(torch.load(path))
    model.eval()
    print(f"Model loaded from {path}")
    return model


def evaluate_agent(env, model, episodes=10, max_steps=1000, render=False):
    """Evaluate a trained DQN agent over multiple episodes."""
    model.eval()
    total_rewards = []

    for episode in range(episodes):
        result = env.reset()
        state = result[0] if isinstance(result, tuple) else result

        episode_reward = 0
        for _ in range(max_steps):
            if render:
                env.render()

            state_tensor = torch.FloatTensor(state).unsqueeze(0)
            with torch.no_grad():
                action = torch.argmax(model(state_tensor)).item()

            result = env.step(action)
            if len(result) == 5:
                next_state, reward, terminated, truncated, _ = result
                done = terminated or truncated
            else:
                next_state, reward, done, _ = result

            episode_reward += reward
            state = next_state
            if done:
                break

        total_rewards.append(episode_reward)

    avg_reward = np.mean(total_rewards)
    print(f"Average reward over {episodes} episodes: {avg_reward:.2f}")
    return avg_reward
