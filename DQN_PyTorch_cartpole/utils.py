import torch

def evaluate_episode(policy_model, env, max_steps=500):
    obs, _ = env.reset()
    total_reward = 0
    for _ in range(max_steps):
        with torch.no_grad():
            obs_tensor = torch.FloatTensor(obs).unsqueeze(0)
            action = torch.argmax(policy_model(obs_tensor)).item()
        obs, reward, terminated, truncated, _ = env.step(action)
        total_reward += reward
        if terminated or truncated:
            break
    return total_reward

def evaluate_policy(policy_model, env, episodes=10, max_steps=500):
    total_rewards = []
    for _ in range(episodes):
        reward = evaluate_episode(policy_model, env, max_steps)
        total_rewards.append(reward)
    avg_reward = sum(total_rewards) / episodes
    return avg_reward
