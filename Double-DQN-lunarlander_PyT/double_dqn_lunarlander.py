# double_dqn_lunarlander.py

# === Dependencies ===
# !pip install gym==0.26.2
# !pip install swig
# !pip install box2d box2d-kengz

import gym
import torch
import torch.nn as nn
import numpy as np
import random
from collections import deque
import matplotlib.pyplot as plt

# Set seed for reproducibility
SEED = 43
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)

# === Neural Network ===
class QT_Network(nn.Module):
    """
    Simple feedforward neural network with one hidden layer for estimating Q-values.
    """
    def __init__(self, input_dim, output_dim):
        super().__init__()
        self.policy_model = nn.Sequential(
            nn.Linear(input_dim, 128),
            nn.ReLU(),
            nn.Linear(128, output_dim)
        )

    def forward(self, x):
        return self.policy_model(x)

# === Training Loop ===
def training_loop(env, Q_network, target_network, loss_fn, optimizer, discounted_factor, n_episodes,
                  epsilon_decay=0.995, epsilon_min=0.01):
    epsilon = 1.0
    rewards_history = []

    for episode in range(n_episodes):
        result = env.reset()
        state, _ = result if isinstance(result, tuple) else (result, {})

        total_reward = 0
        done = False

        while not done:
            # ε-greedy action selection
            if np.random.rand() < epsilon:
                action = env.action_space.sample()
            else:
                state_tensor = torch.FloatTensor(state).unsqueeze(0)
                with torch.no_grad():
                    q_vals = Q_network(state_tensor)
                    action = torch.argmax(q_vals, dim=1).item()

            # Environment interaction
            result = env.step(action)
            if len(result) == 5:
                next_state, reward, terminated, truncated, _ = result
                done = terminated or truncated
            else:
                next_state, reward, done, _ = result

            # Store transition in replay buffer
            replay_buffer.append((state, action, reward, next_state, done))
            total_reward += reward
            state = next_state

            # Train only if we have enough samples
            if len(replay_buffer) >= batch_size:
                batch = random.sample(replay_buffer, batch_size)
                states, actions, rewards, next_states, dones = zip(*batch)

                states = torch.FloatTensor(np.array(states))
                actions = torch.LongTensor(actions).unsqueeze(1)
                rewards = torch.FloatTensor(rewards).unsqueeze(1)
                next_states = torch.FloatTensor(np.array(next_states))
                dones = torch.FloatTensor(dones).unsqueeze(1)

                # Double DQN logic
                q_values = Q_network(states).gather(1, actions)
                with torch.no_grad():
                    # Use the online network to select the best action
                    next_actions = Q_network(next_states).argmax(1, keepdim=True)
                    # Evaluate it using the target network
                    next_q_values = target_network(next_states).gather(1, next_actions)
                    target_q_values = rewards + (1 - dones) * discounted_factor * next_q_values

                loss = loss_fn(q_values, target_q_values)
                optimizer.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(Q_network.parameters(), 1.0)
                optimizer.step()

        # Decay ε
        epsilon = max(epsilon_min, epsilon * epsilon_decay)

        # Periodically update target network
        if episode % 10 == 0:
            target_network.load_state_dict(Q_network.state_dict())

        if episode % 50 == 0:
            print(f"Episode {episode}: Total Reward = {total_reward:.2f}, Epsilon = {epsilon:.3f}")

        rewards_history.append(total_reward)

    return rewards_history

# === Evaluation ===
def evaluate_agent(env, model, episodes=10, max_steps=1000, render=False):
    model.eval()
    total_rewards = []

    for episode in range(episodes):
        result = env.reset()
        state, _ = result if isinstance(result, tuple) else (result, {})
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

# === Main Execution ===
if __name__ == '__main__':
    np.bool8 = np.bool_  # Compatibility fix

    env = gym.make("LunarLander-v2")
    obs, _ = env.reset()
    input_dim = env.observation_space.shape[0]
    output_dim = env.action_space.n
    max_steps = env.spec.max_episode_steps

    # Hyperparameters
    replay_max = 10000
    learning_rate = 1e-3
    n_episodes = 1000
    epsilon_min, epsilon_decay = 0.01, 0.995
    batch_size = 64
    discounted_factor = 0.99
    test_iters = 100

    replay_buffer = deque(maxlen=replay_max)

    Q_network = QT_Network(input_dim, output_dim)
    target_network = QT_Network(input_dim, output_dim)
    target_network.load_state_dict(Q_network.state_dict())
    target_network.eval()

    loss_fn = nn.SmoothL1Loss()
    optimizer = torch.optim.Adam(Q_network.parameters(), lr=learning_rate)

    print(f"Before training model performance")
    evaluate_agent(env, Q_network, episodes=test_iters, max_steps=max_steps, render=False)
    print()

    rewards = training_loop(env, Q_network, target_network, loss_fn, optimizer,
                            discounted_factor, n_episodes, epsilon_decay, epsilon_min)

    print(f"\nAfter training model performance ")
    evaluate_agent(env, Q_network, episodes=test_iters, max_steps=max_steps, render=False)

    plt.plot(rewards)
    plt.title("Episode Rewards (Double DQN)")
    plt.xlabel("Episode")
    plt.ylabel("Total Reward")
    plt.grid(True)
    plt.show()
