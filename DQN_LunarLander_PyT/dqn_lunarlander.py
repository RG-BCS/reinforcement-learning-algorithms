import gym
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from collections import deque
import random

from utils import set_seed, evaluate_agent, plot_rewards, save_model

# Set reproducibility
SEED = 42
set_seed(SEED)

# Q-network definition with two hidden layers of 128 units each
class QNetwork(nn.Module):
    def __init__(self, input_dim, output_dim,hidden_units=128):
        super().__init__()
        self.policy = nn.Sequential(
            nn.Linear(input_dim, hidden_units),
            nn.ReLU(),
            nn.Linear(hidden_units, output_dim)
        )

    def forward(self, x):
        return self.policy(x)


def train_dqn(env, q_net, target_net, optimizer, loss_fn,
              n_episodes=1000, gamma=0.99, epsilon_decay=0.995,
              epsilon_min=0.01, batch_size=128, buffer_size=10000,
              update_target_every=10):

    replay_buffer = deque(maxlen=buffer_size)
    epsilon = 1.0
    rewards_history = []

    for episode in range(n_episodes):
        # Handle different return formats of env.reset()
        result = env.reset()
        state = result[0] if isinstance(result, tuple) else result

        total_reward = 0
        done = False

        while not done:
            # Epsilon-greedy action selection
            if np.random.rand() < epsilon:
                action = env.action_space.sample()
            else:
                with torch.no_grad():
                    state_tensor = torch.FloatTensor(state).unsqueeze(0)
                    action = torch.argmax(q_net(state_tensor)).item()

            # Handle different return formats of env.step()
            result = env.step(action)
            if len(result) == 5:
                next_state, reward, terminated, truncated, _ = result
                done = terminated or truncated
            else:
                next_state, reward, done, _ = result

            # Store transition in replay buffer
            replay_buffer.append((state, action, reward, next_state, done))
            state = next_state
            total_reward += reward

            # Start training after the buffer has enough samples
            if len(replay_buffer) >= batch_size:
                batch = random.sample(replay_buffer, batch_size)
                states, actions, rewards_, next_states, dones = zip(*batch)

                # Convert batch to tensors
                states = torch.FloatTensor(np.array(states))
                actions = torch.LongTensor(actions).unsqueeze(1)
                rewards_ = torch.FloatTensor(rewards_).unsqueeze(1)
                next_states = torch.FloatTensor(np.array(next_states))
                dones = torch.FloatTensor(dones).unsqueeze(1)

                # Q-values for current states
                current_q = q_net(states).gather(1, actions)

                # Target Q-values using the target network
                with torch.no_grad():
                    max_next_q = target_net(next_states).max(1)[0].unsqueeze(1)
                    target_q = rewards_ + (1 - dones) * gamma * max_next_q

                # Compute loss and backpropagate
                loss = loss_fn(current_q, target_q)
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

        # Update epsilon
        epsilon = max(epsilon_min, epsilon * epsilon_decay)
        rewards_history.append(total_reward)

        # Periodically update the target network
        if episode % update_target_every == 0:
            target_net.load_state_dict(q_net.state_dict())

        if episode % 50 == 0:
            print(f"Episode {episode}: Total Reward = {total_reward:.2f}, Epsilon = {epsilon:.3f}")

    return rewards_history


if __name__ == '__main__':
    np.bool8 = np.bool_  # Fix deprecation warning in some gym versions

    env = gym.make("LunarLander-v2")
    result = env.reset()
    obs_dim = result[0].shape[0] if isinstance(result, tuple) else result.shape[0]
    action_dim = env.action_space.n

    # Initialize networks
    q_net = QNetwork(obs_dim, action_dim)
    target_net = QNetwork(obs_dim, action_dim)
    target_net.load_state_dict(q_net.state_dict())
    target_net.eval()

    # Optimizer and loss
    optimizer = optim.Adam(q_net.parameters(), lr=1e-3)
    loss_fn = nn.SmoothL1Loss()

    # Evaluate initial (untrained) performance
    print("Before training:")
    evaluate_agent(env, q_net, episodes=100)

    # Train the agent
    rewards = train_dqn(
        env, q_net, target_net, optimizer, loss_fn,
        n_episodes=1000, gamma=0.99, epsilon_decay=0.995,
        epsilon_min=0.01, batch_size=128, buffer_size=10000,
        update_target_every=10
    )

    # Evaluate performance after training
    print("\nAfter training:")
    evaluate_agent(env, q_net, episodes=100)

    # Plot and save results
    plot_rewards(rewards)
    save_model(q_net, "dqn_lunarlander.pth")
