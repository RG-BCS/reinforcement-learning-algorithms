import torch
import torch.nn as nn
import torch.nn.functional as F
import random
import numpy as np
from collections import deque

# Q-network definition
class QNetwork(nn.Module):
    def __init__(self, state_size, action_size):
        super(QNetwork, self).__init__()
        self.fc1 = nn.Linear(state_size, 64)
        self.fc2 = nn.Linear(64, 64)
        self.out = nn.Linear(64, action_size)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        return self.out(x)

# Experience Replay Buffer
class ReplayBuffer:
    def __init__(self, capacity=10000):
        self.buffer = deque(maxlen=capacity)

    def push(self, state, action, reward, next_state, done):
        self.buffer.append((state, action, reward, next_state, done))

    def sample(self, batch_size):
        batch = random.sample(self.buffer, batch_size)
        states, actions, rewards, next_states, dones = zip(*batch)
        return (
            np.array(states),
            actions,
            rewards,
            np.array(next_states),
            dones
        )

    def __len__(self):
        return len(self.buffer)

# Training function
def train_dqn(
    online_network,
    target_network,
    env,
    buffer,
    loss_fn,
    optimizer,
    num_episodes=500,
    batch_size=64,
    gamma=0.99,
    epsilon=1.0,
    epsilon_min=0.01,
    epsilon_decay=0.995,
    target_update_freq=10
):
    for episode in range(num_episodes):
        state, _ = env.reset()
        total_reward = 0
        done = False

        while not done:
            # Epsilon-greedy action selection
            if random.random() < epsilon:
                action = env.action_space.sample()
            else:
                with torch.no_grad():
                    state_tensor = torch.FloatTensor(state).unsqueeze(0)
                    q_values = online_network(state_tensor)
                    action = q_values.argmax().item()

            # Step the environment
            next_state, reward, terminated, truncated, _ = env.step(action)
            done = terminated or truncated

            # Store experience
            buffer.push(state, action, reward, next_state, done)
            state = next_state
            total_reward += reward

            # Train the network if buffer is ready
            if len(buffer) >= batch_size:
                states, actions, rewards, next_states, dones = buffer.sample(batch_size)

                states = torch.FloatTensor(states)
                actions = torch.LongTensor(actions).unsqueeze(1)
                rewards = torch.FloatTensor(rewards).unsqueeze(1)
                next_states = torch.FloatTensor(next_states)
                dones = torch.FloatTensor(dones).unsqueeze(1)

                # Q(s,a)
                current_q = online_network(states).gather(1, actions)

                # max_a' Q_target(s', a')
                with torch.no_grad():
                    max_next_q = target_network(next_states).max(1)[0].unsqueeze(1)
                    target_q = rewards + (gamma * max_next_q * (1 - dones))

                loss = loss_fn(current_q, target_q)

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

        # Update target network
        if episode % target_update_freq == 0:
            target_network.load_state_dict(online_network.state_dict())

        # Decay epsilon
        epsilon = max(epsilon_min, epsilon * epsilon_decay)

        # Logging
        if episode % 50 == 0:
            print(f"Episode {episode}: Total Reward = {total_reward:.1f}, Epsilon = {epsilon:.3f}")
