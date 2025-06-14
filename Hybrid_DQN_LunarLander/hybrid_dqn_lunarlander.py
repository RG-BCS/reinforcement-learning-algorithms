# hybrid_dqn.py

import numpy as np
import random
import torch
import torch.nn as nn
import torch.nn.functional as F


class DuelingNetwork(nn.Module):
    """
    Dueling Deep Q-Network (Dueling DQN)
    -------------------------------------
    Instead of a single Q-value stream, splits into:
    - A value stream estimating V(s)
    - An advantage stream estimating A(s, a)
    The Q-value is computed as: Q(s,a) = V(s) + (A(s,a) - mean(A(s,*)))
    This helps with better generalization and stability in environments with many similar-valued actions.
    """

    def __init__(self, input_dim, output_dim, hidden_units=128):
        super(DuelingNetwork, self).__init__()

        # Shared base layers
        self.shared = nn.Sequential(
            nn.Linear(input_dim, hidden_units),
            nn.ReLU()
        )

        # Value stream outputs a single value per state
        self.value_stream = nn.Linear(hidden_units, 1)

        # Advantage stream outputs one value per possible action
        self.advantage_stream = nn.Linear(hidden_units, output_dim)

    def forward(self, x):
        x = self.shared(x)

        # Separate into value and advantage
        value = self.value_stream(x)
        advantage = self.advantage_stream(x)

        # Subtract mean advantage to normalize across actions
        advantage = advantage - advantage.mean(dim=1, keepdim=True)

        # Combine value and adjusted advantage into final Q-values
        return value + advantage


class PrioritizedReplayBuffer:
    """
    Prioritized Experience Replay (PER)
    -------------------------------------
    Experiences are sampled based on their TD-error magnitude:
    - High error = more surprising = more important
    - Uses importance-sampling weights to reduce bias introduced by prioritized sampling
    """

    def __init__(self, capacity, alpha=0.6):
        self.capacity = capacity
        self.buffer = []
        self.priorities = []
        self.alpha = alpha  # 0: uniform sampling, 1: full prioritization
        self.pos = 0        # Position for cyclic replacement

    def add(self, transition, td_error=1.0):
        # Compute priority from TD-error
        priority = (abs(td_error) + 1e-5) ** self.alpha

        if len(self.buffer) < self.capacity:
            self.buffer.append(transition)
            self.priorities.append(priority)
        else:
            self.buffer[self.pos] = transition
            self.priorities[self.pos] = priority

        self.pos = (self.pos + 1) % self.capacity

    def sample(self, batch_size, beta=0.4):
        """
        Samples a batch with priority-based probabilities and returns IS weights
        """
        priorities = np.array(self.priorities)
        probs = priorities / priorities.sum()

        indices = np.random.choice(len(self.buffer), batch_size, p=probs)
        samples = [self.buffer[i] for i in indices]

        # Compute importance-sampling weights
        total = len(self.buffer)
        weights = (total * probs[indices]) ** (-beta)
        weights /= weights.max()

        return samples, indices, torch.from_numpy(weights).float().unsqueeze(1)

    def update_priorities(self, indices, td_errors):
        # Update priority values after training step
        for idx, td_err in zip(indices, td_errors):
            self.priorities[idx] = (abs(td_err.item()) + 1e-5) ** self.alpha

    def __len__(self):
        return len(self.buffer)


def train(env, Q_net, target_net, buffer, optimizer, loss_fn, config):
    """
    Training loop for a Dueling Double DQN agent with Prioritized Experience Replay
    -------------------------------------------------------------------------------
    Techniques used:
    - Dueling DQN: better Q-value estimates per state
    - Double DQN: avoids overestimation of future Q-values
    - PER: samples transitions based on TD-error priority
    """

    epsilon = 1.0
    rewards_history = []

    for episode in range(config["n_episodes"]):
        state, _ = env.reset()
        done = False
        total_reward = 0

        while not done:
            # Epsilon-greedy action selection
            if np.random.rand() < epsilon:
                action = env.action_space.sample()
            else:
                state_tensor = torch.FloatTensor(state).unsqueeze(0)
                with torch.no_grad():
                    q_vals = Q_net(state_tensor)
                    action = torch.argmax(q_vals, dim=1).item()

            # Interact with environment
            next_state, reward, terminated, truncated, _ = env.step(action)
            done = terminated or truncated
            buffer.add((state, action, reward, next_state, done))
            total_reward += reward
            state = next_state

            # Only learn if buffer has enough samples
            if len(buffer) >= config["batch_size"]:
                batch, indices, weights = buffer.sample(config["batch_size"], beta=config["beta"])
                states, actions, rewards_, next_states, dones = zip(*batch)

                states = torch.FloatTensor(np.array(states))
                actions = torch.LongTensor(actions).unsqueeze(1)
                rewards_ = torch.FloatTensor(rewards_).unsqueeze(1)
                next_states = torch.FloatTensor(np.array(next_states))
                dones = torch.FloatTensor(dones).unsqueeze(1)

                # Q(s,a) from current network
                q_values = Q_net(states).gather(1, actions)

                # Double DQN logic:
                # - Action is chosen using Q_net
                # - Value is evaluated using target_net
                with torch.no_grad():
                    next_actions = Q_net(next_states).argmax(1, keepdim=True)
                    next_q_values = target_net(next_states).gather(1, next_actions)
                    target_q = rewards_ + (1 - dones) * config["gamma"] * next_q_values

                # TD-error and weighted loss for PER
                td_errors = target_q - q_values
                loss = (weights * loss_fn(q_values, target_q)).mean()

                optimizer.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(Q_net.parameters(), 1.0)
                optimizer.step()

                buffer.update_priorities(indices, td_errors)

        # Epsilon decay
        epsilon = max(config["epsilon_min"], epsilon * config["epsilon_decay"])

        # Periodically sync target network
        if episode % config["target_update_freq"] == 0:
            target_net.load_state_dict(Q_net.state_dict())

        # Logging every 50 episodes
        if episode % 50 == 0:
            print(f"Episode {episode} | Total Reward: {total_reward:.2f} | Epsilon: {epsilon:.3f}")

        rewards_history.append(total_reward)

    return rewards_history
