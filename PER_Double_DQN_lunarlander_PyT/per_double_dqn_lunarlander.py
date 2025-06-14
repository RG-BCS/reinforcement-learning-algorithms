import gym
import torch
import torch.nn as nn
import numpy as np
from utils import set_seed, evaluate_agent, plot_rewards

# Define Q-Network architecture
class QT_Network(nn.Module):
    def __init__(self, input_dim, output_dim):
        super().__init__()
        self.policy_model = nn.Sequential(
            nn.Linear(input_dim, 128),
            nn.ReLU(),
            nn.Linear(128, output_dim)
        )

    def forward(self, x):
        return self.policy_model(x)

# Prioritized Experience Replay Buffer
class PrioritizedReplayBuffer:
    def __init__(self, capacity, alpha=0.6):
        self.capacity = capacity
        self.buffer = []
        self.priorities = []
        self.alpha = alpha
        self.pos = 0  # Circular buffer pointer

    def add(self, transition, td_error=1.0):
        priority = (abs(td_error) + 1e-5) ** self.alpha
        if len(self.buffer) < self.capacity:
            self.buffer.append(transition)
            self.priorities.append(priority)
        else:
            self.buffer[self.pos] = transition
            self.priorities[self.pos] = priority
        self.pos = (self.pos + 1) % self.capacity

    def sample(self, batch_size, beta=0.4):
        priorities = np.array(self.priorities)
        probs = priorities / priorities.sum()
        indices = np.random.choice(len(self.buffer), batch_size, p=probs)
        samples = [self.buffer[i] for i in indices]

        total = len(self.buffer)
        weights = (total * probs[indices]) ** (-beta)
        weights /= weights.max()

        return samples, indices, torch.FloatTensor(weights).unsqueeze(1)

    def update_priorities(self, indices, td_errors):
        for idx, td_err in zip(indices, td_errors):
            self.priorities[idx] = (abs(td_err.item()) + 1e-5) ** self.alpha

    def __len__(self):
        return len(self.buffer)

# Training loop using Double DQN + Prioritized Experience Replay
def training_loop(env, Q_network, target_network, loss_fn, optimizer, discounted_factor,
                  n_episodes, replay_buffer, batch_size, epsilon_decay=0.995,
                  epsilon_min=0.01, beta=0.4):
    
    epsilon = 1.0
    rewards_history = []

    for episode in range(n_episodes):
        result = env.reset()
        state = result[0] if isinstance(result, tuple) else result  # Supports Gym v0.25 and v0.26+
        total_reward = 0
        done = False

        while not done:
            # Epsilon-greedy policy
            if np.random.rand() < epsilon:
                action = env.action_space.sample()
            else:
                with torch.no_grad():
                    q_vals = Q_network(torch.FloatTensor(state).unsqueeze(0))
                    action = torch.argmax(q_vals, dim=1).item()

            # Environment interaction
            result = env.step(action)
            if len(result) == 5:
                next_state, reward, terminated, truncated, _ = result
                done = terminated or truncated
            else:
                next_state, reward, done, _ = result

            replay_buffer.add((state, action, reward, next_state, done))
            state = next_state
            total_reward += reward

            if len(replay_buffer) >= batch_size:
                batch, indices, weights = replay_buffer.sample(batch_size, beta=beta)
                states, actions, rewards_, next_states, dones = zip(*batch)

                states = torch.FloatTensor(np.array(states))
                rewards_ = torch.FloatTensor(rewards_).unsqueeze(1)
                actions = torch.LongTensor(actions).unsqueeze(1)
                next_states = torch.FloatTensor(np.array(next_states))
                dones = torch.FloatTensor(dones).unsqueeze(1)

                q_values = Q_network(states).gather(1, actions)

                with torch.no_grad():
                    next_actions = Q_network(next_states).argmax(1, keepdim=True)
                    next_q_values = target_network(next_states).gather(1, next_actions)
                    target_q_values = rewards_ + (1 - dones) * discounted_factor * next_q_values

                td_errors = (target_q_values - q_values).detach()
                loss = (weights * loss_fn(q_values, target_q_values)).mean()

                optimizer.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(Q_network.parameters(), 1.0)
                optimizer.step()

                replay_buffer.update_priorities(indices, td_errors)

        epsilon = max(epsilon_min, epsilon * epsilon_decay)

        if episode % 10 == 0:
            target_network.load_state_dict(Q_network.state_dict())

        if episode % 50 == 0:
            print(f"Episode {episode}: Total Reward = {total_reward:.2f}, Epsilon = {epsilon:.3f}")

        rewards_history.append(total_reward)

    return rewards_history


if __name__ == '__main__':
    set_seed(43)

    env = gym.make("LunarLander-v2")
    result = env.reset()
    obs = result[0] if isinstance(result, tuple) else result
    input_dim = env.observation_space.shape[0]
    output_dim = env.action_space.n
    max_steps = env.spec.max_episode_steps

    # Hyperparameters
    replay_max = 10000
    learning_rate = 1e-3
    n_episodes = 1000
    epsilon_min = 0.01
    epsilon_decay = 0.995
    batch_size = 64
    discounted_factor = 0.99
    test_iters = 100
    beta = 0.4
    alpha = 0.6

    replay_buffer = PrioritizedReplayBuffer(capacity=replay_max, alpha=alpha)

    Q_network = QT_Network(input_dim, output_dim)
    target_network = QT_Network(input_dim, output_dim)
    target_network.load_state_dict(Q_network.state_dict())
    target_network.eval()

    loss_fn = nn.SmoothL1Loss()
    optimizer = torch.optim.Adam(Q_network.parameters(), lr=learning_rate)

    print("Before training model performance:")
    evaluate_agent(env, Q_network, episodes=test_iters, max_steps=max_steps)
    print()

    rewards = training_loop(
        env, Q_network, target_network, loss_fn, optimizer, discounted_factor,
        n_episodes, replay_buffer, batch_size,
        epsilon_decay=epsilon_decay, epsilon_min=epsilon_min, beta=beta
    )

    print("\nAfter training model performance:")
    evaluate_agent(env, Q_network, episodes=test_iters, max_steps=max_steps)
    plot_rewards(rewards)
