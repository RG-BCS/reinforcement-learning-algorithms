# demo_script/train_agent.py

import gym
import numpy as np
import random
import torch
import torch.nn as nn
import matplotlib.pyplot as plt
from collections import deque

from hybrid_dqn import DuelingNetwork, PrioritizedReplayBuffer, train
from utils.evaluation import evaluate_agent

# ----------------------------
# Set seeds for reproducibility
# ----------------------------
SEED = 43
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)

# ----------------------------
# Initialize environment
# ----------------------------
env = gym.make("LunarLander-v2")
env.reset(seed=SEED)
obs_dim = env.observation_space.shape[0]
action_dim = env.action_space.n
max_steps = env.spec.max_episode_steps

# ----------------------------
# Hyperparameters & config
# ----------------------------
config = {
    "n_episodes": 1000,
    "batch_size": 64,
    "gamma": 0.99,
    "epsilon_decay": 0.995,
    "epsilon_min": 0.01,
    "target_update_freq": 10,
    "beta": 0.4,
    "alpha": 0.6,
    "replay_capacity": 10000,
    "lr": 1e-3,
}

# ----------------------------
# Initialize networks and optimizer
# ----------------------------
Q_network = DuelingNetwork(obs_dim, action_dim)
target_network = DuelingNetwork(obs_dim, action_dim)
target_network.load_state_dict(Q_network.state_dict())
target_network.eval()

optimizer = torch.optim.Adam(Q_network.parameters(), lr=config["lr"])
loss_fn = nn.SmoothL1Loss()

# ----------------------------
# Initialize prioritized replay buffer
# ----------------------------
replay_buffer = PrioritizedReplayBuffer(capacity=config["replay_capacity"], alpha=config["alpha"])

# ----------------------------
# Evaluate before training
# ----------------------------
print("Before training:")
evaluate_agent(env, Q_network, episodes=10, max_steps=max_steps)

# ----------------------------
# Train the agent
# ----------------------------
reward_history = train(env, Q_network, target_network, replay_buffer, optimizer, loss_fn, config)

# ----------------------------
# Evaluate after training
# ----------------------------
print("\nAfter training:")
evaluate_agent(env, Q_network, episodes=10, max_steps=max_steps)

# ----------------------------
# Plot reward curve
# ----------------------------
plt.plot(reward_history)
plt.title("Episode Rewards")
plt.xlabel("Episode")
plt.ylabel("Total Reward")
plt.grid(True)
plt.show()
