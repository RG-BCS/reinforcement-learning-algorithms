import gym
import torch
import torch.nn as nn
import torch.optim as optim

from dqn_agent import QNetwork, ReplayBuffer, train_dqn
from utils import evaluate_policy

def main():
    # Environment
    env = gym.make("CartPole-v1")
    state_size = env.observation_space.shape[0]
    action_size = env.action_space.n

    # Hyperparameters
    num_episodes = 800
    batch_size = 64
    gamma = 0.99
    epsilon = 1.0
    epsilon_min = 0.01
    epsilon_decay = 0.995
    lr = 1e-2
    target_update_freq = 10
    test_iters = 100

    # Networks
    q_network = QNetwork(state_size, action_size)
    target_network = QNetwork(state_size, action_size)
    target_network.load_state_dict(q_network.state_dict())
    target_network.eval()

    # Optimizer and Loss
    optimizer = optim.Adam(q_network.parameters(), lr=lr)
    loss_fn = nn.SmoothL1Loss()

    # Replay Buffer
    buffer = ReplayBuffer(capacity=10000)

    # Evaluate before training
    print(f"\nPolicy model performance before training ({test_iters} episodes): "
          f"{evaluate_policy(q_network, env, test_iters):.3f}\n")

    # Train agent
    train_dqn(
        q_network, target_network, env, buffer, loss_fn, optimizer,
        num_episodes=num_episodes,
        batch_size=batch_size,
        gamma=gamma,
        epsilon=epsilon,
        epsilon_min=epsilon_min,
        epsilon_decay=epsilon_decay,
        target_update_freq=target_update_freq
    )

    # Evaluate after training
    print(f"\nPolicy model performance after training ({test_iters} episodes): "
          f"{evaluate_policy(q_network, env, test_iters):.3f}\n")

    # Save model
    torch.save(q_network.state_dict(), "models/dqn_cartpole.pth")
    print("Model saved to models/dqn_cartpole.pth")

if __name__ == "__main__":
    main()
