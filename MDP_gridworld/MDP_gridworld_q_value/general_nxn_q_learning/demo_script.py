"""
demo_script.py

A demonstration script for Q-learning on an NxN grid world (no obstacles).

Usage:
    python demo_script.py
"""

import numpy as np
from q_learning_nxn import QLearningNxN


def main():
    # Configuration
    grid_size = (5, 5)
    gamma = 0.98
    n_iterations = 50

    print(f"Starting Q-learning on a {grid_size[0]}x{grid_size[1]} GridWorld...")

    # Initialize Q-learning agent
    agent = QLearningNxN(grid_size=grid_size, gamma=gamma)

    # Train the agent
    agent.train(n_iterations=n_iterations)
    print(f"Training completed after {n_iterations} iterations.\n")

    # Show the optimal policy
    print("Learned Optimal Policy:")
    agent.print_policy()


if __name__ == "__main__":
    main()
