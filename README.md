# Reinforcement Learning Algorithms

This repository contains a curated collection of fundamental and deep reinforcement learning algorithms implemented using **Python**, **NumPy**, and industry-standard frameworks like **TensorFlow** and **PyTorch**.

The goal is to deeply understand the theory and practice of RL — from **tabular methods** to **deep function approximation** — by building everything from scratch, with minimal reliance on high-level RL libraries.

---

## What's Included

A wide spectrum of RL methods — each implemented clearly, modularly, and with educational intent.

### Tabular Methods & Planning
- Monte Carlo Prediction & Control
- TD(0), SARSA, Q-Learning
- MDP Value Iteration & Policy Iteration
- GridWorld (with and without obstacles)

###  Policy Gradient Family
- REINFORCE (Vanilla Policy Gradient)
- Actor-Critic
- PG with Advantage Estimation
- Normalization + Baseline Techniques

### Deep Q-Learning (DQN)
- Basic DQN with function approximation
- Target networks & experience replay
- DQN variants (Double DQN, Dueling)

### Actor-Critic with Deep Networks
- Implemented in TensorFlow and/or PyTorch
- Applied to CartPole-v1, LunarLander-v2

---

## Environments Used

- GridWorld (custom envs with and without obstacles)
- CartPole-v1 (OpenAI Gym)
- LunarLander-v2 (Box2D physics)

---

## Each folder contains:

- Main training scripts
- Modular utilities
- Demos & notebooks
- Readable code with comments
- Evaluation outputs

---

## Why This Repo Matters

- Shows mastery of RL theory **and** practical application  
- Combines low-level math with real-world Gym environments  
- Uses both **TensorFlow** and **PyTorch** to demonstrate framework fluency  

---

## Sample Results

From the Actor-Critic CartPole implementation:

| Episode | Reward | Actor Loss | Critic Loss |
|--------:|-------:|-----------:|------------:|
| 0       | 64.0   | -0.0015    | 1.4060      |
| 250     | 203.0  | -0.0097    | 0.6789      |
| 400     | 500.0  | -0.0273    | 0.6555      |
| 450     | 500.0  |  0.0019    | 0.8129      |

- **Initial test avg reward**: 22.62  
- **Final test avg reward**: 487.99 over 100 episodes 

---

## Future Additions

- PPO and DDPG (continuous action)
- Custom reward shaping experiments
- Frame stacking & convolutional policies
- RL blog series based on these implementations

---
