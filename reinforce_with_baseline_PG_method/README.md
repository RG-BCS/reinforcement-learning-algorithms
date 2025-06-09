# REINFORCE with Baseline Policy Gradient Method

This project implements the REINFORCE algorithm with a baseline (advantage function) for policy gradient reinforcement learning. The agent learns to balance the pole on the cart (CartPole-v1 environment) by maximizing expected cumulative reward.

---

## Overview

- **Environment:** CartPole-v1 from OpenAI Gym  
- **Goal:** Keep the pole balanced upright for as long as possible (max 500 steps)  
- **Approach:**  
  - Use a neural network policy to map environment observations to action probabilities  
  - Incorporate a baseline (average reward) to reduce variance in policy gradient updates  
  - Train using stochastic gradient ascent on the policy parameters  
- **Libraries:** TensorFlow, Keras, NumPy, Matplotlib, Gym

---

## Training Progress and Results
```bash

| Episode | Reward | Loss    |
|---------|--------|---------|
| 0       | 14.0   | 0.0052  |
| 50      | 75.0   | 0.0012  |
| 100     | 45.0   | 0.0157  |
| 150     | 111.0  | 0.0177  |
| 200     | 500.0  | 0.0069  |
| 250     | 277.0  | -0.0038 |
| 300     | 213.0  | -0.0144 |
| 350     | 500.0  | 0.0101  |
| 400     | 500.0  | -0.0125 |
| 450     | 232.0  | -0.0683 |
```

---

## Evaluation
``bash
| Metric                      | Value     |
|-----------------------------|-----------|
| Test before training reward | 9.0       |
| Test after training reward  | 500.0     |
| Average reward over 20 runs | 500.00    |
```
---

## *Notes*
```bash
- The baseline (average discounted reward) helps stabilize training by reducing variance in gradient estimates.
- Training typically converges faster and more reliably than vanilla REINFORCE without baseline.
- The stochastic policy selects actions probabilistically, enabling exploration of the environment.
```
---

## How to Run

```bash
python reinforce_with_baseline.py

or you can use

jupyter notebook demo.ipynb
```
