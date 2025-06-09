# Stepwise Policy Gradient Reinforcement Learning (CartPole-v1)

This project implements a **Stepwise Policy Gradient (REINFORCE)** algorithm from scratch, applied to the classic `CartPole-v1` control problem using **TensorFlow** and **OpenAI Gym**.

It is designed as a clean, modular, and educational reinforcement learning portfolio project for ML/DL job applications.

---

## Problem Overview

In the **CartPole-v1** environment, a pole is attached to a cart that moves along a frictionless track. The objective is to prevent the pole from falling over by applying forces (left or right) to the cart.

- **State Space:** Continuous (position, velocity, angle, angular velocity)
- **Action Space:** Discrete (left or right)
- **Goal:** Keep the pole balanced upright for **as close to 500 steps as possible**

---

## Project Structure
```bash
reinforce_stepwise_pg/
    ├── stepwise_policy_gradient.py # Main algorithm class & training logic
    ├── demo.ipynb # Interactive demo & results (with markdown explanations)
    ├── demo_script.py # CLI script to run full training & evaluation
    ├── README.md # Project documentation
    ├── requirements.txt # Python dependencies
```

---

## Key Features
```bash

- Implements the REINFORCE algorithm with **discounted rewards**
- Uses a **custom neural network policy** trained via gradient estimation
- Includes both:
  - **Hardcoded baseline policy** (based on pole angle)
  - **Learned policy** (trained end-to-end)
- Includes both Jupyter notebook and CLI interface
- Fully **modular** and easy to adapt to other environments
```
---

## Results

```bash
The Stepwise Policy Gradient agent was trained on the CartPole-v1 environment for 150 iterations,
with performance evaluated every 10 iterations.

- **Before training**, the untrained neural network policy achieved an average reward
  of approximately **9.36** over 50 episodes, close to random.
- During training, the average reward steadily improved, surpassing **400** steps
  by iteration 80 and reaching near-perfect performance by the end.
- At the final iteration (iteration 149), the policy consistently achieved the
  **maximum possible reward of 500 steps**.
- **After training**, the policy averaged **499.42** over 50 evaluation episodes,
  demonstrating near-optimal control.

| Stage            | Average Reward |
|------------------|----------------|
| Before Training  | 9.36           |
| After Training   | 499.42         |
| Max Achieved     | 500.00         |

Training over 150 iterations yielded strong learning performance:


### Training Performance Snapshot

| Iteration | Average Reward (10 episodes) |
|-----------|------------------------------|
| 0         | 9.70                         |
| 20        | 64.70                        |
| 50        | 229.20                       |
| 80        | 414.10                       |
| 110       | 488.50                       |
| 140       | 473.30                       |
| 149       | 500.00                       |

This demonstrates effective learning and convergence of the policy gradient method on this classic control task.Moreover,
These results highlight the capability of the stepwise policy gradient algorithm to solve reinforcement learning problems effectively, making this portfolio piece a strong demonstration of practical RL expertise.
```

---
## Installation

```bash
git clone https://github.com/your-username/reinforce_stepwise_pg.git
cd reinforce_stepwise_pg
pip install -r requirements.txt
```

---

## How to Run

```bash
option 1. Jupyter Notebook (recommended)

    jupyter notebook demo.ipynb

Option 2. Command-Line Script

    python demo_script.py
```

