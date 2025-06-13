#  Double DQN on LunarLander-v2
```bash

This repository implements the **Double Deep Q-Network (Double DQN)** algorithm using PyTorch on the classic `LunarLander-v2` environment from OpenAI Gym. It is an improvement over the vanilla DQN, mitigating Q-value overestimation by decoupling action selection and action evaluation.

> This repo is separate from the [Vanilla DQN implementation](https://github.com/yourusername/vanilla-dqn-lunarlander) to highlight the difference in architecture, training stability, and results.
```
---

##  Repository Structure
```bash
    ├── double_dqn_lunarlander.py       # Main vanilla DQN training script
    ├── demo_script.py                  # Quick demo script to test the trained agent
    ├── demo.ipynb                      # Interactive Jupyter notebook walkthrough
    ├── requirements.txt                # Python dependencies
    ├── README.md                       # This file

```
---

## Getting Started

### 1. Install dependencies
```bash
    pip install -r requirements.txt

### 2. Train the agent
    python q_learning_tf.py

    . Use demo_script.py to test a trained agent quickly.
    . Explore demo.ipynb for an interactive walkthrough and analysis.

```
---

## Model Architecture
```bash

Dense(8 → 128) + relu
   ↓
Dense(128 → action_dim = 4)

. Optimizer: Adam (lr = 1e-3)
. Loss: nn.SmoothL1Loss()/hubber
. γ (discount factor): 0.99
. Epsilon decay: epsilon = max(epsilon_min, epsilon * epsilon_decay)

```
---
## Training Results
```bash
### Before Training
        . Average reward over 100 evaluation episodes: -372.75

### Training Progress (Sample Episodes)
        
        | Episode | Total Reward | Epsilon |
        | ------- | ------------ | ------- |
        | 0       | -240.69      | 0.995   |
        | 50      | -66.27       | 0.774   |
        | 100     | -98.99       | 0.603   |
        | 150     | 88.69        | 0.469   |
        | 200     | -1.93        | 0.365   |
        | 250     | -126.02      | 0.284   |
        | 300     | -129.92      | 0.221   |
        | 350     | -188.69      | 0.172   |
        | 400     | 84.58        | 0.134   |
        | 450     | 43.25        | 0.104   |
        | 500     | -158.02      | 0.081   |
        | 550     | 253.35       | 0.063   |
        | 600     | 41.28        | 0.049   |
        | 650     | 251.60       | 0.038   |
        | 700     | 191.28       | 0.030   |
        | 750     | 140.30       | 0.023   |
        | 800     | 251.20       | 0.018   |
        | 850     | 272.62       | 0.014   |
        | 900     | 282.67       | 0.011   |
        | 950     | 236.64       | 0.010   |


### After Training
        . Average reward over 100 evaluation episodes: 208.11

```
---
![Training Rewards](rewards_vs_episodes.png)
---

## How it Works (Brief)
```bash
    . The agent learns a Q-value function approximated by a neural network.
    . At each step, it chooses actions using an epsilon-greedy policy balancing exploration and exploitation.
    . Experience replay buffer stores past transitions for sampling during training.
    . A target network stabilizes training by providing fixed Q-targets updated periodically.
```
---
