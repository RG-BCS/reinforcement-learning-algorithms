# NxN GridWorld — Q-Learning (No Obstacles)

This project implements **Q-Learning** to solve a deterministic Markov Decision Process (MDP) in an arbitrary-sized NxN GridWorld. It is a clean and scalable framework for understanding value-based reinforcement learning using model-based Q-value iteration.

---

## Problem Overview

- **State Space**: Each cell in an `N x N` grid represents a state.
- **Actions**: The agent can move `Up`, `Down`, `Left`, or `Right`.
- **Rewards**:
  - Reaching the goal state (bottom-right cell): **+10**
  - All other transitions: **-1**
- **Transitions**: Deterministic; the agent stays in place if it tries to move out of bounds.
- **Discount Factor**: γ = 0.98
- **Learning Approach**: Model-based Q-value iteration.

---

## Folder Structure
```bash
MDP_gridworld_q_value/
  ├── general_nxn_q_learning/
    ├── q_learning_nxn.py # Core Q-learning implementation for NxN grid
    ├── demo.ipynb # Notebook with explanation, training, and policy visualization
    ├── demo_script.py # Command-line script to train and print optimal policy
    ├── requirements.txt # Required Python packages
    └── README.md # This file
```

---

## How It Works

```bash
The Q-values are updated over multiple iterations using:

    Q(s, a) = Σ_s' [P(s'|s,a) * (R(s,a,s') + γ * max_a' Q(s',a'))]


This is done in a vectorized fashion with full control over grid topology, allowing future extension to obstacles or stochasticity.
```

---

##  Sample Output
```bash

For a `5x5` GridWorld:
    
    Optimal policy [1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 3 3 3 3 0]

    Optimal Policy Grid:
    ↓ ↓ ↓ ↓ ↓
    ↓ ↓ ↓ ↓ ↓
    ↓ ↓ ↓ ↓ ↓
    ↓ ↓ ↓ ↓ ↓
    → → → → G


This shows the agent prefers going **down** (↓) until the last row, then **right** (→) to reach the goal cell marked as **G**.
```

---

## How to Run

### 1. Install dependencies

```bash
    pip install -r requirements.txt

2. Run the script

    python demo_script.py

3. Or explore the Jupyter notebook

    jupyter notebook demo.ipynb
```




