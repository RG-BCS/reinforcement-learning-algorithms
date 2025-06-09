# NxN GridWorld — Q-Learning Both With and Without Obstacles (Value Iteration)
```bash
This project implements a general-purpose NxN GridWorld MDP solved using **Value Iteration** to compute optimal Q-values and derive a policy. It supports:

- Arbitrary grid sizes (e.g., 4x4, 6x6, etc.)
- Configurable obstacle positions
- Flexible start and goal states
- Visual and textual policy/path rendering
```

---

## Features
```bash
- **Value Iteration** for Q-value optimization
- Deterministic environment (no stochasticity)
- Reward-shaping with goal reward and step penalty
- Simulate greedy rollout from any start state
- Text-based & Matplotlib visualizations
- Easily configurable grid, obstacles, start/goal
```

---

## Requirements

Install dependencies:

```bash
pip install numpy matplotlib

---

## How to Run
```bash

Option 1. 
    python demo_script.py

Option 2.
    jupyter notebook demo.ipynb

```
---

## Example Output
```bash
### Optimal Policy Grid

        ↓    →    ↓    →    →    ↓
        ↓    █    ↓    ↑    █    ↓
        ↓    █    ↓    ↓    █    ↓
        ↓    █    ↓    ←    █    ↓
        ↓    ↓    ↓    █    █    ↓
        →    →    →    →    →    🏆

### Simulated Path from Start
        
        ↓    →    ↓    →     →    ↓
        ↓    █    ↓    ↑     █    ↓
        ↓    █    ↓    ⛳    █    ↓
        ↓    █    2    1     █    ↓
        ↓    ↓    3    █     █    ↓
        →    →    4    5     6    🏆

⛳ = start

█ = obstacle

🏆 = goal
```
---
## Matplotlib Visualization
```bash

The script also generates a visual version:

    . Blue arrows = policy

    . Red "G★" = goal

    . Green digits = agent path

    . Black cells = obstacles
```



