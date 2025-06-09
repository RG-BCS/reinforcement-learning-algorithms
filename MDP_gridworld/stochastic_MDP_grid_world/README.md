# Stochastic MDP Grid World

This repository contains a Python implementation of a **stochastic Markov Decision Process (MDP)** for MxN grid world environment with obstacles, stochastic action outcomes, and a goal state. It demonstrates value iteration for finding an optimal policy, simulating policy rollouts, and visualizing the results.

---

## Project Overview
```bash
The grid world used as an example in this repository is 5x6 environment where an agent navigates through states,
avoiding obstacles to reach a goal state with a high reward. Actions are stochastic — the agent may veer off the
intended direction with certain probabilities, adding realism to the environment.

Key features:
- Stochastic transitions modeled with specified probabilities.
- Value iteration to compute optimal Q-values and policy.
- Support for obstacles and special goal state.
- Policy simulation from arbitrary start states.
- Multiple visualization modes: textual arrows and matplotlib graphical grid.
```

---

## File Structure
```bash

- `stochastic_mdp_grid_world.py`  
  Main implementation of the stochastic MDP, including value iteration, policy simulation, and visualization functions.

- `demo_script.py`  
  Script demonstrating usage of the module: plotting policy and simulating a rollout.

- `demo.ipynb`  
  Jupyter notebook walkthrough of the MDP environment, value iteration, policy simulation, and visualizations.

- `requirements.txt`  
  Python dependencies to run the code.
```

---

## Installation
```bash

    pip install -r requirements.txt
```

---

## Usage
```bash
1. Run the demo script to see the optimal policy and a simulated path from a start state to the goal.

        python demo_script.py

2. Or explore the interactive Jupyter notebook:

        jupyter notebook demo.ipynb
```

---

## Example Output
```bash

Optimal Policy Grid (arrows indicate the best action in each cell):

      →     →     →     →     →     ↓   
      →     →     ↑     ↑     █     ↓   
      →     →     ↑     █     🏆     ←   
      →     →     ↓     ↓     █     ↑   
      →     →     →     →     →     ↑   


Optimal Policy Grid with rollout path from start state (⛳):

      →     →     →     →     →     ↓   
      →     →     ↑     ↑     █     ↓   
      →     →     ↑     █     🏆    7   
      →     →     ↓     ↓     █     6   
      ⛳     1     2     3     4    5
      

. →, ↑, ↓, ←: Optimal actions.

. █: Obstacles.

. 🏆: Goal state.

. ⛳: Start state.

Numbers indicate steps along the rollout path.

```
---



