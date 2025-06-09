# Q-Learning in a 2×2 GridWorld (Deterministic MDP)

This minimal example demonstrates how Q-values evolve in a **small 2×2 deterministic GridWorld**, helping build intuition about how reinforcement learning agents learn optimal policies in Markov Decision Processes (MDPs).

---

## Environment Description
```bash
- **Grid size:** 2 rows × 2 columns
- **States:** S0 to S3, where `S3` is the **goal**
- **Transitions:** Deterministic — if an action would move the agent off-grid, it stays in place
- **Rewards:**
  - Moving into **S3 (goal)** → `+10`
  - All other transitions → `-1` (to encourage shorter paths)
```

---

## Grid Layout:
```bash

    S0 S1
    S2 S3 (Goal)
```
---

## Actions and Movement

```bash
- **Actions:** `Up`, `Down`, `Left`, `Right`
- Each action maps to a move:  
  - `'Up'`: (-1, 0)  
  - `'Down'`: (1, 0)  
  - `'Left'`: (0, -1)  
  - `'Right'`: (0, 1)
```
---

## Algorithm Summary

```bash

- Implements basic **Q-learning with deterministic transitions**
- Q-values are updated for a fixed number of iterations (`5`) using the Bellman equation
- No learning rate needed since we use full backups (like value iteration)
```
---

## Output
```bash

After 5 iterations, the agent converges to the following **Q-values** and **optimal policy**:

    Q-Value Table:
    [[ 7.439 7.39 7.439 8.61 ]
    [ 6.55 6.55 6.55 9. ]
    [ 7.39 7.439 6.55 8.61 ]
    [ 0. 0. 0. 0. ]] ← Terminal state (goal)

    Optimal Policy Grid:
    → ↓
    → G
```

---


Legend:  
- `→`, `↓` = direction to move  
- `G` = goal state

---

## Files

| File                | Description                                |
|---------------------|--------------------------------------------|
| `q_learning_2x2.py` | Core implementation of the MDP solver      |
| `demo.ipynb`        | Notebook showing code, explanation, output |
| `README.md`         | This file                                  |

---

## What You Learn Here

- How Q-values are derived in simple deterministic settings
- How rewards and transitions shape optimal policy
- A clear stepping stone toward solving more general GridWorlds

---

This 2×2 MDP serves as a foundational exercise before scaling to **arbitrary NxN environments** or adding complexity such as **obstacles**.



