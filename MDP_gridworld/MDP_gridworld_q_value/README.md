# Grid World MDPs — Deterministic

This repository contains three progressively complex implementations of **deterministic Markov Decision Processes (MDPs)** solved using **Q-Value Iteration**. These models assume **fully deterministic transitions**—i.e., the intended action always results in the expected state change unless blocked (e.g., by obstacles or grid boundaries).

## Implementations

1. **`q_learning_2x2`**  
   A minimal 2x2 grid used to understand the core intuition behind Q-values and policy derivation.

2. **`q_learning_nxn_no_obstacles`**  
   Generalizes to an NxN grid world without obstacles. Demonstrates scalable value iteration and optimal policy generation.

3. **`q_learning_nxn_with_obstacles`**  
   A more realistic grid environment with support for arbitrarily placed obstacles. Includes:
   - Deterministic policy evaluation
   - Optimal path simulation from any start to goal
   - Rich text and graphical visualizations of policies and rollouts

## Important Note

All three implementations here assume **deterministic transitions**. That is:
- The agent always moves in the intended direction unless the move is blocked.
- There is **no probability or randomness** in transitions.

## Coming Next

A follow-up module will extend these ideas to **stochastic MDPs**, where transitions are probabilistic. This will demonstrate how policy and value iteration adapt when outcomes are uncertain.

Stay tuned!
