# REINFORCE Algorithm (No Baseline) — CartPole-v1

This repository contains an implementation of the **REINFORCE policy gradient algorithm without a baseline** applied to the classic `CartPole-v1` environment from OpenAI Gym.

---

## Overview

The REINFORCE algorithm is a fundamental policy gradient method in reinforcement learning. This implementation uses a stochastic neural network policy trained with Monte Carlo returns and the log-likelihood gradient. It serves as a simple but effective demonstration of policy optimization without variance reduction techniques such as baselines.

---

## Features

- Custom TensorFlow 2 implementation of REINFORCE (no baseline).
- Stochastic policy network outputs action probabilities.
- Trained on `CartPole-v1` environment.
- Evaluates before and after training.
- Visualizes training progress with reward curves.

---

## Installation

1. Clone this repository:

   ```bash
   git clone <your-repo-url>
   cd reinforcement-learning-algorithms/reinforce_no_baseline_PG_method

2. Install required packages:

    pip install -r requirements.txt
```

---

## Usage

```bash
Run the training and evaluation script:

    python demo_script.py

This will:

-Evaluate the untrained policy.
-Train the policy for 500 episodes.
-Evaluate the trained policy.
-Display training reward curves.
```
---

## Results
```bash
## Training Results

The REINFORCE policy gradient method was trained on the `CartPole-v1` environment for 500 episodes.

| Episode | Reward | Loss     |
|---------|--------|----------|
| 0       | 42     | -0.0577  |
| 50      | 26     | -0.3014  |
| 100     | 158    | -0.0640  |
| 150     | 206    | -0.0918  |
| 200     | 358    | -0.0599  |
| 250     | 403    | -0.0208  |
| 300     | 216    | -0.0270  |
| 350     | 500    | -0.0087  |
| 400     | 500    | 0.0331   |
| 450     | 500    | 0.0003   |

- **Test reward before training:** 11.0  
- **Test reward after training:** 500.0 (maximum possible)  
- **Average test reward over 20 episodes:** 500.0  

This demonstrates that the model successfully learned a policy to keep the pole upright for the maximum duration.
```
---

## File Structure

```bash
. reinforce_policy.py: Core implementation of the policy network, training loop, and evaluation.

. demo_script.py: Script to train and evaluate the policy from the command line.

. demo.ipynb: Interactive Jupyter notebook showcasing usage and step-by-step explanation.

. requirements.txt: Python dependencies for running the code.
```

---

## Next Steps
```bash

This project demonstrates REINFORCE without baseline. Future improvements can include:

. Adding baseline functions (e.g., value function) to reduce variance.

. Implementing Actor-Critic methods.

. Testing on more complex environments.
