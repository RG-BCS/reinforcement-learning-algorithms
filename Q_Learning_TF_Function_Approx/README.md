# Q-Learning with Function Approximation (TensorFlow)
```bash
This project implements **Q-learning with a neural network function approximator** on the classic `CartPole-v1`
environment using **TensorFlow**. Unlike Deep Q-Networks (DQN), this implementation **does not use a target
network**. It serves as a foundational exercise in reinforcement learning, translating core theory into working code.
```
---

## Key Features
```bash
- Q-learning with a feedforward neural network
- Epsilon-greedy exploration strategy
- Experience replay buffer
- Online updates without target network
- TensorFlow (Keras API) implementation
- Evaluation routine with average reward tracking
```
---

## What You’ll Learn
```bash
- How to implement function approximation in tabular Q-learning
- How online bootstrapping works in practice
- Where instability can arise without target networks
- TensorFlow basics in the context of reinforcement learning
```
---

## Background
```bash
This project is a stepping stone toward Deep Q-Learning. In standard DQN, a separate **target network** is used to stabilize training. Here, the same network is used for both action selection and bootstrapping, demonstrating the **"off-policy"** nature of Q-learning more directly.
```
---

## Training Results
```bash
| Metric             | Value         |
|--------------------|---------------|
| Episodes           | 600           |
| Max Reward         | 500           |
| Discount Factor    | 0.95          |
| Batch Size         | 64            |

Training progress is printed every 50 episodes.
```
---

##  Project Structure

```bash
    Q_Learning_TF_Function_Approx/
        ├── q_learning_tf.py
        ├── requirements.txt
        └── README.md

```
---

## Getting Started

### 1. Install dependencies
```bash
    pip install -r requirements.txt

### 2. Train the agent
    python q_learning_tf.py

```
---

## Model Architecture
```bash
Input (state_dim = 4)
   ↓
Dense(4 → 32) + elu
   ↓
Dense(32 → 32) + elu
   ↓
Dense(32 → action_dim = 2)

. Optimizer: Adam (lr = 1e-3)
. Loss: Mean Squared Eror (keras.losses.MeanSqauaredError)
. γ (discount factor): 0.95
. Epsilon decay: epsilon = max(1-episode/500, 0.01)

```
---


