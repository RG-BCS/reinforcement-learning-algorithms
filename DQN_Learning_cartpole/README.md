# Deep Q-Network (DQN) on CartPole-v1

This project implements a Deep Q-Network (DQN) agent to solve the classic **CartPole-v1** environment from OpenAI Gym. The agent learns to balance a pole on a moving cart by interacting with the environment and learning from experience using reinforcement learning.

---

## Environment

- **CartPole-v1**
  - Goal: Keep the pole balanced as long as possible
  - Episode ends if the pole falls or after 500 steps
  - Reward: +1 for every timestep the pole is balanced

---

## Project Highlights

- Built from scratch using **PyTorch**
- Implemented experience replay with a **replay buffer**
- Used a separate **target network** for stable Q-value updates
- Epsilon-greedy exploration with exponential decay
- Achieved **perfect performance (avg reward = 500)** in multiple trials

---

##  Project Structure

```bash
DQN_Learning_CartPole/
├── README.md
├── requirements.txt
├── main.py
├── dqn_agent.py
├── utils.py
├── demo.ipynb
├── demo_script.py

```

---

## Getting Started

### 1. Install dependencies
```bash
    pip install -r requirements.txt

2. Train the agent
    python main.py

3. Evaluate the trained model
    Evaluation will be performed at the end of training (and can be repeated with evaluate_policy() in utils.py).

```
---

## Model Architecture
```bash
Input (state_dim = 4)
   ↓
Linear(4 → 64) + ReLU
   ↓
Linear(64 → 64) + ReLU
   ↓
Linear(64 → action_dim = 2)

. Optimizer: Adam (lr = 1e-2)
. Loss: Huber loss (nn.SmoothL1Loss)
. γ (discount factor): 0.99
. Epsilon decay: epsilon = max(ε_min, ε * ε_decay)


```
---

## Results
```bash
Trial 1 Policy model performance after training 100 episodes average reward: 500.000
Trial 2 Policy model performance after training 100 episodes average reward: 500.000
Trial 3 Policy model performance after training 100 episodes average reward: 500.000
Trial 4 Policy model performance after training 100 episodes average reward: 500.000
Trial 5 Policy model performance after training 100 episodes average reward: 500.000
Trial 6 Policy model performance after training 100 episodes average reward: 500.000
```
