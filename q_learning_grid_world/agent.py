
### `qlearning/agent.py`

import numpy as np

class QLearningAgent:
    def __init__(self, states, actions, goal_state, learning_rate=0.1, gamma=1.0, epsilon=0.1, epsilon_decay=0.995):
        self.states = states
        self.actions = actions
        self.goal_state = goal_state
        self.goal_index = states.index(goal_state)
        self.state_to_index = {s: i for i, s in enumerate(states)}
        self.Q = np.zeros((len(states), len(actions)))
        self.lr = learning_rate
        self.gamma = gamma
        self.epsilon = epsilon
        self.epsilon_decay = epsilon_decay

    def select_action(self, s_idx):
        if np.random.rand() < self.epsilon:
            return np.random.randint(len(self.actions))
        return np.argmax(self.Q[s_idx])

    def update(self, s_idx, a_idx, r, sp_idx):
        td_target = r + self.gamma * np.max(self.Q[sp_idx])
        self.Q[s_idx, a_idx] += self.lr * (td_target - self.Q[s_idx, a_idx])

    def decay_epsilon(self):
        self.epsilon = max(0.01, self.epsilon * self.epsilon_decay)
