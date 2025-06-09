"""
Q-Learning implementation for a deterministic NxN Gridworld without obstacles.

Goal:
- Reach the bottom-right corner of the grid for a reward of +10.
- Each step has a penalty of -1 to encourage shorter paths.

States are grid coordinates (row, col).
Actions are deterministic moves: up, down, left, right.
"""

import numpy as np

class QLearningNxN:
    def __init__(self, grid_size=(5,5), gamma=0.98):
        self.grid_size = grid_size
        self.gamma = gamma
        
        self.states = [(r, c) for r in range(grid_size[0]) for c in range(grid_size[1])]
        self.goal_state = self.states[-1]
        self.goal_index = self.states.index(self.goal_state)
        
        self.actions = np.array(['up', 'down', 'left', 'right'])
        self.action_moves = {
            'up': (-1, 0),
            'down': (1, 0),
            'left': (0, -1),
            'right': (0, 1)
        }
        
        self.transition_probability = np.zeros((len(self.states), len(self.actions), len(self.states)))
        self.reward_matrix = np.zeros((len(self.states), len(self.actions), len(self.states)))
        self.Q_values = np.zeros((len(self.states), len(self.actions)))
        
        self._build_transition_and_rewards()
        
    def _build_transition_and_rewards(self):
        """Build deterministic transition probabilities and rewards."""
        for i, state in enumerate(self.states):
            if i == self.goal_index:
                continue  # Terminal state
            
            for j, action in enumerate(self.actions):
                move = self.action_moves[action]
                next_state = (state[0] + move[0], state[1] + move[1])
                
                # Clamp to grid boundaries
                next_state_clamped = (
                    max(min(next_state[0], self.grid_size[0] - 1), 0),
                    max(min(next_state[1], self.grid_size[1] - 1), 0)
                )
                next_index = self.states.index(next_state_clamped)
                
                self.transition_probability[i, j, next_index] = 1.0
                
                # Reward +10 if next state is goal, else -1
                if next_index == self.goal_index:
                    self.reward_matrix[i, j, next_index] = 10
                else:
                    self.reward_matrix[i, j, next_index] = -1
    
    def train(self, n_iterations=50):
        """Run Q-learning iterations to update Q-values."""
        for _ in range(n_iterations):
            q_old = self.Q_values.copy()
            for s in range(len(self.states)):
                for a in range(len(self.actions)):
                    self.Q_values[s, a] = sum([
                        self.transition_probability[s, a, sp] *
                        (self.reward_matrix[s, a, sp] + self.gamma * np.max(q_old[sp]))
                        for sp in range(len(self.states))
                    ])
    
    def get_optimal_policy(self):
        """Return the best action indices and corresponding symbols for each state."""
        action_symbol_map = {'up': '↑', 'down': '↓', 'left': '←', 'right': '→'}
        best_action_indices = np.argmax(self.Q_values, axis=1)
        policy_symbols = [action_symbol_map[self.actions[i]] for i in best_action_indices]
        policy_symbols[self.goal_index] = 'G'  # Mark goal
        return policy_symbols.reshape(self.grid_size)
    
    def print_policy(self):
        """Print the optimal policy grid."""
        policy_grid = self.get_optimal_policy()
        print("Optimal Policy Grid:")
        for row in policy_grid:
            print(' '.join(row))

# Example usage
if __name__ == "__main__":
    ql = QLearningNxN(grid_size=(5,5), gamma=0.98)
    ql.train(n_iterations=50)
    ql.print_policy()
