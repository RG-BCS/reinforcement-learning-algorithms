# run.py
import numpy as np

from utils import (
    plot_arrows_MDP_QL,
    plot_grid_world_MDP_QL,
    simulate_policy_MDP_QL,
    print_value_grid
)

# Environment setup
grid_world = [5, 6]
GOAL_REWARD = 10.0
STEP_PENALTY = -1.0

OBSTACLES = [(1, 4), (2, 3), (3, 4)]
GOAL_STATE = (2, 4)

actions = np.array(['up', 'down', 'left', 'right'])
action_moves = {'up': (-1, 0), 'down': (1, 0), 'left': (0, -1), 'right': (0, 1)}
action_to_index = {act: i for i, act in enumerate(actions)}

states = [(j, i) for j in range(grid_world[0]) for i in range(grid_world[1]) if (j, i) not in OBSTACLES]
state_to_index = {s: idx for idx, s in enumerate(states)}
GOAL_INDEX = state_to_index[GOAL_STATE]

num_actions, num_states = len(actions), len(states)
Q_values = np.zeros((num_states, num_actions))


# Environment transition check
def validate_policy_result(current_state, move):
    next_state = current_state[0] + move[0], current_state[1] + move[1]
    if next_state not in OBSTACLES and (0 <= next_state[0] < grid_world[0]) and (0 <= next_state[1] < grid_world[1]):
        return next_state, True
    else:
        return next_state, False


# Q-learning training loop
def run_episodes_update_Q(Q_values, valid_starts, n_episodes, learning_rate=0.1,
                          gamma=1.0, max_steps=100, epsilon=0.1, epsilon_decay=0.995):
    for episode in range(n_episodes):
        s = valid_starts[np.random.choice(len(valid_starts))]
        steps = 0
        while s != GOAL_STATE and steps < max_steps:
            if np.random.rand() < epsilon:
                action_name = np.random.choice(actions)
            else:
                action_name = actions[np.argmax(Q_values[state_to_index[s]])]

            move = action_moves[action_name]
            sp, valid = validate_policy_result(s, move)
            if not valid:
                sp = s  # Bounce back from wall or obstacle

            s_idx = state_to_index[s]
            sp_idx = state_to_index[sp]
            action_idx = action_to_index[action_name]
            reward = GOAL_REWARD if sp == GOAL_STATE else STEP_PENALTY

            Q_values[s_idx, action_idx] += learning_rate * (
                reward + gamma * np.max(Q_values[sp_idx]) - Q_values[s_idx, action_idx]
            )

            s = sp
            steps += 1
        epsilon = max(0.01, epsilon * epsilon_decay)


if __name__ == "__main__":
    # Training config
    learning_rate = 0.1
    gamma = 1.0
    epsilon = 0.1
    epsilon_decay = 0.995
    max_steps = 100
    n_episodes = 2000

    valid_starts = [s for s in states if s != GOAL_STATE]

    # Train
    run_episodes_update_Q(Q_values, valid_starts, n_episodes,
                          learning_rate=learning_rate,
                          gamma=gamma,
                          max_steps=max_steps,
                          epsilon=epsilon,
                          epsilon_decay=epsilon_decay)

    # Visualize policy
    print_value_grid(Q_values)
    print()
    plot_arrows_MDP_QL(Q_values, grid_world, actions, states,
                       goal_state=GOAL_STATE, OBSTACLES=OBSTACLES,
                       state_to_index=state_to_index)

    # Simulate a specific path from a chosen start state
    START_STATE = (4, 0)
    assert START_STATE not in OBSTACLES, "Start state is inside an obstacle!"
    START_INDEX = state_to_index[START_STATE]

    path = simulate_policy_MDP_QL(Q_values, states, actions,
                                  start_index=START_INDEX,
                                  goal_index=GOAL_INDEX,
                                  grid_world=grid_world,
                                  OBSTACLES=OBSTACLES,
                                  action_moves=action_moves)

    plot_arrows_MDP_QL(Q_values, grid_world, actions, states,
                       goal_state=GOAL_STATE, OBSTACLES=OBSTACLES,
                       path=path, state_to_index=state_to_index)

    plot_grid_world_MDP_QL(states, grid_size=grid_world, obstacles=OBSTACLES,
                           goal=GOAL_STATE, path=path, Q_values=Q_values, actions=actions)
