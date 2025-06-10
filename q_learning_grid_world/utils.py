# utils.py

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches

def print_value_grid(Q):
    print("\nState-Action Grid:")
    for row in range(Q.shape[0]):
        line = ""
        for col in range(Q.shape[1]):
            val = Q[row, col]
            line += f"{val:6.2f}  "
        print(line)

def plot_arrows_MDP_QL(Q_values, grid_world, actions, states, goal_state, path=None, OBSTACLES=None, state_to_index=None):
    action_symbol_map = {'up': '↑','down': '↓','left': '←','right': '→'}
    action_symbols = [action_symbol_map[action] for action in actions]
    policy_indices = np.argmax(Q_values, axis=1)
    policy_symbols = [action_symbols[i] for i in policy_indices]
    
    GOAL_STATE_SYMBOL = '\U0001F3C6'
    if state_to_index:
        policy_symbols[state_to_index[goal_state]] = GOAL_STATE_SYMBOL
    
    symbol_grid = []
    OBSTACLE_SYMBOL = '\u2588'
    idx = 0
    for i in range(grid_world[0]):
        row = []
        for j in range(grid_world[1]):
            if OBSTACLES and (i, j) in OBSTACLES:
                row.append(OBSTACLE_SYMBOL)
            else:
                row.append(policy_symbols[idx])
                idx += 1
        symbol_grid.append(row)

    if path:
        for step_num, (i, j) in enumerate(path):
            if (i, j) == path[-1]:
                continue
            elif (i, j) == path[0]:
                symbol_grid[i][j] = '\u26F3'
            else:
                symbol_grid[i][j] = str(step_num % 10)
    
    print("Optimal Policy Grid:")
    for row in symbol_grid:
        print('    '.join(row))
    print()

def plot_grid_world_MDP_QL(states, grid_size, obstacles, goal, path=None, Q_values=None, actions=None):
    GOAL_STATE_SYMBOL = '\U0001F3C6'
    fig, ax = plt.subplots()
    ax.set_xlim(0, grid_size[1])
    ax.set_ylim(0, grid_size[0])
    ax.set_xticks(range(grid_size[1]+1))
    ax.set_yticks(range(grid_size[0]+1))
    ax.set_aspect('equal')
    ax.invert_yaxis()

    for i in range(grid_size[0]):
        for j in range(grid_size[1]):
            rect = patches.Rectangle((j, i), 1, 1, linewidth=1, edgecolor='gray', facecolor='white')
            ax.add_patch(rect)
            if (i, j) in obstacles:
                rect.set_facecolor('black')
            elif (i, j) == goal:
                ax.text(j+0.5, i+0.5, "G:\u2605", ha='center', va='center', fontsize=20, color='red')

    if Q_values is not None and actions is not None:
        action_to_vector = {
            'up': (0, -0.3),
            'down': (0, 0.3),
            'left': (-0.3, 0),
            'right': (0.3, 0)
        }
        for idx, state in enumerate(states):
            if state == goal:
                continue
            best_action = actions[np.argmax(Q_values[idx])]
            dx, dy = action_to_vector[best_action]
            x, y = state[1] + 0.5, state[0] + 0.5
            ax.arrow(x, y, dx, dy, head_width=0.1, head_length=0.1, fc='blue', ec='blue')

    if path:
        for idx, (i, j) in enumerate(path):
            if (i, j) != goal:
                ax.text(j+0.5, i+0.5, str(idx % 10), ha='center', va='center', fontsize=20, color='green')
    
    ax.grid(True)
    plt.show()

def simulate_policy_MDP_QL(Q_values, states, actions, start_index, goal_index, grid_world, OBSTACLES, action_moves):
    path = [states[start_index]]
    current_index = start_index

    while current_index != goal_index:
        best_action_index = np.argmax(Q_values[current_index])
        move = action_moves[actions[best_action_index]]
        current_state = states[current_index]
        next_state = (current_state[0] + move[0], current_state[1] + move[1])

        next_state = (max(0, min(next_state[0], grid_world[0]-1)),
                      max(0, min(next_state[1], grid_world[1]-1)))
        if next_state in OBSTACLES:
            next_state = current_state

        current_index = states.index(next_state)
        path.append(states[current_index])
    
    return path
