# -*- coding: utf-8 -*-
"""perceptron.ipynb

Automatically generated by Colab.

Original file is located at
    https://colab.research.google.com/drive/1vBkRx3zcZqLB5_Zuq6uHAsPLXGc655Hl
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches

# A simple implementation of value learning using TD(0) in a 2D grid world
# with obstacles, followed by greedy policy rollout and visualization.

grid_world = [5, 6]
GOAL_REWARD = 0.0
STEP_PENALTY = -1.0

OBSTACLES = [(1, 4), (2, 3), (3, 4)]

states = []  # reward -1 except special states mentioned below
for j in range(grid_world[0]):
    for i in range(grid_world[1]):
        if (j, i) not in OBSTACLES:
            states.append((j, i))
state_to_index = {s: idx for idx, s in enumerate(states)}

GOAL_STATE = (2, 4)  # (4,4) #states[-1] # Reward = 0
GOAL_INDEX = states.index(GOAL_STATE)

actions = np.array(['up', 'down', 'left', 'right'])
action_moves = {'up': (-1, 0), 'down': (1, 0), 'left': (0, -1), 'right': (0, 1)}
action_symbol_map = {'up': '↑', 'down': '↓', 'left': '←', 'right': '→'}

S_values = np.zeros(grid_world)
for (i, j) in OBSTACLES:
    S_values[i, j] = -np.inf  # or np.nan, depending on your later use


def validate_policy_result(current_state, move, g_w=grid_world, OBSTACLES=OBSTACLES):
    next_state = current_state[0] + move[0], current_state[1] + move[1]
    if next_state not in OBSTACLES and (0 <= next_state[0] < g_w[0]) and (0 <= next_state[1] < g_w[1]):
        return next_state, True
    else:
        return next_state, False


learning_rate = 0.1
gamma = 1.0
n_iteration = 2000
for _ in range(n_iteration):
    s_val = S_values.copy()
    for i, s in enumerate(states):
        if s == GOAL_STATE:
            continue
        shuffled_actions = actions.copy()
        np.random.shuffle(shuffled_actions)  # Enables exploration — critical for learning
        for j, action_name in enumerate(shuffled_actions):
            move = action_moves[action_name]
            # Check if the action chosen is appropriate otherwise try another one
            sp, pass_fail = validate_policy_result(s, move, g_w=grid_world, OBSTACLES=OBSTACLES)
            if pass_fail:
                r = GOAL_REWARD if state_to_index[sp] == GOAL_INDEX else STEP_PENALTY
                S_values[s[0], s[1]] = (
                    (1 - learning_rate) * s_val[s[0], s[1]] + learning_rate * (r + gamma * s_val[sp[0], sp[1]])
                )
                break


#### Helper functions


def print_value_grid(V):
    print("\nValue Grid:")
    for row in range(V.shape[0]):
        line = ""
        for col in range(V.shape[1]):
            val = V[row, col]
            line += f"{val:6.2f}  "
        print(line)


def plot_arrows_TD0(S_values, grid_world, actions, states, goal_state, OBSTACLES=None, path=None):
    action_symbol_map = {'up': '↑', 'down': '↓', 'left': '←', 'right': '→'}
    action_moves = {'up': (-1, 0), 'down': (1, 0), 'left': (0, -1), 'right': (0, 1)}

    policy_symbols = []
    for s in states:
        if s == goal_state:
            policy_symbols.append('\U0001F3C6')  # 🏆 Goal
            continue
        best_action = None
        best_value = -np.inf
        for action in actions:  # check all the actions and find the action that gives max next state value
            move = action_moves[action]  # pick an action and get the move increments
            sp = s[0] + move[0], s[1] + move[1]  # Get the new landing state
            if (0 <= sp[0] < grid_world[0]) and (0 <= sp[1] < grid_world[1]) and (sp not in OBSTACLES):
                val = S_values[sp[0], sp[1]]  # Get the next state value
                if val > best_value:  # update best value and best action so far for the given state
                    best_value = val
                    best_action = action
        symbol = action_symbol_map[best_action] if best_action else ' '  # Pick the arrow direction
        policy_symbols.append(symbol)

    # Build visual grid
    symbol_grid = []
    OBSTACLE_SYMBOL = '\u2588'
    START_STATE_SYMBOL = "\u26F3"
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

    # Overlay path if provided. This is when user specifies a start state
    STEP_MARK = '\u25CF'
    if path:
        for step_num, (i, j) in enumerate(path):
            if (i, j) == path[-1]:  # goal state already marked so do nothing
                continue
            if (i, j) == path[0]:  # start cell/state
                symbol_grid[i][j] = START_STATE_SYMBOL
            else:
                symbol_grid[i][j] = str(step_num % 10)

    print("Policy (greedy from state-values):")
    for row in symbol_grid:
        print(''.join(f'{cell:^6}' for cell in row))
    print()


def plot_grid_world_TD0(states, grid_size, obstacles, goal, path=None, S_values=None, actions=None):
    GOAL_STATE_SYMBOL = '\U0001F3C6'  # 🏆 Goal
    fig, ax = plt.subplots()
    ax.set_xlim(0, grid_size[1])
    ax.set_ylim(0, grid_size[0])
    ax.set_xticks(range(grid_size[1] + 1))
    ax.set_yticks(range(grid_size[0] + 1))
    ax.set_aspect('equal')
    ax.invert_yaxis()  # So (0,0) is top-left

    # Draw grid
    for i in range(grid_size[0]):
        for j in range(grid_size[1]):
            rect = patches.Rectangle((j, i), 1, 1, linewidth=1, edgecolor='gray', facecolor='white')
            ax.add_patch(rect)
            if (i, j) in obstacles:
                rect.set_facecolor('black')
            elif (i, j) == goal:
                ax.text(j + 0.5, i + 0.5, "G:\u2605", ha='center', va='center', fontsize=20, color='red')

    # Draw policy arrows if S_values are provided
    if S_values is not None and actions is not None:
        action_to_vector = {
            'up': (0, -0.3),
            'down': (0, 0.3),
            'left': (-0.3, 0),
            'right': (0.3, 0),
        }
        for idx, state in enumerate(states):
            if state == goal:
                continue
            best_action = None
            best_value = -np.inf
            for action_name in actions:
                move = action_moves[action_name]
                sp = state[0] + move[0], state[1] + move[1]
                if (0 <= sp[0] < grid_size[0]) and (0 <= sp[1] < grid_size[1]) and (sp not in obstacles):
                    val = S_values[sp[0], sp[1]]
                    if val > best_value:
                        best_value = val
                        best_action = action_name
            dx, dy = action_to_vector[best_action]
            x, y = state[1] + 0.5, state[0] + 0.5
            ax.arrow(x, y, dx, dy, head_width=0.1, head_length=0.1, fc='blue', ec='blue')

    # Draw path
    if path:
        for idx, (i, j) in enumerate(path):
            if (i, j) != goal:
                ax.text(j + 0.5, i + 0.5, str(idx % 10), ha='center', va='center', fontsize=20, color='green')

    ax.grid(True)
    plt.show()


# This is called when the user specifies a start position
def simulate_policy_TD0(S_values, states, actions, start_index, goal_index, action_moves, OBSTACLES, grid_world):
    path = [states[start_index]]
    current_state = states[start_index]
    current_index = start_index
    g_w = grid_world
    while current_index != goal_index:
        best_value = -np.inf
        best_next_state = None
        for action in actions:
            move = action_moves[action]
            next_state = (current_state[0] + move[0], current_state[1] + move[1])
            # Check if next_state valid
            if next_state not in OBSTACLES and 0 <= next_state[0] < g_w[0] and 0 <= next_state[1] < g_w[1]:
                next_value = S_values[next_state[0], next_state[1]]  # Get the next state value
                if next_value > best_value:  # update the next state choice based on max state value
                    best_value = next_value
                    best_next_state = next_state

        if best_next_state is None:
            print("No valid moves from current state. Stuck!")
            break

        current_state = best_next_state
        current_index = state_to_index[current_state]
        path.append(current_state)

        # Optional: Break if stuck in a loop (avoid infinite loops)
        if len(path) > 100:
            print("Too long path, stopping to avoid infinite loop")
            break

    return path


if __name__ == "__main__":
    plot_arrows_TD0(S_values, grid_world, actions, states, goal_state=GOAL_STATE, path=None, OBSTACLES=OBSTACLES)

    START_STATE = (2, 0)
    START_INDEX = states.index(START_STATE)
    path = simulate_policy_TD0(
        S_values,
        states,
        actions,
        start_index=START_INDEX,
        goal_index=GOAL_INDEX,
        action_moves=action_moves,
        OBSTACLES=OBSTACLES,grid_world=grid_world
    )

    plot_grid_world_TD0(states, grid_size=grid_world, obstacles=OBSTACLES, goal=GOAL_STATE,
                    path=path, S_values=S_values, actions=actions)

