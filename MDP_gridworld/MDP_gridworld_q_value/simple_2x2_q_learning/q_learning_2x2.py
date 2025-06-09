import numpy as np

def q_learning_2x2(gamma=0.90, iterations=5, verbose=True):
    """
    Performs Q-value iteration on a simple 2x2 deterministic gridworld.

    Grid layout:
        S0  S1
        S2  S3 (goal)

    - Goal state S3 yields a reward of +10
    - All other transitions give -1
    - Actions are deterministic and bounded by grid edges

    Returns:
        Q_values (np.ndarray): Final Q-value table [state, action]
        policy_symbols (list): Optimal action per state as symbols
    """

    # Define 2x2 gridworld states and deterministic actions
    states = [(0, 0), (0, 1), (1, 0), (1, 1)]  # S0, S1, S2, S3
    actions = ['Up', 'Down', 'Left', 'Right']
    action_moves = {
        'Up': (-1, 0),
        'Down': (1, 0),
        'Left': (0, -1),
        'Right': (0, 1)
    }

    num_states = len(states)
    num_actions = len(actions)
    Q_values = np.zeros((num_states, num_actions))

    transition_prob = np.zeros((num_states, num_actions, num_states))
    rewards = np.zeros((num_states, num_actions, num_states))

    # Populate transition and reward matrices
    for i, state in enumerate(states):
        for j, action in enumerate(actions):
            move = action_moves[action]
            next_state = (state[0] + move[0], state[1] + move[1])

            # Clip next_state to stay within grid boundaries
            next_state = (
                max(min(next_state[0], 1), 0),
                max(min(next_state[1], 1), 0)
            )

            next_state_index = states.index(next_state)

            # Deterministic transition
            transition_prob[i, j, next_state_index] = 1.0

            # Reward setup
            rewards[i, j, next_state_index] = 10 if next_state_index == 3 else -1

    # Perform value iteration on Q-table
    for _ in range(iterations):
        Q_old = Q_values.copy()
        for s in range(num_states):
            for a in range(num_actions):
                Q_values[s, a] = sum([
                    transition_prob[s][a][sp] * (
                        rewards[s][a][sp] + gamma * np.max(Q_old[sp])
                    )
                    for sp in range(num_states)
                ])

    # Determine best action for each state
    best_actions = np.argmax(Q_values, axis=1)
    action_symbols = ['↑', '↓', '←', '→']
    policy_symbols = [action_symbols[a] for a in best_actions]

    if verbose:
        print("Q-Value Table:")
        print(np.round(Q_values, 3))
        print("\nOptimal Policy:")
        print(f"{policy_symbols[0]} {policy_symbols[1]}")
        print(f"{policy_symbols[2]} G")  # G = goal

    return Q_values, policy_symbols


if __name__ == "__main__":
    q_learning_2x2()


 


