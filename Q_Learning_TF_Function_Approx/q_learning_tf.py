import gym
import numpy as np
import tensorflow as tf
from tensorflow import keras
from collections import deque

# Create the environment
env = gym.make('CartPole-v1')
input_dim = env.observation_space.shape[0]
output_dim = env.action_space.n

# Define the Q-network (function approximator)
model = keras.Sequential([
    keras.layers.Input(shape=(input_dim,)),
    keras.layers.Dense(32, activation='elu'),
    keras.layers.Dense(32, activation='elu'),
    keras.layers.Dense(output_dim)
])

# Epsilon-greedy policy
def epsilon_greedy_policy(state, epsilon=0.1):
    if np.random.rand() < epsilon:
        return np.random.randint(output_dim)
    Q_values = model.predict(state[np.newaxis], verbose=0)
    return np.argmax(Q_values[0])

# Experience replay buffer
replay_buffer = deque(maxlen=2000)

def sample_experiences(batch_size):
    indices = np.random.randint(len(replay_buffer), size=batch_size)
    batch = [replay_buffer[i] for i in indices]
    states, actions, rewards, next_states, dones = [
        np.array([experience[k] for experience in batch]) for k in range(5)
    ]
    return states, actions, rewards, next_states, dones

# Play one environment step and store experience
def play_one_step(env, state, epsilon):
    action = epsilon_greedy_policy(state, epsilon)
    next_state, reward, done, _ = env.step(action)
    replay_buffer.append((state, action, reward, next_state, done))
    return next_state, reward, done

# Training step using sampled experiences
def training_step(batch_size):
    states, actions, rewards, next_states, dones = sample_experiences(batch_size)
    next_Q_values = model.predict(next_states, verbose=0)
    max_next_Q_values = np.max(next_Q_values, axis=1)
    target_Q_values = rewards + (1 - dones) * discount_factor * max_next_Q_values

    with tf.GradientTape() as tape:
        all_Q_values = model(states)
        mask = tf.one_hot(actions, output_dim)
        Q_values = tf.reduce_sum(all_Q_values * mask, axis=1)
        loss = tf.reduce_mean(loss_fn(target_Q_values, Q_values))

    grads = tape.gradient(loss, model.trainable_variables)
    optimizer.apply_gradients(zip(grads, model.trainable_variables))

# Training loop
def training_loop(n_episodes, max_steps=500):
    rewards_history = []
    for episode in range(n_episodes):
        state = env.reset()
        total_reward = 0
        for step in range(max_steps):
            epsilon = max(1 - episode / 500, 0.01)
            state, reward, done = play_one_step(env, state, epsilon)
            total_reward += reward
            if done:
                break
            if len(replay_buffer) >= batch_size:
                training_step(batch_size)
        rewards_history.append(total_reward)
        if episode % 50 == 0:
            print(f"Episode {episode}: Total Reward = {total_reward}, Epsilon = {epsilon:.3f}")
    return rewards_history

# Evaluation
def evaluate_policy(episodes=10, max_steps=500):
    total_rewards = []
    for _ in range(episodes):
        state = env.reset()
        total_reward = 0
        for _ in range(max_steps):
            action = np.argmax(model.predict(state[np.newaxis], verbose=0)[0])
            state, reward, done, _ = env.step(action)
            total_reward += reward
            if done:
                break
        total_rewards.append(total_reward)
    return sum(total_rewards) / episodes

def plot_rewards(rewards, window=10):
    import matplotlib.pyplot as plt
    smoothed = [np.mean(rewards[max(0, i - window):(i + 1)]) for i in range(len(rewards))]
    plt.figure(figsize=(10, 5))
    plt.plot(rewards, label='Episode Reward')
    plt.plot(smoothed, label=f'{window}-Episode Moving Average', linewidth=2)
    plt.xlabel('Episode')
    plt.ylabel('Total Reward')
    plt.title('Training Progress')
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()

# Hyperparameters
n_episodes = 600
learning_rate = 1e-3
batch_size = 64
discount_factor = 0.95
test_iters = 100

optimizer = keras.optimizers.Adam(learning_rate=learning_rate)
loss_fn = keras.losses.MeanSquaredError()

# Run training and evaluation
if __name__ == "__main__":
    print(f"Initial average reward over {test_iters} episodes: {evaluate_policy(test_iters):.2f}\n")
    rewards = training_loop(n_episodes)
    print(f"\nFinal average reward over {test_iters} episodes: {evaluate_policy(test_iters):.2f}")
    plot_rewards(rewards)
