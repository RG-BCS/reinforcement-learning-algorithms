# ac_cartpole.py
import warnings
warnings.filterwarnings("ignore", category=DeprecationWarning)

import gym
import numpy as np
import tensorflow as tf
from tensorflow import keras
import matplotlib.pyplot as plt

from utils import set_seed, play_game, evaluate_episode

def train_policy_model(n_episodes, env_name, actor_model, critic_model, critic_loss_fn,
                       actor_optimizer, critic_optimizer, gamma=0.99, epsilon=1e-9, seed=65):
    """
    Train the actor-critic models on the environment.
    """
    reward_history = []
    
    env = gym.make(env_name)
    env.seed(seed)
    env.action_space.seed(seed)
    
    for episode in range(n_episodes):
        with tf.GradientTape(persistent=True) as tape:
            states, actions, discounted_rewards, rewards, state_values = play_game(env, actor_model, critic_model, gamma, epsilon)
            
            states = tf.convert_to_tensor(states, dtype=tf.float32)
            actions = tf.convert_to_tensor(actions, dtype=tf.float32)
            discounted_rewards = tf.convert_to_tensor(discounted_rewards, dtype=tf.float32)
            state_values = tf.convert_to_tensor(state_values, dtype=tf.float32)
            
            # Critic loss: MSE between state values and discounted rewards
            critic_loss = critic_loss_fn(state_values, discounted_rewards)
            
            # Actor loss
            p_left = tf.reshape(actor_model(states), (-1,))
            action_probs = tf.where(actions == 0, p_left, 1 - p_left)
            action_probs = tf.clip_by_value(action_probs, 1e-8, 1.0)
            
            advantage = discounted_rewards - state_values
            advantage = (advantage - tf.reduce_mean(advantage)) / (tf.math.reduce_std(advantage) + epsilon)
            
            log_actions = tf.math.log(action_probs + epsilon)
            actor_loss = -tf.reduce_mean(log_actions * advantage)
        
        # Compute gradients and apply updates
        actor_grads = tape.gradient(actor_loss, actor_model.trainable_variables)
        actor_optimizer.apply_gradients(zip(actor_grads, actor_model.trainable_variables))
        
        critic_grads = tape.gradient(critic_loss, critic_model.trainable_variables)
        critic_optimizer.apply_gradients(zip(critic_grads, critic_model.trainable_variables))
        
        reward_history.append(sum(rewards))
        
        if episode % 50 == 0:
            print(f"Episode {episode:4d} | Reward: {sum(rewards):5.1f} | "
                  f"Actor Loss: {actor_loss.numpy():8.4f} | Critic Loss: {critic_loss.numpy():8.4f}")
    
    return reward_history

def main():
    # Set seeds
    seed = 65
    set_seed(seed)
    
    env_name = 'CartPole-v1'
    n_inputs = 4  # state size for CartPole
    
    # Define actor model (policy network)
    actor_model = keras.Sequential([
        keras.layers.InputLayer(shape=(n_inputs,)),
        keras.layers.Dense(5, activation="elu"),
        keras.layers.Dense(1, activation="sigmoid")
    ])
    
    # Define critic model (value network)
    critic_model = keras.Sequential([
        keras.layers.InputLayer(shape=(n_inputs,)),
        keras.layers.Dense(5, activation="elu"),
        keras.layers.Dense(1)
    ])
    
    critic_loss_fn = keras.losses.MeanSquaredError()
    
    # Optimizers
    lr = 0.01
    actor_optimizer = keras.optimizers.Adam(learning_rate=lr)
    critic_optimizer = keras.optimizers.Adam(learning_rate=lr)
    
    n_episodes = 500
    gamma = 0.99
    test_iters = 100
    
    # Test before training
    test_env = gym.make(env_name)
    test_env.seed(seed)
    test_env.action_space.seed(seed)
    
    initial_test_rewards = [evaluate_episode(actor_model, test_env) for _ in range(test_iters)]
    print(f"Initial test average reward over {test_iters} episodes: {np.mean(initial_test_rewards):.2f}\n")
    
    # Train the model
    reward_history = train_policy_model(n_episodes, env_name, actor_model, critic_model,
                                        critic_loss_fn, actor_optimizer, critic_optimizer,
                                        gamma=gamma, seed=seed)
    
    # Test after training
    final_test_rewards = [evaluate_episode(actor_model, test_env) for _ in range(test_iters)]
    print(f"\nFinal test average reward over {test_iters} episodes: {np.mean(final_test_rewards):.2f}\n")
    
    # Plot training progress
    plt.plot(reward_history)
    plt.xlabel("Episode")
    plt.ylabel("Total Reward")
    plt.title("Training Progress")
    plt.show()

if __name__ == "__main__":
    main()
