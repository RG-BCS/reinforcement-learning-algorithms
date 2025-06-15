# utils.py
import numpy as np
import gym
import tensorflow as tf

def set_seed(seed=65):
    """Set random seeds for reproducibility."""
    np.random.seed(seed)
    tf.random.set_seed(seed)

def play_game(env, actor_model, critic_model, gamma=0.99, epsilon=1e-9):
    """
    Runs one episode in the environment using current actor and critic models.
    Returns states, actions, discounted normalized rewards, raw rewards, and state values.
    """
    states, actions, rewards, state_values = [], [], [], []
    obs = env.reset()
    done = False
    
    while not done:
        states.append(obs)
        # Critic predicts state value
        state_val = critic_model(obs.reshape(1, -1), training=False)[0, 0]
        state_values.append(state_val)
        
        # Actor predicts probability of taking left action
        p_left = actor_model(obs.reshape(1, -1), training=False)[0, 0]
        action = np.random.rand() > p_left  # Stochastic policy
        obs, reward, done, _ = env.step(int(action))
        
        actions.append(action)
        rewards.append(reward)
    
    # Compute discounted rewards
    discounted_rewards = []
    G = 0
    for r in reversed(rewards):
        G = r + gamma * G
        discounted_rewards.insert(0, G)
    discounted_rewards = np.array(discounted_rewards)
    
    # Normalize discounted rewards
    discounted_rewards_norm = (discounted_rewards - discounted_rewards.mean()) / (discounted_rewards.std() + epsilon)
    
    return np.array(states), np.array(actions), discounted_rewards_norm, rewards, state_values

def evaluate_episode(model, env, max_steps=500):
    """
    Run a single episode to evaluate the actor model performance.
    Returns total reward obtained.
    """
    obs = env.reset()
    done = False
    total_reward = 0
    steps = 0
    
    while not done and steps < max_steps:
        p_left = model(obs.reshape(1, -1), training=False)[0, 0].numpy()
        action = np.random.rand() > p_left
        obs, reward, done, _ = env.step(int(action))
        total_reward += reward
        steps += 1
    
    return total_reward
