# demo_script.py
import gym
import numpy as np
import tensorflow as tf
from utils import evaluate_episode

def demo(env_name='CartPole-v1', model_path=None, n_episodes=5, seed=65):
    """
    Runs n_episodes in the environment using a trained actor model.
    If model_path is provided, loads weights from there.
    """
    env = gym.make(env_name)
    env.seed(seed)
    env.action_space.seed(seed)
    
    # Define the actor model architecture (must match training)
    n_inputs = env.observation_space.shape[0]
    actor_model = tf.keras.Sequential([
        tf.keras.layers.InputLayer(shape=(n_inputs,)),
        tf.keras.layers.Dense(5, activation="elu"),
        tf.keras.layers.Dense(1, activation="sigmoid")
    ])
    
    if model_path:
        actor_model.load_weights(model_path)
        print(f"Loaded model weights from {model_path}")
    else:
        print("No model path provided. Using untrained model.")
    
    for episode in range(1, n_episodes + 1):
        total_reward = evaluate_episode(actor_model, env)
        print(f"Episode {episode}: Total Reward = {total_reward}")

if __name__ == "__main__":
    # Example usage:
    # demo(model_path='saved_models/actor_model_weights.h5')
    demo()
