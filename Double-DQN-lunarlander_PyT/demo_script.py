# demo_script.py

"""
This script demonstrates evaluating a pre-trained Double DQN agent on the LunarLander-v2 environment.
It does not train from scratch. Make sure the model is trained or load a pre-trained checkpoint before running.
"""

import gym
import torch
from double_dqn_lunarlander import QT_Network, evaluate_agent

# Environment setup
env = gym.make("LunarLander-v2")
obs, _ = env.reset()
input_dim = env.observation_space.shape[0]
output_dim = env.action_space.n

# Load trained model (you can modify this to load from file if checkpointing is implemented)
model = QT_Network(input_dim, output_dim)
model.load_state_dict(torch.load("double_dqn_lunarlander.pth"))  # Optional: add model saving in main script
model.eval()

# Evaluate agent
print("Evaluating Double DQN agent on LunarLander-v2...")
evaluate_agent(env, model, episodes=10, max_steps=1000, render=True)

env.close()
