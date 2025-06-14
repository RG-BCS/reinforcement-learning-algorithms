import gym
import torch
from per_double_dqn_lunarlander import QT_Network

def demo():
    env = gym.make("LunarLander-v2")
    state = env.reset()
    if isinstance(state, tuple):  # Support for gym reset returning (obs, info)
        state = state[0]

    input_dim = env.observation_space.shape[0]
    output_dim = env.action_space.n

    # Load the trained model
    model = QT_Network(input_dim, output_dim)
    model.load_state_dict(torch.load("q_network.pth"))
    model.eval()

    episodes = 3
    for episode in range(episodes):
        state = env.reset()
        if isinstance(state, tuple):
            state = state[0]
        done = False
        total_reward = 0
        while not done:
            env.render()
            state_tensor = torch.FloatTensor(state).unsqueeze(0)
            with torch.no_grad():
                action = torch.argmax(model(state_tensor)).item()

            result = env.step(action)
            if len(result) == 5:
                next_state, reward, terminated, truncated, _ = result
                done = terminated or truncated
            else:
                next_state, reward, done, _ = result

            total_reward += reward
            state = next_state
        print(f"Episode {episode+1} reward: {total_reward:.2f}")

    env.close()

if __name__ == "__main__":
    demo()
