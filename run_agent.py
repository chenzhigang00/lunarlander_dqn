import gymnasium as gym
import torch
import numpy as np
from model import DQN

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

env = gym.make("LunarLander-v3", render_mode="None")  # render_mode=Human for pygame based interactive evaluation

state_dim = env.observation_space.shape[0]
action_dim = env.action_space.n

policy_net = DQN(state_dim, action_dim).to(device)
policy_net.load_state_dict(torch.load("dqn_lander.pth", map_location=device))
policy_net.eval()

NUM_EVAL_EPISODES = 100
rewards = []

for ep in range(NUM_EVAL_EPISODES):
    state, _ = env.reset()
    total_reward = 0
    done = False

    while not done:
        with torch.no_grad():
            state_tensor = torch.FloatTensor(state).unsqueeze(0).to(device)
            action = policy_net(state_tensor).argmax().item()

        state, reward, terminated, truncated, _ = env.step(action)
        done = terminated or truncated
        total_reward += reward

    rewards.append(total_reward)
    print(f"Episode {ep+1}: Reward = {total_reward:.2f}")

avg_reward = np.mean(rewards)
print(f"\n Average reward over {NUM_EVAL_EPISODES} episodes: {avg_reward:.2f}")
