import torch
import gymnasium as gym
import numpy as np
from model import DQN

# --- Parameters ---
RENDER_MODE = "human"
EPISODES = 5
MODEL_PATH = "dqn_lander.pth"

# --- Setup ---
env = gym.make("LunarLander-v3", render_mode=RENDER_MODE)
state_dim = env.observation_space.shape[0]
action_dim = env.action_space.n

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# --- Load trained model ---
model = DQN(state_dim, action_dim).to(device)
model.load_state_dict(torch.load(MODEL_PATH, map_location=device))
model.eval()

# --- Run episodes ---
for episode in range(EPISODES):
    state, _ = env.reset()
    total_reward = 0
    done = False

    while not done:
        state_tensor = torch.FloatTensor(state).unsqueeze(0).to(device)
        with torch.no_grad():
            action = model(state_tensor).argmax().item()

        next_state, reward, terminated, truncated, _ = env.step(action)
        done = terminated or truncated
        total_reward += reward
        state = next_state

    print(f"Episode {episode + 1} — Total Reward: {total_reward:.2f}")

env.close()
