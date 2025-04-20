import torch
import torch.nn as nn
import torch.optim as optim
import gymnasium as gym
import numpy as np
import random
from model import DQN
from utils import ReplayBuffer, seed_everything

# --- Hyperparameters ---
BATCH_SIZE = 64
GAMMA = 0.99
LR = 1e-3
BUFFER_SIZE = 100_000
TARGET_UPDATE_FREQ = 1000
EPS_START = 1.0
EPS_END = 0.05
EPS_DECAY = 100_000  # how many steps to decay epsilon
MAX_EPISODES = 1000
MAX_STEPS = 1000

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# --- Initialize ---
env = gym.make("LunarLander-v3", render_mode=None)  # LunarLander-v3 with Gymnasium
seed_everything()
state_dim = env.observation_space.shape[0]
action_dim = env.action_space.n

policy_net = DQN(state_dim, action_dim).to(device)
target_net = DQN(state_dim, action_dim).to(device)
target_net.load_state_dict(policy_net.state_dict())
target_net.eval()

optimizer = optim.Adam(policy_net.parameters(), lr=LR)
replay_buffer = ReplayBuffer(BUFFER_SIZE)

# --- ε-greedy helper ---
def select_action(state, steps_done):
    epsilon = EPS_END + (EPS_START - EPS_END) * np.exp(-1. * steps_done / EPS_DECAY)
    if random.random() < epsilon:
        return env.action_space.sample()
    else:
        with torch.no_grad():
            state_tensor = torch.FloatTensor(state).unsqueeze(0).to(device)
            return policy_net(state_tensor).argmax().item()

# --- Training loop ---
steps_done = 0
episode_rewards = []

for episode in range(MAX_EPISODES):
    state, _ = env.reset()
    total_reward = 0

    for step in range(MAX_STEPS):
        action = select_action(state, steps_done)
        next_state, original_reward, terminated, truncated, _ = env.step(action)
        x, y, x_dot, y_dot, angle, angular_vel, leg1, leg2 = next_state
        shaped_reward = original_reward

        shaped_reward += -abs(angle) * 0.3

        target_x = 0
        target_y = 0
        distance_from_target = np.sqrt((x - target_x) ** 2 + (y - target_y) ** 2)
        shaped_reward += -distance_from_target * 0.5

        if abs(y) < 0.2:
            shaped_reward += -abs(x_dot) * 0.3

        reward = shaped_reward
        done = terminated or truncated

        replay_buffer.push(state, action, reward, next_state, done)
        state = next_state
        total_reward += reward
        steps_done += 1

        # Learn from experience
        if len(replay_buffer) >= BATCH_SIZE:
            batch = replay_buffer.sample(BATCH_SIZE)
            states, actions, rewards, next_states, dones = zip(*batch)

            states = torch.FloatTensor(states).to(device)
            actions = torch.LongTensor(actions).unsqueeze(1).to(device)
            rewards = torch.FloatTensor(rewards).unsqueeze(1).to(device)
            next_states = torch.FloatTensor(next_states).to(device)
            dones = torch.FloatTensor(dones).unsqueeze(1).to(device)

            # Q(s, a)
            q_values = policy_net(states).gather(1, actions)

            # target Q(s', a')
            with torch.no_grad():
                next_q_values = target_net(next_states).max(1)[0].unsqueeze(1)
                target_q_values = rewards + GAMMA * next_q_values * (1 - dones)

            loss = nn.MSELoss()(q_values, target_q_values)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        # Target network sync
        if steps_done % TARGET_UPDATE_FREQ == 0:
            target_net.load_state_dict(policy_net.state_dict())

        if done:
            break

    episode_rewards.append(total_reward)
    if episode % 10 == 0:
        avg = np.mean(episode_rewards[-10:])
        print(f"Episode {episode}, Avg Reward: {avg:.2f}")

# Save model
torch.save(policy_net.state_dict(), "dqn_lander.pth")

# Save rewards
np.save("rewards.npy", episode_rewards)
