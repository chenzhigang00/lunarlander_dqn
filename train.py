import torch
import torch.nn as nn
import torch.optim as optim
import gymnasium as gym
import numpy as np
import random
from model import DQN
from utils import ReplayBuffer, seed_everything

BATCH_SIZE = 64
GAMMA = 0.99
LR = 1e-3
BUFFER_SIZE = 100_000
TARGET_UPDATE_FREQ = 1000  
EPS_START = 1.0
EPS_END = 0.05
EPS_DECAY = 100_000
MAX_EPISODES = 1000
MAX_STEPS = 1000

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

env = gym.make("LunarLander-v3", continuous = False, gravity = -10.0,
               enable_wind = False, wind_power=15.0, turbulence_power=1.5, render_mode=None)  # LunarLander-v3 with Gymnasium
seed_everything()
state_dim = env.observation_space.shape[0]
action_dim = env.action_space.n

# DQN引入的两大技术：目标网络和经验回放

# 建立 Q值神经网络
policy_net = DQN(state_dim, action_dim).to(device)
# 目标网络，固定TD目标
# DQN使用目标网络来解决TD目标（有监督训练的标签）随着Q函数更新变动的问题
target_net = DQN(state_dim, action_dim).to(device)
target_net.load_state_dict(policy_net.state_dict())
target_net.eval()

optimizer = optim.Adam(policy_net.parameters(), lr=LR)
# 经验回放池
# 利用随机采样来减小经验数据之间的相关性（强相关，有偏差）
replay_buffer = ReplayBuffer(BUFFER_SIZE)  

# --- ε-greedy ---
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
        next_state, original_reward, terminated, truncated, _ = env.step(action)  # 智能体采样
        x, y, x_dot, y_dot, angle, angular_vel, leg1, leg2 = next_state
        shaped_reward = original_reward

        shaped_reward += -abs(angle) * 0.05

        target_x = 0
        target_y = 0
        distance_from_target = np.sqrt((x - target_x) ** 2 + (y - target_y) ** 2)
        shaped_reward += -distance_from_target * 0.05

        if abs(y) < 0.2:
            shaped_reward += -abs(x_dot) * 0.1

        reward = shaped_reward
        done = terminated or truncated

        replay_buffer.add(state, action, reward, next_state, done)  # 添加采样样本
        state = next_state
        total_reward += reward
        steps_done += 1

        if len(replay_buffer) >= BATCH_SIZE:    # 开始训练
            batch = replay_buffer.sample(BATCH_SIZE)   # 随机采样训练批次样本
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
                next_q_values = target_net(next_states).max(1)[0].unsqueeze(1)  # 基于目标网络计算TD target
                target_q_values = rewards + GAMMA * next_q_values * (1 - dones)

            loss = nn.MSELoss()(q_values, target_q_values)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()   # 更新原始网络（非目标网络）

        # Target network sync
        if steps_done % TARGET_UPDATE_FREQ == 0:
            target_net.load_state_dict(policy_net.state_dict())   # 更新目标网络

        if done:
            break

    episode_rewards.append(total_reward)
    if episode % 10 == 0:
        avg = np.mean(episode_rewards[-10:])
        print(f"Episode {episode}, Avg Reward: {avg:.2f}")

torch.save(policy_net.state_dict(), "dqn_lander.pth")

np.save("rewards.npy", episode_rewards)
