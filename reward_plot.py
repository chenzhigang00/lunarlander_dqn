import numpy as np
import matplotlib.pyplot as plt

rewards = np.load("DQN_v3/lunarlander_dqn/rewards_2.npy")
plt.plot(rewards)
plt.xlabel("Episode")
plt.ylabel("Reward")
plt.title("Training Reward Curve")
plt.grid(True)
plt.show()
