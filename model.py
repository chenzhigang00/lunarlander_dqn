import torch
import torch.nn as nn

class DQN(nn.Module):
    def __init__(self, state_dim, action_dim):
        super(DQN, self).__init__()
        self.net = nn.Sequential(
            nn.Linear(state_dim, 64),  # Fully connected layer from input to 128 units
            nn.ReLU(),                 # Activation function
            nn.Linear(64, 64),       # Hidden layer
            nn.ReLU(),                 
            nn.Linear(64, action_dim) # Output layer -> Q-values for each action
        )

    def forward(self, x):
        return self.net(x)