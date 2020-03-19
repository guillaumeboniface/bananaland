import os
import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

class MLP:
    def __init__(self, mlp_layers, learning_rate, batch_size, action_space_size, state_size, is_dueling=False):
        self.batch_size = batch_size
        self.action_space_size = action_space_size
        self.model = QNetwork(state_size=state_size, action_size=action_space_size, mlp_specs=mlp_layers, is_dueling=is_dueling).to(device)
        self.optimizer = optim.Adam(self.model.parameters(), lr=learning_rate)
        self.criterion = nn.MSELoss()
        self.last_loss = None
        
    def predict(self, states):
        # returning to the agent the predicted action_values
        states = torch.from_numpy(states).float().to(device)
        self.model.eval()
        with torch.no_grad():
            action_values = self.model(states)
        return action_values.cpu().data.numpy()
        
    
    def train(self, states, actions, targets, weights=None):
        if weights is None:
            weights = np.ones(len(targets))
        states = torch.from_numpy(states).float().to(device)
        actions = torch.from_numpy(actions).long().to(device)
        targets = torch.from_numpy(targets).float().to(device)
        weights = torch.from_numpy(weights).float().to(device)
        self.optimizer.zero_grad()
        self.model.train()
        output = self.model(states)
        restricted_output = output[range(len(output)),actions.view(-1)]
        td_errors = restricted_output - targets
        weighted_losses = (td_errors) ** 2 * weights
        loss = torch.mean(weighted_losses)
        self.last_loss = loss.cpu().data.numpy()
        loss.backward()
        self.optimizer.step()
        return np.abs(td_errors.cpu().data.numpy())
    
    def get_weights(self):
        return [w.data for w in self.model.parameters()]
    
    def set_weights(self, weights):
        for w1, w2 in zip(self.model.parameters(), weights):
            w1.data.copy_(w2)
        
class QNetwork(nn.Module):
    """Actor (Policy) Model."""

    def __init__(self, state_size=8, action_size=4, mlp_specs=(120, 84), is_dueling=False, seed=0):
        """Initialize parameters and build model.
        Params
        ======
            state_size (int): Dimension of each state
            action_size (int): Dimension of each action
            seed (int): Random seed
        """
        super(QNetwork, self).__init__()
        self.seed = torch.manual_seed(seed)
        # an affine operation: y = Wx + b
        if is_dueling:
            self.fc1 = nn.Linear(state_size, mlp_specs[0])
            self.fc2 = nn.Linear(mlp_specs[0], mlp_specs[1])
            self.state_value_fc = nn.Linear(mlp_specs[1], mlp_specs[1])
            self.state_value = nn.Linear(mlp_specs[1], 1)
            self.action_advantage_value_fc = nn.Linear(mlp_specs[1], mlp_specs[1])
            self.action_advantage_value = nn.Linear(mlp_specs[1], action_size)
            self.forward = self.forward_dueling
        else:
            self.fc1 = nn.Linear(state_size, mlp_specs[0])
            self.fc2 = nn.Linear(mlp_specs[0], mlp_specs[1])
            self.q_value = nn.Linear(mlp_specs[1], action_size)
            self.forward = self.simple_forward

    def simple_forward(self, state):
        """Build a network that maps state -> action values."""
        x = F.relu(self.fc1(state))
        x = F.relu(self.fc2(x))
        x = self.q_value(x)
        return x
    
    def forward_dueling(self, state):
        x = F.relu(self.fc1(state))
        x = F.relu(self.fc2(x))
        state_value = self.state_value(F.relu(self.state_value_fc(x)))
        action_advantage_value = self.action_advantage_value(F.relu(self.action_advantage_value_fc(x)))
        action_advantage_value = action_advantage_value - torch.mean(action_advantage_value, 1).unsqueeze(1)
        return state_value + action_advantage_value
        