import os
import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

class MLP:
    def __init__(self, mlp_layers, learning_rate, batch_size, output_shape, state_size, is_dueling=False, is_distributional=False):
        """
        Constructor for the MLP class
        
        Parameters
        ----------
        - mlp_layers, int tuple, one layer will be created with n hidden units for each n in the tuple
        - learning_rate, float, learning rate used for the model training
        - batch_size, int, number of samples used for the training step
        - output_shape, int or tuple, shapes of the expected model output
        - state_size, int, size of the input space
        - is_dueling, boolean, whether the model should use a dueling architecture
        - is_distributional, boolean, whether the model is distributional
        
        """
        self.batch_size = batch_size
        self.output_shape = output_shape
        self.is_distributional = is_distributional
        self.model = Q_model(state_size=state_size, output_shape=output_shape, mlp_specs=mlp_layers, is_dueling=is_dueling)
        self.optimizer = optim.Adam(self.model.parameters(), lr=learning_rate)
        # we keep track of the loss for last training step for logging purposes
        self.last_loss = 0
        if is_distributional:
            self.loss = self.softmax_cross_entropy_with_logits
        else:
            self.loss = self.squared_loss
        
    def predict(self, states):
        """
        Computes the Q-values for a given state
        
        Parameters
        ----------
        - states, float array shape=(batch_size, state_size)
        
        Return
        ----------
        Q-values, float array shape=(batch_size,) + output_shape        
        
        """
        states = torch.from_numpy(states).float().to(device)
        self.model.eval()
        with torch.no_grad():
            action_values = self.model(states)
            if self.is_distributional:
                action_values = F.softmax(action_values, dim=-1)
        return action_values.cpu().data.numpy()
    
    def train(self, states, actions, targets, weights=None):
        """
        Trains the model 
        
        Parameters
        ----------
        - states, float array shape=(batch_size, state_size)
        - actions, int array shape=(batch_size,)
        - targets, float array shape=(batch_size,) + output_shape
        - weights, float array shape=(batch_size,)
        
        """
        if weights is None:
            weights = np.ones(len(targets))
        states = torch.from_numpy(states).float().to(device)
        actions = torch.from_numpy(actions).long().to(device)
        targets = torch.from_numpy(targets).float().to(device)
        weights = torch.from_numpy(weights).float().to(device)
        loss, td_errors = self._train(states, actions, targets, weights)
        self.last_loss = loss.cpu().data.numpy()
        return np.abs(td_errors.cpu().data.numpy())
    
    def _train(self, states, actions, targets, weights):
        """
        Helper training function, implements the training operation as
        a tensorflow graph
        
        Parameters
        ----------
        - states, float Tensor shape=(batch_size, state_size)
        - actions, int Tensor shape=(batch_size,)
        - targets, float Tensor shape=(batch_size,) + output_shape
        - weights, float Tensor shape=(batch_size,)
        
        """
        self.optimizer.zero_grad()
        self.model.train()
        output = self.model(states)
        restricted_output = output[range(len(output)), actions.view(-1)]
        losses = self.loss(restricted_output, targets)
        weighted_losses = losses * weights
        loss = torch.mean(weighted_losses)
        loss.backward()
        self.optimizer.step()
        return loss, losses
        
    def squared_loss(self, predictions, targets):
        """
        Squared loss function 
        
        Parameters
        ----------
        - targets, float Tensor shape=(batch_size,)
        - predictions, float Tensor shape=(batch_size,)
        
        Return
        ---------
        Losses, float array shape=(batch_size,)
        
        """
        return (predictions - targets) ** 2
    
    def softmax_cross_entropy_with_logits(self, logits, targets):
        """
        Softmax cross entropy with logits loss function 
        
        Parameters
        ----------
        - targets, float Tensor shape=(batch_size, num_atoms)
        - predictions, float Tensor shape=(batch_size, num_atoms)
        
        Return
        ---------
        Losses, float array shape=(batch_size,)
        
        """
        logits_max = torch.max(logits, dim=-1, keepdim=True)[0]
        shifted_logits = logits - logits_max
        
        return torch.sum(-targets * (shifted_logits - torch.log(torch.sum(torch.exp(shifted_logits), dim=-1, keepdim=True))), dim=-1)
    
    def get_weights(self):
        """
        Returns the model weights, necessary to update the target model of the DQN 
        
        Return
        ---------
        Model weights, float array, 2D arrays (one sub-array per layer in the model)
        
        """
        return [w.data for w in self.model.parameters()]
    
    def set_weights(self, weights):
        """
        Replace the model weights, necessary to update the target model of the DQN 
        
        Parameters
        ---------
        - weights, float array, 2D arrays (one sub-array per layer in the model)
        
        """
        for w1, w2 in zip(self.model.parameters(), weights):
            w1.data.copy_(w2)
        
class Q_model(nn.Module):

    def __init__(self, state_size=37, output_shape=4, mlp_specs=(120, 84), is_dueling=False):
        """
        Constructor for the Q_model.
        
        Parameters
        ----------
        - state_size, int, size of the input to the model
        - output_shape, int tuple, shape of the model output
        - mlp_specs, int tuple, one layer will be created with n hidden units for each n in the tuple
        - is_dueling, boolean, whether to use the dueling architecture

        """
        super(Q_model, self).__init__()
        self.seed = torch.manual_seed(0)
        if isinstance(output_shape, int):
            output_shape = (output_shape,)
        self.is_dueling = is_dueling
        self.out_shape = output_shape
        output_neurons = int(np.prod(output_shape))
        self.fc = []
        in_node = state_size
        for spec in mlp_specs:
            self.fc.append(nn.Linear(in_node, spec))
            in_node = spec
        # the layers need to be properties of the class instance for the train operation to work
        for i, fc in enumerate(self.fc):
            setattr(self, 'fc_' + str(i), fc)
        if is_dueling:
            self.state_value_out_shape = (1, output_shape[-1]) if len(output_shape) > 1 else (1,)
            self.state_value_fc = nn.Linear(mlp_specs[1], mlp_specs[1])
            self.state_value = nn.Linear(mlp_specs[1], 1)
            self.action_advantage_value_fc = nn.Linear(mlp_specs[1], mlp_specs[1])
            self.action_advantage_value = nn.Linear(mlp_specs[1], output_neurons)
            self.forward = self.forward_dueling
        else:
            self.q_value = nn.Linear(mlp_specs[-1], output_neurons)
            self.forward = self.forward_simple

    def forward_simple(self, state):
        """
        Implements the forward pass of the Q_model.
        
        Parameters
        ----------
        - input_data, float Tensor shape=(batch_size, state_size)
        
        Return
        ----------
        Predictions, float Tensor shape=(batch_size, action_space_size)

        """
        x = state
        for fc in self.fc:
            x = F.relu(fc(x))
        x = self.q_value(x).view(*((-1,) + self.out_shape))
        return x
    
    def forward_dueling(self, state):
        """
        Implements the forward pass of the Q_model in the dueling case.
        
        Parameters
        ----------
        - input_data, float Tensor shape=(batch_size, state_size)
        
        Return
        ----------
        Predictions, float Tensor shape=(batch_size, action_space_size, num_atoms)

        """
        x = state
        for fc in self.fc:
            x = F.relu(fc(x))
        state_value = self.state_value(F.relu(self.state_value_fc(x))).view(*((-1,) + self.state_value_out_shape))
        action_advantage_value = self.action_advantage_value(F.relu(self.action_advantage_value_fc(x))).view(*(-1,) + self.out_shape)
        action_advantage_value = action_advantage_value - torch.mean(action_advantage_value, 1, keepdim=True)
        return state_value + action_advantage_value