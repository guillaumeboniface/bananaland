import numpy as np
from torch_model import MLP
from operator import itemgetter

class DLQAgent:
    def __init__(self, config, action_space_size, state_size):
        self.__dict__.update(config.as_dict())
        self.action_space_size = action_space_size
        self.state_size = state_size
        self.memory = AgentMemory(self.memory_size)
        # using 2 models for fixed targets
        self.trained_model = MLP(self.mlp_layers, self.learning_rate, self.batch_size, action_space_size, state_size)
        self.target_model = MLP(self.mlp_layers, self.learning_rate, self.batch_size, action_space_size, state_size)
        self.step_counter = 0
    
    def act(self, state):
        return np.argmax(self.trained_model.predict(state.reshape((1,) + state.shape))[0])
        
    def step(self, state, action, next_state, reward, done):
        self.memory.add((state, action, next_state, reward, done))
        
        if not self.step_counter % self.update_every and self.memory.size >= self.batch_size:
            self.update()
        
        self.step_counter += 1
    
    def update(self):
        states, actions, next_states, rewards, dones = self.memory.sample(self.batch_size)
        mask = dones * -1 + 1 # flip 0s and 1s so that we use only the reward for final states
        targets = rewards + self.gamma * np.max(self.target_model.predict(next_states), axis=1) * mask
        td_errors = self.trained_model.train(states, actions, targets)
        
        # soft_update
        target_model_weights = self.target_model.get_weights()
        train_model_weights = self.trained_model.get_weights()
        for w1, w2 in zip(target_model_weights, train_model_weights):
            w1.data.copy_(self.tau*w2.data + (1.0-self.tau)*w1.data)
            
    def health_check(self):
        return (self.trained_model.last_loss, 0, self.memory.size)
        
class AgentMemory:
    def __init__(self, size):
        self.max_len = size
        self.sample_store = list()
        
    def add(self, sample):
        # new memories should have the max priority, see https://arxiv.org/pdf/1511.05952.pdf
        self.sample_store.append(sample)
        
    def sample(self, n):
        samples_index = np.random.choice(len(self.sample_store), replace=False, size=n)
        samples = [self.sample_store[i] for i in samples_index]
        
        states = np.array([x for x in map(itemgetter(0), samples)])
        actions = np.array([x for x in map(itemgetter(1), samples)])
        next_states = np.array([x for x in map(itemgetter(2), samples)])
        rewards = np.array([x for x in map(itemgetter(3), samples)])
        dones = np.array([x for x in map(itemgetter(4), samples)])
        
        return states, actions, next_states, rewards, dones
    
    @property
    def size(self):
        return len(self.sample_store)
    
    def check_length(self):
        # enforce the max size for the memory by clipping on the left
        if len(self.sample_store) > self.max_len:
            self.sample_store = self.sample_store[len(self.sample_store) - self.max_len:]