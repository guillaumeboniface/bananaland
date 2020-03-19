import numpy as np
from torch_model import MLP

class DLQAgent:
    def __init__(self, config, action_space_size, state_size):
        self.__dict__.update(config.as_dict())
        self.action_space_size = action_space_size
        self.state_size = state_size
        self.memory = AgentMemory(
            [(state_size,), None, (state_size,), None, None],
            self.memory_size,
            min_prioritization=self.min_prioritization,
            is_prioritization=self.is_prioritization)
        # using 2 models for fixed targets
        self.trained_model = MLP(self.mlp_layers, self.learning_rate, self.batch_size, action_space_size, state_size, self.is_dueling)
        self.target_model = MLP(self.mlp_layers, self.learning_rate, self.batch_size, action_space_size, state_size, self.is_dueling)
        self.compute_target = self.ddqn_compute_targets if self.is_ddqn else self.simple_compute_targets
        self.step_counter = 0
        if self.is_prioritization:
            self.update = self.update_with_prioritization
            self.prioritization_importance_sampling = self.prioritization_importance_sampling_start
        else:
            self.update = self.simple_update
    
    def act(self, state):
        return np.argmax(self.trained_model.predict(state.reshape((1,) + state.shape))[0])
        
    def step(self, state, action, next_state, reward, done):
        self.memory.add((state, action, next_state, reward, done))
        
        if not self.step_counter % self.update_every and self.memory.size >= self.batch_size:
            self.update()
        
        self.step_counter += 1
        
        if done and self.is_prioritization:
            self.prioritization_importance_sampling += \
                (self.prioritization_importance_sampling_end - self.prioritization_importance_sampling_start) / self.num_episodes
    
    def simple_update(self):
        states, actions, next_states, rewards, dones = self.memory.sample(self.batch_size)
        
        targets = self.compute_target(rewards, next_states, dones)
        self.trained_model.train(states, actions, targets)
        
        self.target_network_update()
        
    def update_with_prioritization(self):
        states, actions, next_states, rewards, dones, prioritizations = self.memory.sample(self.batch_size)
        # for an explanation of the importance of weights to avoid bias due to memory replay prioritization see https://arxiv.org/pdf/1511.05952.pdf
        weights = (self.memory.size * prioritizations) ** (- 1 * self.prioritization_importance_sampling)
        # normalize weights so that they can only lower the update, see https://arxiv.org/pdf/1511.05952.pdf
        weights /= max(weights)
        
        targets = self.compute_target(rewards, next_states, dones)
        td_errors = self.trained_model.train(states, actions, targets, weights)
        new_prioritizations = td_errors ** self.prioritization_exponent
        
        self.target_network_update()
        
    def target_network_update(self):
        # soft_update
        target_model_weights = self.target_model.get_weights()
        train_model_weights = self.trained_model.get_weights()
        new_weights = []
        for w1, w2 in zip(target_model_weights, train_model_weights):
            new_weights.append(w1 * (1 - self.tau) + w2 * self.tau)
        self.target_model.set_weights(new_weights)
            
    def health_check(self):
        return (self.trained_model.last_loss, 0, self.memory.size)
    
    def simple_compute_targets(self, rewards, next_states, dones):
        mask = dones * -1 + 1 # flip 0s and 1s so that we use only the reward for final states
        return rewards + self.gamma * np.max(self.trained_model.predict(next_states), axis=1) * mask
    
    def ddqn_compute_targets(self, rewards, next_states, dones):
        mask = dones * -1 + 1 # flip 0s and 1s so that we use only the reward for final states
        greedy_actions = np.argmax(self.trained_model.predict(next_states), axis=1)
        # implement double DQN, using one model for choosing the action and another to evaluate its value
        return rewards + self.gamma * self.target_model.predict(next_states)[list(range(self.batch_size)), greedy_actions] * mask
        
class AgentMemory:
    def __init__(self, storage_shapes, maxlen, min_prioritization=None, is_prioritization=False):
        self.buckets = []
        for shape in storage_shapes:
            if shape is None:
                self.buckets.append(np.zeros(maxlen))
            else:
                self.buckets.append(np.zeros((maxlen,) + shape))
        self.index = 0
        self.maxlen = maxlen
        self.min_prioritization = min_prioritization
        self.max_prioritization = min_prioritization
        self.is_prioritization = is_prioritization
        if is_prioritization:
            self.last_index = []
            self.prioritizations = np.zeros(maxlen)
            self.sample = self.sample_with_prioritization
        else:
            self.sample = self.simple_sample
        
    def add(self, sample):
        for atom, bucket in zip(sample, self.buckets):
            bucket[self.index % self.maxlen] = atom
        if self.is_prioritization:
            self.prioritizations[self.index % self.maxlen] = self.max_prioritization
        self.index += 1
        
    def simple_sample(self, n):
        samples_index = np.random.choice(min(self.index, self.maxlen), replace=False, size=n)
        return [bucket[samples_index] for bucket in self.buckets]
    
    def sample_with_prioritization(self, n):
        if self.index < self.maxlen:
            probabilities = self.prioritizations[:self.index]/sum(self.prioritizations[:self.index])
        else:
            probabilities = self.prioritizations/sum(self.prioritizations)
        samples_index = np.random.choice(min(self.index, self.maxlen), replace=False, size=n, p=probabilities)
        self.last_index = samples_index
        samples = [bucket[samples_index] for bucket in self.buckets]
        samples.append(self.prioritizations[samples_index])
        return samples
    
    def update_prioritization(self, prioritizations):
        self.prioritizations[self.last_index] = prioritizations + self.min_prioritization
        self.max_prioritization = max(self.prioritizations)
    
    @property
    def size(self):
        return min(self.index, self.maxlen)