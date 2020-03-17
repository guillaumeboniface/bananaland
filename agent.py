import numpy as np
from model import MLP
from operator import itemgetter
from collections import deque

class DLQAgent:
    def __init__(self, config, action_space_size, state_size):
        self.__dict__.update(config.as_dict())
        self.action_space_size = action_space_size
        self.state_size = state_size
        self.memory = AgentMemory(self.memory_size, self.min_prioritization)
        # using 2 models for fixed targets
        self.trained_model = MLP(self.mlp_layers, self.learning_rate, self.batch_size, action_space_size, state_size)
        self.target_model = MLP(self.mlp_layers, self.learning_rate, self.batch_size, action_space_size, state_size)
        self.step_counter = 0
        self.epsilon = self.epsilon_start
        self.prioritization_importance_sampling = self.prioritization_importance_sampling_start
    
    def act(self, state):
        greedy_action = np.argmax(self.trained_model.predict([state])[0])
        return np.random.choice([greedy_action, np.random.randint(self.action_space_size)],
                         p=[1 - self.epsilon, self.epsilon])
        
    def step(self, state, action, next_state, reward, done):
        self.memory.add((state, action, next_state, reward, done))
        
        if not self.step_counter % self.update_every and self.memory.size >= self.batch_size:
            self.update()
        
        self.step_counter += 1
        self.epsilon = max(self.epsilon_min, self.epsilon * self.epsilon_decay)
        if done:
            self.prioritization_importance_sampling += \
                (self.prioritization_importance_sampling_end - self.prioritization_importance_sampling_start) / self.num_episodes
    
    def update(self):
        probabilities, states, actions, next_states, rewards, dones = self.memory.sample(self.batch_size)
        mask = dones * -1 + 1 # flip 0s and 1s so that we use only the reward for final states
        greedy_actions = np.argmax(self.trained_model.predict(next_states), axis=1)
        # TODO: consider multi-step for the target
        # implement double DQN, using one model for choosing the action and another to evaluate its value
        target = rewards + self.gamma * self.target_model.predict(next_states)[list(range(self.batch_size)), greedy_actions] * mask
        # for an explanation of the importance of weights to avoid bias due to memory replay prioritization see https://arxiv.org/pdf/1511.05952.pdf
        weights = (self.memory.size * probabilities) ** (- 1 * self.prioritization_importance_sampling)
        # normalize weights so that they can only lower the update, see https://arxiv.org/pdf/1511.05952.pdf
        weights /= max(weights)
        td_errors = self.trained_model.train(states, actions, target, weights)
        new_prioritizations = td_errors ** self.prioritization_exponent
        self.memory.batch_add_and_prioritize(new_prioritizations, states, actions, next_states, rewards, dones)
        
        # soft_update
        target_model_weights = self.target_model.get_weights()
        train_model_weights = self.trained_model.get_weights()
        for i, weights in enumerate(target_model_weights):
            weights *= (1 - self.tau)
            weights += train_model_weights[i] * self.tau
        self.target_model.update_weights(target_model_weights)

        
class AgentMemory:
    def __init__(self, size, min_prioritization):
        self.max_len = size
        self.max_prioritization = min_prioritization
        self.sample_store = list()
        
    def add(self, sample):
        # new memories should have the max priority, see https://arxiv.org/pdf/1511.05952.pdf
        self.sample_store.append((self.max_prioritization,) + sample)
        
    def batch_add_and_prioritize(self, prioritizations, states, actions, next_states, rewards, dones):
        self.max_prioritization = max(self.max_prioritization, max(prioritizations))
        self.sample_store.extend(zip(prioritizations, states, actions, next_states, rewards, dones))
        self.check_length()
        
    def sample(self, n):
        prioritizations = np.array([x for x in map(itemgetter(0), self.sample_store)])
        probabilities = prioritizations / sum(prioritizations)
        samples_index = np.random.choice(len(self.sample_store), replace=False, size=n, p=probabilities)
        # need the index to be sorted in a descending order otherwise poping one might affect the index of the others
        # this syntax is counterintuitive but performs a reverse sort in place on a numpy array
        samples_index[::-1].sort()
        # remove the sampled memories as we will add them back to the rightmost with a new priority
        samples = [self.sample_store.pop(i) for i in samples_index]
        
        prioritizations = np.array([x for x in map(itemgetter(0), samples)])
        probabilities = prioritizations / sum(prioritizations)
        states = np.array([x for x in map(itemgetter(1), samples)])
        actions = np.array([x for x in map(itemgetter(2), samples)])
        next_states = np.array([x for x in map(itemgetter(3), samples)])
        rewards = np.array([x for x in map(itemgetter(4), samples)])
        dones = np.array([x for x in map(itemgetter(5), samples)])
        
        return probabilities, states, actions, next_states, rewards, dones
    
    @property
    def size(self):
        return len(self.sample_store)
    
    def check_length(self):
        # enforce the max size for the memory by clipping on the left
        if len(self.sample_store) > self.max_len:
            self.sample_store = self.sample_store[len(self.sample_store) - self.max_len:]