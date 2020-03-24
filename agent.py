import numpy as np
from tf_model import MLP

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
        # using 2 models for fixed targets and potentially, double DQN
        model_output_shape = action_space_size if not self.is_distributional else (action_space_size, self.distributional_n)
        self.trained_model = MLP(self.mlp_layers, self.learning_rate, self.batch_size, model_output_shape, state_size, self.is_dueling, self.is_distributional)
        self.target_model = MLP(self.mlp_layers, self.learning_rate, self.batch_size, model_output_shape, state_size, self.is_dueling, self.is_distributional)
        self.step_counter = 0
        # adapt the model if we use replay prioritization, see https://arxiv.org/pdf/1511.05952.pdf
        if self.is_prioritization:
            self.update = self.update_with_prioritization
            self.prioritization_importance_sampling = self.prioritization_importance_sampling_start
        else:
            self.update = self.simple_update
        # adapt the model to use a distributional architecture, see https://arxiv.org/pdf/1707.06887.pdf
        if self.is_distributional:
            self.atoms, self.delta_z = self.compute_distribution_params()
            self.compute_targets = self.compute_distributional_targets
            self.act = self.act_distributional
        else:
            self.compute_targets = self.compute_simple_targets
            self.act = self.act_simple
    
    def act_simple(self, state):
        return np.argmax(self.trained_model.predict(state.reshape((1,) + state.shape))[0])
    
    def act_distributional(self, state):
        return np.argmax(self.get_action_value_from_distributional_model(self.trained_model, state.reshape((1,) + state.shape))[0])
        
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
        
        targets = self.compute_targets(rewards, next_states, dones)
        self.trained_model.train(states, actions, targets)
        
        self.target_network_update()
        
    def update_with_prioritization(self):
        states, actions, next_states, rewards, dones, prioritizations = self.memory.sample(self.batch_size)
        # for an explanation of the importance of weights to avoid bias due to memory replay prioritization see https://arxiv.org/pdf/1511.05952.pdf
        weights = (self.memory.size * prioritizations) ** (- 1 * self.prioritization_importance_sampling)
        # normalize weights so that they can only lower the update, see https://arxiv.org/pdf/1511.05952.pdf
        weights /= max(weights)
        
        targets = self.compute_targets(rewards, next_states, dones)
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
    
    def compute_simple_targets(self, rewards, next_states, dones):
        mask = 1 - dones # flip 0s and 1s so that we use only the reward for final states
        if self.is_ddqn:
            # implement double DQN, using one model for choosing the action and another to evaluate its value
            greedy_actions = np.argmax(self.trained_model.predict(next_states), axis=1)
            future_reward = self.gamma * self.target_model.predict(next_states)[list(range(self.batch_size)), greedy_actions]
        else:
            # fixed targets
            future_reward = self.gamma * np.max(self.target_model.predict(next_states), axis=1)
        return rewards + future_reward * mask
    
    def compute_distributional_targets(self, rewards, next_states, dones):
        mask = 1 - dones # flip 0s and 1s so that we use only the reward for final states
        if self.is_ddqn:
            # implement double DQN, using one model for choosing the action and another to evaluate its value
            future_actions = np.argmax(self.get_action_value_from_distributional_model(self.trained_model, next_states), axis=1)
        else:
            future_actions = np.argmax(self.get_action_value_from_distributional_model(self.target_model, next_states), axis=1)
        predictions = self.target_model.predict(next_states)[list(range(self.batch_size)), future_actions]
        
        expanded_atoms = np.tile(self.atoms, self.batch_size).reshape(self.batch_size, self.distributional_n)
        
        Tz = np.clip(rewards.reshape(self.batch_size, 1) + self.gamma * expanded_atoms * mask.reshape(self.batch_size, 1), self.distributional_min, self.distributional_max)
        Tz = np.tile(Tz, self.distributional_n).reshape((self.batch_size, self.distributional_n, self.distributional_n))
        expanded_atoms = expanded_atoms.reshape((self.batch_size, self.distributional_n, 1))
        clipped_quotient = np.clip(1 - np.abs((Tz - expanded_atoms) / self.delta_z), 0, 1)
        
        targets = np.sum(clipped_quotient * predictions.reshape((self.batch_size, 1, self.distributional_n)), axis=-1)

        return targets
        
        
    def get_action_value_from_distributional_model(self, model, states):
        return np.sum(model.predict(states) * self.atoms, axis=-1)                                            
                                                
    def compute_distribution_params(self):
        delta_z = (self.distributional_max - self.distributional_min) / (self.distributional_n - 1)
        atoms = np.array([self.distributional_min + i * delta_z for i in range(self.distributional_n)])
        return atoms, delta_z
        
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