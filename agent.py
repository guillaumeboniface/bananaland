import numpy as np
from tf_model import MLP

class DLQAgent:
    """
    Deep learning agent based on a standard DQN architecture. Implements various options such as:
    - Double DQN (based on https://arxiv.org/pdf/1509.06461.pdf)
    - Dueling DQN (based on https://arxiv.org/pdf/1511.06581.pdf)
    - Prioritized replay DQN (based on https://arxiv.org/pdf/1511.05952.pdf)
    - Distributional DQN (based on https://arxiv.org/pdf/1707.06887.pdf)
    
    All options can be combined. The model itself can only be a multi-layers perceptron (or a dueling 
    architecture on top of a multi-layers perceptron)
    """
    
    def __init__(self, config, action_space_size, state_size):
        """
        Constructor methods to create the agents
        
        Parameters
        ----------
        config - Dictionary containing the following keys:
        - 'gamma', float, discount rate for future rewards
        - 'tau', float, rate for the soft update of the target network
        - 'memory_size', int, size of the replay buffer in number of samples
        - 'batch_size', int, size of the batches sampled to train the model on each update
        - 'update_every', int, update frequency, in number of steps
        - 'mlp_layers', int tuple, shape of the multilayer perceptron model
        - 'learning_rate', float, learning rate for the training of the model
        - 'is_ddqn', boolean, whether to use a double dqn
        - 'is_dueling', boolean, whether to use a dueling architecture
        - 'is_prioritization', boolean, whether to prioritize replay
        - 'min_prioritization', float, minimum prioritization value for a sample
        - 'prioritization_exponent', float, exponent applied to the td errors to get the prioritization weights
        - 'prioritization_importance_sampling_start', float, start of the linearly annealed exponent
        used to adjust the loss and counterbalance the weight sampling when using prioritization
        - 'prioritization_importance_sampling_end', float, end of the linearly annealed exponent
        used to adjust the loss and counterbalance the weight sampling when using prioritization
        - 'is_distributional', boolean, whether to use a distributional architecture
        - 'distributional_n', int, number of atoms for the distributional architecture
        - 'distributional_min', int, minimum support value
        - 'distributional_max', int, maximum support value
        
        action_space_size, int, number of possible actions
        
        state_size, int, length of the state array
        
        """
        self.__dict__.update(config.as_dict())
        self.action_space_size = action_space_size
        self.state_size = state_size
        self.memory = AgentMemory(
            [(state_size,), None, (state_size,), None, None],
            self.memory_size,
            is_prioritization=self.is_prioritization)
        # using 2 models for fixed targets and potentially, double DQN
        model_output_shape = action_space_size if not self.is_distributional else (action_space_size, self.distributional_n)
        self.trained_model = MLP(
            self.mlp_layers,
            self.learning_rate,
            self.batch_size,
            model_output_shape,
            state_size,
            self.is_dueling,
            self.is_distributional)
        self.target_model = MLP(
            self.mlp_layers,
            self.learning_rate,
            self.batch_size,
            model_output_shape,
            state_size,
            self.is_dueling,
            self.is_distributional)
        self.step_counter = 0
        # adapt the model if we use replay prioritization
        if self.is_prioritization:
            self.update = self.update_with_prioritization
            self.prioritization_importance_sampling = self.prioritization_importance_sampling_start
        else:
            self.update = self.simple_update
        # adapt the model to use a distributional architecture
        if self.is_distributional:
            self.atoms, self.delta_z = self.compute_distribution_params()
            self.compute_targets = self.compute_distributional_targets
            self.act = self.act_distributional
        else:
            self.compute_targets = self.compute_simple_targets
            self.act = self.act_simple
    
    def act_simple(self, state):
        """
        Based on a state, returns the on-policy action
        
        Parameter
        ---------
        state - float array
        
        Return
        ---------
        Int, chosen action
        """
        if np.random.random() < self.epsilon:
            return np.random.randint(self.action_space_size)
        return np.argmax(self.trained_model.predict(state.reshape((1,) + state.shape))[0])
    
    def act_distributional(self, state):
        """
        Based on a state, returns the on-policy action for the distributional case, combining
        the estimated probabilities for each atom with the support to estimate Q-values
        
        Parameter
        ---------
        state - float array
        
        Return
        ---------
        Int, chosen action
        """
        if np.random.random() < self.epsilon:
            return np.random.randint(self.action_space_size)
        return np.argmax(self.get_action_value_from_distributional_model(self.trained_model, state.reshape((1,) + state.shape))[0])
        
    def step(self, state, action, next_state, reward, done):
        """
        Based on a state, returns the on-policy action for the distributional case, combining
        the estimated probabilities for each atom with the support to estimate Q-values
        
        Parameter
        ---------
        state - float array shape=(state_size,)
        action - int
        next_state - float array shape=(state_size,)
        reward - float
        done - int
        
        Return
        ---------
        Int, chosen action
        """
        self.memory.add((state, action, next_state, reward, done))
        
        if not self.step_counter % self.update_every and self.memory.size >= self.batch_size:
            self.update()
        
        self.step_counter += 1
        
        if done:
            self.epsilon = max(self.epsilon_min, self.epsilon * self.epsilon_decay)
            if self.is_prioritization:
                self.prioritization_importance_sampling += \
                    (self.prioritization_importance_sampling_end - self.prioritization_importance_sampling_start) / (self.num_episodes - 1)
    
    def simple_update(self):
        """
        Perform an update of the model from a batch of samples
        
        """
        states, actions, next_states, rewards, dones = self.memory.sample(self.batch_size)
        
        targets = self.compute_targets(rewards, next_states, dones)
        self.trained_model.train(states, actions, targets)
        
        self.target_network_update()
        
    def update_with_prioritization(self):
        """
        Performs an update of the model from a batch of samples. The sampling is not uniform but based
        on priorities derived from the temporal difference.
        
        """
        states, actions, next_states, rewards, dones, prioritizations = self.memory.sample(self.batch_size)
        # for an explanation of the importance of weights to avoid bias due to memory replay
        # prioritization see https://arxiv.org/pdf/1511.05952.pdf
        weights = 1 / ((self.memory.size * prioritizations) ** self.prioritization_importance_sampling)
        # normalize weights so that they can only lower the update, see https://arxiv.org/pdf/1511.05952.pdf
        weights /= np.max(weights)
        
        targets = self.compute_targets(rewards, next_states, dones)
        td_errors = self.trained_model.train(states, actions, targets, weights)
        new_prioritizations = (td_errors + self.min_prioritization) ** self.prioritization_exponent
        
        self.memory.update_prioritization(new_prioritizations)
        
        self.target_network_update()
        
    def target_network_update(self):
        """
        Performs a soft update with rate tau from the trained_model to the target_model.
        
        """
        target_model_weights = self.target_model.get_weights()
        train_model_weights = self.trained_model.get_weights()
        new_weights = []
        for w1, w2 in zip(target_model_weights, train_model_weights):
            new_weights.append(w1 * (1 - self.tau) + w2 * self.tau)
        self.target_model.set_weights(new_weights)
            
    def health_check(self):
        """
        Exposes a few key internal metric to help monitor the inner working of the agent
        
        Return
        ----------
        Tuple made of:
        - Index 0, float, loss value from the last update
        - Index 1, float, max prioritization value in the replay buffer
        - Index 2, int, current number of samples held in the replay buffer
        
        """
        return (self.trained_model.last_loss, self.memory.max_prioritization, self.memory.size)
    
    def compute_simple_targets(self, rewards, next_states, dones):
        """
        Computes the target for the training step.
        
        Parameters
        ----------
        - rewards, float array shape=(batch_size,), rewards for the samples
        - next_states, float array shape=(batch_size, state_size), next_states for the samples
        - dones, int array shape=(batch_size,), whether the samples where terminal
        
        Return
        ----------
        Target for the model training, float array shape=(batch_size,)
        
        """
        mask = 1 - dones # flip 0s and 1s so that we use only the reward for final states
        if self.is_ddqn:
            # implement double DQN, using one model for choosing the action and another to evaluate its value
            future_actions = np.argmax(self.trained_model.predict(next_states), axis=1)
            future_reward = self.gamma * self.target_model.predict(next_states)[list(range(self.batch_size)), future_actions]
        else:
            # fixed targets
            future_reward = self.gamma * np.max(self.target_model.predict(next_states), axis=1)
        return rewards + future_reward * mask
    
    def compute_distributional_targets(self, rewards, next_states, dones):
        """
        Computes the target for the training step in a distributional setting. Implementation follows
        the algorithm layed out in https://github.com/google/dopamine/blob/master/dopamine/agents/rainbow/rainbow_agent.py
        
        Parameters
        ----------
        - rewards, float array shape=(batch_size,), rewards for the samples
        - next_states, float array shape=(batch_size, state_size), next_states for the samples
        - dones, int array shape=(batch_size,), whether the samples where terminal
        
        Return
        ----------
        Target for the model training, float array shape=(batch_size, num_atoms)
        
        """
        mask = 1 - dones # flip 0s and 1s so that we use only the reward for final states
        if self.is_ddqn:
            # implement double DQN, using one model for choosing the action and another to evaluate its value
            future_actions = np.argmax(self.get_action_value_from_distributional_model(self.trained_model, next_states), axis=1)
        else:
            # fixed targets
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
        """
        In a distributional setting, computes the Q-values from the model output and the support
        
        Parameters
        ----------
        - model, distributional model instance exposing a predict(states) method
        - states, float array shape=(batch_size, state_size), states from which to calculate Q-values
        
        Return
        ----------
        Q-values, float array, shape=(batch_size, action_space_size)
        
        """
        return np.sum(model.predict(states) * self.atoms, axis=-1)                                            
                                                
    def compute_distribution_params(self):
        """
        For the distributional DQN, computes the support and delta_z values
        
        Return
        ----------
        support, float array, shape=(num_atoms,)
        delta_z, float
        
        """
        delta_z = (self.distributional_max - self.distributional_min) / (self.distributional_n - 1)
        atoms = np.array([self.distributional_min + i * delta_z for i in range(self.distributional_n)])
        return atoms, delta_z
        
class AgentMemory:
    """
    Replay buffer storing samples made of states, actions, next_states, rewards and dones.
    Implemented through numpy arrays. Can support prioritized sampling.
    
    """
    def __init__(self, storage_shapes, maxlen, is_prioritization=False):
        """
        Constructor for AgentMemory
        
        Parameters
        ----------
        - storage_shapes, array, each element describes the shape of the item that will be stored in a bucket
        'None' is used to indicate a scalar bucket.
        - maxlen, int, maximum number of samples to store in the buffer
        - is_prioritization, boolean, whether to use prioritization

        """
        # creating the buckets of each type of items
        self.buckets = []
        for shape in storage_shapes:
            if shape is None:
                self.buckets.append(np.zeros(maxlen))
            else:
                self.buckets.append(np.zeros((maxlen,) + shape))
        self.index = 0
        self.maxlen = maxlen
        self.max_prioritization = 1.0
        self.is_prioritization = is_prioritization
        if is_prioritization:
            # in the prioritization case, maintain a memory of the last samples index to allow for weights updates
            self.last_index = []
            self.prioritizations = np.zeros(maxlen)
            self.sample = self.sample_with_prioritization
        else:
            self.sample = self.simple_sample
        
    def add(self, sample):
        """
        Add a sample to the replay buffer
        
        Parameter
        ---------
        - sample, tuple of items with orders and shapes matching the initial storage_shapes passed to the constructor
        
        """
        for item, bucket in zip(sample, self.buckets):
            bucket[self.index % self.maxlen] = item
        if self.is_prioritization:
            # new samples should always be stored with max_prioritization
            self.prioritizations[self.index % self.maxlen] = self.max_prioritization
        # point to the next empty location in the buffer
        self.index += 1
        
    def simple_sample(self, n):
        """
        Returns n samples from the replay buffer, chosen with uniform probability
        
        Parameter
        ---------
        - n, int, number of samples
        
        Return
        ---------
        - samples, array of buckets of length n
        
        """
        samples_index = np.random.choice(min(self.index, self.maxlen), replace=False, size=n)
        return [bucket[samples_index] for bucket in self.buckets]
    
    def sample_with_prioritization(self, n):
        """
        Returns n samples from the replay buffer, sampled according to respective priorities
        
        Parameter
        ---------
        - n, int, number of samples
        
        Return
        ---------
        - samples, array of buckets of length n, the last bucket in the array are the samples' priorities
        
        """
        if self.index < self.maxlen:
            probabilities = self.prioritizations[:self.index]/sum(self.prioritizations[:self.index])
        else:
            probabilities = self.prioritizations/sum(self.prioritizations)
        samples_index = np.random.choice(min(self.index, self.maxlen), replace=False, size=n, p=probabilities)
        # maintain a memory of the last samples index to allow for weights updates
        self.last_index = samples_index
        samples = [bucket[samples_index] for bucket in self.buckets]
        samples.append(probabilities[samples_index])
        return samples
    
    def update_prioritization(self, prioritizations):
        """
        Updates the priorities of the last sampled elements
        
        Parameter
        ---------
        - prioritizations, float array, length must match the last_index length
        
        """
        self.prioritizations[self.last_index] = prioritizations
        self.max_prioritization = np.max(self.prioritizations)
    
    @property
    def size(self):
        return min(self.index, self.maxlen)