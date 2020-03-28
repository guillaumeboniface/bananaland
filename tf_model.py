import os
import numpy as np

import tensorflow as tf
import tensorflow.keras.layers as layers

tf.random.set_seed(0)

class MLP:
    """
    Multi-layer perception model designed to support a DQN agent. Supports dueling and distributional architectures.
    
    """
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
        self.optimizer = tf.keras.optimizers.Adam(learning_rate=learning_rate)
        # we keep track of the loss for last training step for logging purposes
        self.last_loss = 0
        if is_distributional:
            self.loss = tf.nn.softmax_cross_entropy_with_logits
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
        action_values = self.model(tf.Variable(states))
        if self.is_distributional:
            action_values = tf.nn.softmax(action_values, axis=-1)
        return action_values.numpy()        
    
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
            weights = tf.ones(len(targets))
        states = tf.Variable(states, dtype=tf.float32)
        actions = tf.Variable(actions, dtype=tf.int32)
        targets = tf.Variable(targets, dtype=tf.float32)
        weights = tf.Variable(weights, dtype=tf.float32)
        loss, td_errors = self._train(states, actions, targets, weights)
        self.last_loss = loss.numpy()
        return np.abs(td_errors.numpy())
    
    @tf.function
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
        with tf.GradientTape() as tape:
            output = self.model(states)
            actions_index = tf.stack([tf.range(self.batch_size), actions], axis=-1)
            restricted_output = tf.gather_nd(output, actions_index)
            losses = self.loss(targets, restricted_output)
            weighted_losses = losses * weights
            loss = tf.math.reduce_mean(weighted_losses)
        grads = tape.gradient(loss, self.model.trainable_variables)
        self.optimizer.apply_gradients(zip(grads, self.model.trainable_variables))
        return loss, losses
    
    @tf.function
    def squared_loss(self, targets, predictions):
        """
        Squared loss function 
        
        Parameters
        ----------
        - targets, float Tensor shape=(batch_size,) + output_shape
        - predictions, float Tensor shape=(batch_size,) + output_shape
        
        Return
        ---------
        Losses, float array shape=(batch_size)
        
        """
        return (targets - predictions) ** 2
    
    def get_weights(self):
        """
        Returns the model weights, necessary to update the target model of the DQN 
        
        Return
        ---------
        Model weights, float array, 2D arrays (one sub-array per layer in the model)
        
        """
        return self.model.get_weights()
    
    def set_weights(self, weights):
        """
        Replace the model weights, necessary to update the target model of the DQN 
        
        Parameters
        ---------
        - weights, float array, 2D arrays (one sub-array per layer in the model)
        
        """
        return self.model.set_weights(weights)
        
class Q_model(tf.keras.Model):
    """
    Custom Keras model implementing the MLP architecture.
    
    """

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
        super(Q_model, self).__init__(self)
        if isinstance(output_shape, int):
            output_shape = (output_shape,)
        self.is_dueling = is_dueling
        self.out_shape = output_shape
        output_neurons = int(np.prod(output_shape))
        self.fc = [layers.Dense(spec, activation='relu') for spec in mlp_specs]
        if is_dueling:
            # if distributional needs to output the number of atoms
            self.state_value_out_shape = (1, output_shape[-1]) if len(output_shape) > 1 else (1,)
            self.state_value_fc = layers.Dense(mlp_specs[1], activation='relu')
            self.state_value = layers.Dense(self.state_value_out_shape[-1])
            self.action_advantage_value_fc = layers.Dense(mlp_specs[1], activation='relu')
            self.action_advantage_value = layers.Dense(output_neurons)
            self.call = self.call_dueling
        else:
            self.q_value = layers.Dense(output_neurons)
            self.call = self.call_simple
    
    @tf.function
    def call_simple(self, input_data):
        """
        Implements the forward pass of the Q_model.
        
        Parameters
        ----------
        - input_data, float Tensor shape=(batch_size, state_size)
        
        Return
        ----------
        Predictions, float Tensor shape=(batch_size, action_space_size)

        """
        x = input_data
        for layer in self.fc:
            x = layer(x)
        return tf.reshape(self.q_value(x), (-1,) + self.out_shape)
    
    @tf.function
    def call_dueling(self, input_data):
        """
        Implements the forward pass of the Q_model in the dueling case.
        
        Parameters
        ----------
        - input_data, float Tensor shape=(batch_size, state_size)
        
        Return
        ----------
        Predictions, float Tensor shape=(batch_size, action_space_size, num_atoms)

        """
        x = input_data
        for layer in self.fc:
            x = layer(x)
        state_value = self.state_value_fc(x)
        state_value = tf.reshape(self.state_value(state_value), (-1,) + self.state_value_out_shape)
        action_advantage_value = self.action_advantage_value_fc(x)
        action_advantage_value = tf.reshape(self.action_advantage_value(action_advantage_value), (-1,) + self.out_shape)
        action_advantage_value = action_advantage_value - tf.math.reduce_mean(action_advantage_value, axis=1, keepdims=True)
        return state_value + action_advantage_value
        