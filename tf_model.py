import os
import numpy as np

import tensorflow as tf
import tensorflow.keras.layers as layers

class MLP:
    def __init__(self, mlp_layers, learning_rate, batch_size, output_shape, state_size, is_dueling=False, is_distributional=False):
        self.batch_size = batch_size
        self.output_shape = output_shape
        self.is_distributional = is_distributional
        self.model = Q_model(state_size=state_size, output_shape=output_shape, mlp_specs=mlp_layers, is_dueling=is_dueling)
        self.optimizer = tf.keras.optimizers.Adam(learning_rate=learning_rate)
        self.last_loss = None
        if is_distributional:
            self._train = self._distributional_train
        else:
            self._train = self._simple_train
        
    def predict(self, states):
        # returning to the agent the predicted action_values
        action_values = self.model(tf.Variable(states))
        if self.is_distributional:
            action_values = tf.nn.softmax(action_values, axis=-1)
        return action_values.numpy()        
    
    def train(self, states, actions, targets, weights=None):
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
    def _simple_train(self, states, actions, targets, weights):
        with tf.GradientTape() as tape:
            output = self.model(states)
            actions_index = tf.stack([tf.range(self.batch_size), actions], axis=-1)
            restricted_output = tf.gather_nd(output, actions_index)
            td_errors = restricted_output - targets
            weighted_losses = (td_errors) ** 2 * weights
            loss = tf.math.reduce_mean(weighted_losses)
        grads = tape.gradient(loss, self.model.trainable_variables)
        self.optimizer.apply_gradients(zip(grads, self.model.trainable_variables))
        return loss, td_errors  
    
    @tf.function
    def _distributional_train(self, states, actions, targets, weights):
        with tf.GradientTape() as tape:
            logits = self.model(states)
            actions_index = tf.stack([tf.range(self.batch_size), actions], axis=-1)
            selected_action_logits = tf.gather_nd(logits, actions_index)
            losses = tf.nn.softmax_cross_entropy_with_logits(targets, selected_action_logits)
            weighted_losses = losses * weights
            loss = tf.math.reduce_mean(weighted_losses)
        grads = tape.gradient(loss, self.model.trainable_variables)
        self.optimizer.apply_gradients(zip(grads, self.model.trainable_variables))
        return loss, losses
    
    def get_weights(self):
        return self.model.get_weights()
    
    def set_weights(self, weights):
        return self.model.set_weights(weights)
        
class Q_model(tf.keras.Model):

    def __init__(self, state_size=8, output_shape=4, mlp_specs=(120, 84), is_dueling=False, seed=0):
        super(Q_model, self).__init__(self)
        if isinstance(output_shape, int):
            output_shape = (output_shape,)
        self.is_dueling = is_dueling
        self.out_shape = output_shape
        output_neurons = int(np.prod(output_shape))
        state_value_output = output_shape[-1] if len(output_shape) > 1 else 1
        self.fc1 = layers.Dense(mlp_specs[0], activation='relu')
        self.fc2 = layers.Dense(mlp_specs[1], activation='relu')
        if is_dueling:
            self.state_value_fc = layers.Dense(mlp_specs[1], activation='relu')
            self.state_value = layers.Dense(state_value_output)
            self.action_advantage_value_fc = layers.Dense(mlp_specs[1], activation='relu')
            self.action_advantage_value = layers.Dense(output_neurons)
            self.call = self.call_dueling
        else:
            self.q_value = layers.Dense(output_neurons)
            self.call = self.call_simple
    
    @tf.function
    def call_simple(self, input_data):
        x = self.fc1(input_data)
        x = self.fc2(x)
        return tf.reshape(self.q_value(x), (-1,) + self.out_shape)
    
    @tf.function
    def call_dueling(self, input_data):
        x = self.fc1(input_data)
        x = self.fc2(x)
        state_value = self.state_value_fc(x)
        state_value = self.state_value(state_value)
        action_advantage_value = self.action_advantage_value_fc(x)
        action_advantage_value = tf.reshape(self.action_advantage_value(action_advantage_value), (-1,) + self.out_shape)
        action_advantage_value = action_advantage_value - tf.math.reduce_mean(action_advantage_value, axis=1, keepdims=True)
        return state_value[:, None] + action_advantage_value
        