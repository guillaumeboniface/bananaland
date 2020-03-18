import os
import numpy as np
#os.environ["KERAS_BACKEND"] = "plaidml.keras.backend"

import keras
from keras import layers
from keras import backend as K

class ActionSelection(layers.Layer):
    def compute_output_shape(self, input_shape):
        assert(len(input_shape) == 2) # this layer takes 2 inputs
        return input_shape[0][:-1] + (1,)

    def call(self, inputs):
        action_values, actions = inputs
        # assume actions encoded as a 1-hot vector
        return K.sum(action_values * actions, axis=-1)

class MLP:
    def __init__(self, mlp_layers, learning_rate, batch_size, action_space_size, state_size):
        self.batch_size = batch_size
        self.action_space_size = action_space_size
        inputs = layers.Input(shape=(state_size,))
        actions = layers.Input(shape=(action_space_size,))
        x = inputs
        for n in mlp_layers:
            x = layers.Dense(n, activation='relu')(x)
        action_values = layers.Dense(action_space_size)(x)
        selected_action_value = ActionSelection()([action_values, actions])
        # the selected_action_value is useful as an output for training
        # while action_values are used for agent action
        self.trainable_model = keras.models.Model(inputs=(inputs, actions), outputs=selected_action_value)
        self.inference_model = keras.models.Model(inputs=inputs, outputs=action_values)
        optimizer = keras.optimizers.Adam(lr=learning_rate)
        self.trainable_model.compile(optimizer, loss='mse')
        self.inference_model.compile(optimizer, loss='mse')
        self.last_loss = None
        
    def predict(self, states):
        # returning to the agent the predicted action_values
        return self.inference_model.predict(states)
    
    def train(self, states, actions, targets, weights=None):
        if weights is None:
            weights = np.ones(len(targets))
        # TODO: this is highly inefficient since we're doing 2 forward passes
        predicted = self.predict(states)[list(range(self.batch_size)), actions]
        td_errors = np.abs(predicted - targets)
        # encode actions as a 1-hot vector
        actions = keras.utils.to_categorical(actions, num_classes=self.action_space_size)
        # keras fit effectively compute the weighted td_errors and backpropagate the gradient
        history = self.trainable_model.fit(
            x=[states, actions],
            y=targets,
            sample_weight=weights,
            verbose=0)
        self.last_loss = history.history['loss'][-1]
        return td_errors
    
    def get_weights(self):
        return self.trainable_model.get_weights()
    
    def set_weights(self, weights):
        self.trainable_model.set_weights(weights)
        for w1, w2 in zip(self.trainable_model.get_weights(), self.inference_model.get_weights()):
            assert(np.all(w1 == w2))