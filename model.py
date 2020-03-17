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
        # The model topography is optimized for the training step, enabling us to propagate the loss
        # calculated on a chosen action. Sharing that topography with the predict flow requires us
        # to pass a bogus actions array. This won't impact neither performance nor output but is hacky.
        # TODO: build 2 architectures, one for update, one for inference and sharing the same weights.
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
        self.model = keras.models.Model(inputs=(inputs, actions), outputs=(selected_action_value, action_values))
        optimizer = keras.optimizers.Adam(lr=learning_rate)
        # the loss for action_values shouldn't be used for training and is therefore set to 0
        self.model.compile(optimizer, loss=['mae', 'mae'], loss_weights=[1, 0])
        
    def predict(self, states):
        # returning to the agent the predicted action_values
        return self.model.predict([states, np.ones((len(states), self.action_space_size))])[1]
    
    def train(self, states, actions, targets, weights):
        # TODO: this is highly inefficient since we're doing 2 forward passes
        predicted = self.predict(states)[list(range(self.batch_size)), actions]
        td_errors = np.abs(predicted - targets)
        # encode actions as a 1-hot vector
        actions = keras.utils.to_categorical(actions, num_classes=self.action_space_size)
        # keras fit effectively compute the weighted td_errors and backpropagate the gradient
        self.model.fit(
            x=[states, actions],
            # need to supply a 'target' for action_values which won't matter because its loss is ignored
            y=[targets, np.ones((self.batch_size, self.action_space_size))],
            sample_weight=[weights, weights],
            verbose=0)
        return td_errors
    
    def get_weights(self):
        return self.model.get_weights()
        
    def update_weights(self, weights):
        self.model.set_weights(weights)