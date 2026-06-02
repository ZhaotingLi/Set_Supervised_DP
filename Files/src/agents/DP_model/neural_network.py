import tensorflow as tf
import os
import numpy
import tensorlayer as tl
from keras import backend as K

import tensorflow_probability

import numpy as np
# use image
from tensorflow.keras.applications import ResNet50, VGG16
import math
from tensorflow.keras import layers


class NeuralNetwork:
    def __init__(self, dim_a, dim_a_used, dim_o, act_func_agent="tanh"):
        self.dim_a = dim_a
        self.dim_a_used = dim_a_used
        self.dim_o = dim_o
        self.act_func_agent = act_func_agent

    def policy_model(self):
        """
        Creates the policy model (actor).
        It takes a state representation and outputs an action.
        """
        state_representation_input = tf.keras.layers.Input(shape=(self.dim_o), batch_size=None, name='policy_state_input')
        
        # Dense layers for the policy network
        fc_1 = tf.keras.layers.Dense(512, activation=self.act_func_agent, name='policy_fc_1')(state_representation_input)
        fc_2 = tf.keras.layers.Dense(512, activation=self.act_func_agent, name='policy_fc_2')(fc_1)
        fc_3 = tf.keras.layers.Dense(512, activation=self.act_func_agent, name='policy_fc_3')(fc_2)
        fc_4 = tf.keras.layers.Dense(256, activation=self.act_func_agent, name='policy_fc_4')(fc_3)
        
        # Output layer with sigmoid activation to constrain outputs between 0 and 1
        sigmoid_output = tf.keras.layers.Dense(self.dim_a, activation="sigmoid", name='policy_sigmoid_output')(fc_4)
        
        # Scale and shift the sigmoid output to be in the range [-1, 1]
        # This is a common practice for actions that have symmetric bounds.
        scaled_output = tf.keras.layers.Lambda(lambda x: 2 * x - 1, name='policy_scaled_output')(sigmoid_output)
        
        self.policy_output = scaled_output
        
        # Create the Keras model
        model_policy = tf.keras.Model(inputs=[state_representation_input], outputs=[self.policy_output], name="Policy_Model")

        return model_policy
    
    def action_value_function_model(self):
        """
        Creates the action-value function model.
        """
        # Define the inputs for the model
        state_input = tf.keras.layers.Input(shape=(self.dim_o), batch_size=None, name='q_state_input')
        action_input = tf.keras.layers.Input(shape=(self.dim_a), batch_size=None, name='q_action_input')

        # Concatenate state and action inputs to be fed into the network
        concat_input = tf.concat([state_input, action_input], axis=1, name='q_concat_input')

        # Dense layers for the Q-value network with Layer Normalization
        fc_1 = tf.keras.layers.LayerNormalization()(tf.keras.layers.Dense(512, activation="relu", name='q_fc_1')(concat_input))
        fc_2 = tf.keras.layers.LayerNormalization()(tf.keras.layers.Dense(512, activation="relu", name='q_fc_2')(fc_1))
        fc_3 = tf.keras.layers.LayerNormalization()(tf.keras.layers.Dense(512, activation="relu", name='q_fc_3')(fc_2))
        fc_4 = tf.keras.layers.LayerNormalization()(tf.keras.layers.Dense(256, activation="relu", name='q_fc_4')(fc_3))

        # Output layer for the predicted Q-value
        # Initialized with zeros to encourage initial Q-values to be small.
        q_prediction = tf.keras.layers.Dense(1, name='q_output', kernel_initializer=tf.keras.initializers.Zeros())(fc_4)
        
        self.v_prediction = q_prediction
        
        # Create the Keras model
        model_q_function = tf.keras.Model(inputs=[state_input, action_input], outputs=[self.v_prediction], name="Action_Value_Model")
        return model_q_function

    def Human_model(self):
        """
        Creates the "Human" model.
        The purpose of this model is not immediately clear from the code,
        but it takes a state and action and outputs a prediction of dimension `dim_a`.
        """
        # Define the inputs for the model
        state_input = tf.keras.layers.Input(shape=(self.dim_o), batch_size=None, name='h_state_input')
        action_input = tf.keras.layers.Input(shape=(self.dim_a), batch_size=None, name='h_action_input')

        # Concatenate state and action inputs
        concat_input = tf.concat([state_input, action_input], axis=1, name='h_concat_input')
        
        # Dense layers for the network
        fc_1 = tf.keras.layers.Dense(256, activation="relu", name='h_fc_1')(concat_input)
        fc_2 = tf.keras.layers.Dense(512, activation="relu", name='h_fc_2')(fc_1)
        fc_3 = tf.keras.layers.Dense(128, activation="relu", name='h_fc_3')(fc_2)
        
        # Output layer
        h_prediction = tf.keras.layers.Dense(self.dim_a, name='h_output')(fc_3)

        self.h_prediction = h_prediction
        
        # Create the Keras model
        model_human = tf.keras.Model(inputs=[state_input, action_input], outputs=[self.h_prediction], name="Human_Model")
        return model_human
