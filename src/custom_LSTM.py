

import tensorflow as tf
import numpy as np
import config

class LSTM_Network(tf.keras.Model):
    def __init__(self,input_size,LSTM_units,dense_units):
        super(LSTM_Network, self).__init__()
        self.input_size = input_size # size of the input_vector
        self.LSTM_units = LSTM_units # a list containing the size of hidden unit in each of the LSTMs
        self.num_of_LSTM_units = len(LSTM_units)
        self.dense_units = dense_units # a list containing the size of each of the dense layers
        self.num_of_dense_layers = len(dense_units)

        # building initial model
        cells = [tf.keras.layers.LSTMCell(self.LSTM_units[i]) for i in range(self.num_of_LSTM_units)]

        self.lstm = tf.keras.layers.StackedRNNCells(cells)
        self.lstm_state = None

        dense_layers= [tf.keras.Input(shape=(self.LSTM_units[-1],))]
        for i in range(self.num_of_dense_layers - 1):
            dense_layers.append(tf.keras.layers.Dense(self.dense_units[i], activation=tf.keras.activations.relu,kernel_regularizer=tf.keras.regularizers.l1(config.l1_lambda),
                                                          bias_regularizer=tf.keras.regularizers.l1(config.l1_lambda)))
        dense_layers.append(tf.keras.layers.Dense(self.dense_units[-1], activation=tf.keras.activations.softmax,kernel_regularizer=tf.keras.regularizers.l1(config.l1_lambda),
                                                          bias_regularizer=tf.keras.regularizers.l1(config.l1_lambda)))
        self.dense_model = tf.keras.Sequential(dense_layers)

    def reset_state(self):
        self.lstm_state = None


    def call(self, inputs, training=None, mask=None):
        if self.lstm_state is None: # we want the current sequence to be independent of the last one
            self.lstm_state = self.lstm.get_initial_state(inputs)
        state = self.lstm_state
        outputs = None
        finals = []
        for i in range(inputs.shape[1]):
            # print(i)
            outputs, state = self.lstm(inputs[:, i], state)
            # print(outputs.shape)
            outputs = self.dense_model(outputs)
            finals.append(outputs)
        self.lstm_state = state
        # todo fix the line below
        # return tf.convert_to_tensor(np.array(finals))
        return outputs





