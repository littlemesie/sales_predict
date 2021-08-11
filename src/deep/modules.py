# -*- coding:utf-8 -*-

"""
@ide: PyCharm
@author: mesie
@date: 2021/8/10 15:22
@summary:
"""
import tensorflow as tf
from tensorflow.keras.layers import Layer, Dense, Dropout
from tensorflow.keras.layers import SimpleRNNCell

class DNN(Layer):
    """DNN Layer"""
    def __init__(self, hidden_units, activation='relu', dnn_dropout=0., **kwargs):
        """
        DNN part
        :param hidden_units: A list. List of hidden layer units's numbers
        :param activation: A string. Activation function
        :param dnn_dropout: A scalar. dropout number
        """
        super(DNN, self).__init__(**kwargs)
        self.dnn_network = [Dense(units=unit, activation=activation) for unit in hidden_units]
        self.dropout = Dropout(dnn_dropout)

    def call(self, inputs, **kwargs):
        x = inputs
        for dnn in self.dnn_network:
            x = dnn(x)
        x = self.dropout(x)
        return x

class RNN(Layer):
    """RNN Layer"""
    def __init__(self, hidden_units, activation='sigmoid', dropout=0., **kwargs):
        """
        RNN part
        :param hidden_units: A list. List of hidden layer units's numbers
        :param activation: A string. Activation function
        :param dropout: A scalar. dropout number
        """
        super(RNN, self).__init__(**kwargs)
        self.rnn_cells = [SimpleRNNCell(units=unit, dropout=dropout) for unit in hidden_units]
        self.dense = Dense(1, activation=activation)


    def call(self, inputs, **kwargs):
        x = inputs
        for rnn in self.rnn_cells:
            x = rnn(x)

        x = self.dense(x)
        return x