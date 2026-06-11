#!/usr/bin/env python3
"""RNN Cell"""
import numpy as np


class RNNCell:
    """Represents a simple RNN cell"""

    def __init__(self, i, h, o):
        self.i = i
        self.h = h
        self.o = o

        self.Wh = np.random.randn(i + h, h)
        self.Wy = np.random.randn(h, o)

        self.bh = np.zeros((1, h))
        self.by = np.zeros((1, o))

    def forward(self, h_prev, x_t):
        """
        Performs forward propagation for one time step

        Args:
            h_prev: previous hidden state (m, h)
            x_t: input data (m, i)

        Returns:
            h_next: next hidden state (m, h)
            y: output (m, o)
        """

        concat = np.concatenate((h_prev, x_t), axis=1)

        h_linear = np.dot(concat, self.Wh) + self.bh
        h_next = np.tanh(h_linear)

        y_linear = np.dot(h_next, self.Wy) + self.by
        y = self.softmax(y_linear)

        return h_next, y

    @staticmethod
    def softmax(x):
        """Softmax activation function"""
        exp_x = np.exp(x - np.max(x, axis=1, keepdims=True))
        return exp_x / np.sum(exp_x, axis=1, keepdims=True)
