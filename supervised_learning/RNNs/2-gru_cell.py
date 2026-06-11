#!/usr/bin/env python3
import numpy as np


class GRUCell:
    """
    Gated Recurrent Unit (GRU) cell
    """

    def __init__(self, i, h, o):
        """
        i: input dimensionality
        h: hidden state dimensionality
        o: output dimensionality
        """
        self.i = i
        self.h = h
        self.o = o

        # Weights (random normal)
        self.Wz = np.random.randn(i + h, h)
        self.Wr = np.random.randn(i + h, h)
        self.Wh = np.random.randn(i + h, h)
        self.Wy = np.random.randn(h, o)

        # Biases (zeros)
        self.bz = np.zeros((1, h))
        self.br = np.zeros((1, h))
        self.bh = np.zeros((1, h))
        self.by = np.zeros((1, o))

    @staticmethod
    def sigmoid(x):
        """Sigmoid activation"""
        return 1 / (1 + np.exp(-x))

    @staticmethod
    def softmax(x):
        """Softmax activation (stable)"""
        exp_x = np.exp(x - np.max(x, axis=1, keepdims=True))
        return exp_x / np.sum(exp_x, axis=1, keepdims=True)

    def forward(self, h_prev, x_t):
        """
        Forward propagation for one time step

        h_prev: (m, h)
        x_t: (m, i)

        Returns:
            h_next: (m, h)
            y: (m, o)
        """

        # Concatenate input and previous hidden state
        concat = np.concatenate((h_prev, x_t), axis=1)

        # Update gate
        z = self.sigmoid(np.matmul(concat, self.Wz) + self.bz)

        # Reset gate
        r = self.sigmoid(np.matmul(concat, self.Wr) + self.br)

        # Candidate hidden state
        r_h = r * h_prev
        concat_h = np.concatenate((r_h, x_t), axis=1)
        h_tilde = np.tanh(np.matmul(concat_h, self.Wh) + self.bh)

        # Next hidden state
        h_next = (1 - z) * h_prev + z * h_tilde

        # Output
        y = self.softmax(np.matmul(h_next, self.Wy) + self.by)

        return h_next, y
