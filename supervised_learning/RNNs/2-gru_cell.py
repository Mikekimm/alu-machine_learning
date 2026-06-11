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

        # Weights initialization (random normal)
        self.Wz = np.random.randn(i + h, h)
        self.Wr = np.random.randn(i + h, h)
        self.Wh = np.random.randn(i + h, h)
        self.Wy = np.random.randn(h, o)

        # Biases initialization (zeros)
        self.bz = np.zeros((1, h))
        self.br = np.zeros((1, h))
        self.bh = np.zeros((1, h))
        self.by = np.zeros((1, o))

    @staticmethod
    def softmax(x):
        """Numerically stable softmax"""
        shift = x - np.max(x, axis=1, keepdims=True)
        exp = np.exp(shift)
        return exp / np.sum(exp, axis=1, keepdims=True)

    def forward(self, h_prev, x_t):
        """
        Forward propagation for one time step

        h_prev: (m, h)
        x_t: (m, i)

        returns: h_next, y
        """

        # Concatenate hidden state and input
        concat = np.concatenate((h_prev, x_t), axis=1)

        # Update gate
        z = np.sigmoid(concat @ self.Wz + self.bz)

        # Reset gate
        r = np.sigmoid(concat @ self.Wr + self.br)

        # Candidate hidden state
        r_h_prev = r * h_prev
        concat_candidate = np.concatenate((r_h_prev, x_t), axis=1)
        h_tilde = np.tanh(concat_candidate @ self.Wh + self.bh)

        # Next hidden state
        h_next = (1 - z) * h_prev + z * h_tilde

        # Output
        y_linear = h_next @ self.Wy + self.by
        y = self.softmax(y_linear)

        return h_next, y
