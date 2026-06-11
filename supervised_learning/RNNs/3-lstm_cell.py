#!/usr/bin/env python3
import numpy as np


class LSTMCell:
    """
    Represents an LSTM cell.
    """

    def __init__(self, i, h, o):
        """
        i: input size
        h: hidden state size
        o: output size
        """
        self.i = i
        self.h = h
        self.o = o

        # Weights (random normal)
        self.Wf = np.random.randn(i + h, h)
        self.Wu = np.random.randn(i + h, h)
        self.Wc = np.random.randn(i + h, h)
        self.Wo = np.random.randn(i + h, h)
        self.Wy = np.random.randn(h, o)

        # Biases (zeros)
        self.bf = np.zeros((1, h))
        self.bu = np.zeros((1, h))
        self.bc = np.zeros((1, h))
        self.bo = np.zeros((1, h))
        self.by = np.zeros((1, o))

    def sigmoid(self, x):
        return 1 / (1 + np.exp(-x))

    def softmax(self, x):
        e = np.exp(x - np.max(x, axis=1, keepdims=True))
        return e / np.sum(e, axis=1, keepdims=True)

    def forward(self, h_prev, c_prev, x_t):
        """
        Forward propagation for one time step.
        """

        # concatenate h_prev and x_t
        concat = np.concatenate((h_prev, x_t), axis=1)

        # forget gate
        f_t = self.sigmoid(np.matmul(concat, self.Wf) + self.bf)

        # update gate
        u_t = self.sigmoid(np.matmul(concat, self.Wu) + self.bu)

        # intermediate cell state
        c_tilde = np.tanh(np.matmul(concat, self.Wc) + self.bc)

        # next cell state
        c_next = f_t * c_prev + u_t * c_tilde

        # output gate
        o_t = self.sigmoid(np.matmul(concat, self.Wo) + self.bo)

        # next hidden state
        h_next = o_t * np.tanh(c_next)

        # output
        y = self.softmax(np.matmul(h_next, self.Wy) + self.by)

        return h_next, c_next, y
