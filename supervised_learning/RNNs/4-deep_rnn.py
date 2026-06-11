#!/usr/bin/env python3
import numpy as np


def softmax(x):
    e = np.exp(x - np.max(x, axis=1, keepdims=True))
    return e / np.sum(e, axis=1, keepdims=True)


def deep_rnn(rnn_cells, X, h_0):
    """
    rnn_cells: list of RNNCell objects (length l)
    X: (t, m, i)
    h_0: (l, m, h)

    Returns:
    H: (t+1, l, m, h)
    Y: (t, m, o) from last layer
    """

    t, m, _ = X.shape
    l = len(rnn_cells)

    # hidden states (include h_0 at time step 0)
    H = np.zeros((t + 1, l, m, rnn_cells[0].h))
    H[0] = h_0

    Y = None

    for step in range(t):
        x_t = X[step]

        for layer in range(l):
            cell = rnn_cells[layer]
            h_prev = H[step, layer]

            # forward pass for this layer
            h_next, y = cell.forward(h_prev, x_t)

            H[step + 1, layer] = h_next

            # input for next layer is current layer output
            x_t = h_next

        # output comes from last layer
        Y = softmax(np.matmul(H[step + 1, -1], rnn_cells[-1].Wy) + rnn_cells[-1].by)

        # if model provides its own softmax inside cell, this still matches expected behavior

    return H, Y
