#!/usr/bin/env python3
"""Gradient Descent with L2 Regularization"""

import numpy as np


def l2_reg_gradient_descent(Y, weights, cache, alpha, lambtha, L):
    """Updates weights and biases using gradient descent with L2 regularization

    Args:
        Y: one-hot numpy.ndarray of shape (classes, m) with correct labels
        weights: dict of weights and biases
        cache: dict of all activations of the network
        alpha: learning rate
        lambtha: L2 regularization parameter
        L: number of layers in the network
    """
    m = Y.shape[1]
    dZ = cache['A{}'.format(L)] - Y  # Output layer derivative

    for l in reversed(range(1, L + 1)):
        A_prev = cache['A{}'.format(l - 1)]
        W_key = 'W{}'.format(l)
        b_key = 'b{}'.format(l)
        W = weights[W_key]

        # Gradients with L2 regularization
        dW = (1 / m) * (np.matmul(dZ, A_prev.T) + lambtha * W)
        db = (1 / m) * np.sum(dZ, axis=1, keepdims=True)

        # Update parameters
        weights[W_key] -= alpha * dW
        weights[b_key] -= alpha * db

        # Compute dZ for next layer (only if not the input layer)
        if l > 1:
            dA_prev = np.matmul(W.T, dZ)
            A_prev = cache['A{}'.format(l - 1)]
            dZ = dA_prev * (1 - A_prev ** 2)  # tanh derivative

