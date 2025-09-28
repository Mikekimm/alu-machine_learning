#!/usr/bin/env python3
import numpy as np


def l2_reg_gradient_descent(Y, weights, cache, alpha, lambtha, L):
    """Updates weights and biases using gradient descent with L2 regularization.

    Args:
        Y: one-hot numpy.ndarray of shape (classes, m) with correct labels
        weights: dict of the weights and biases of the network
        cache: dict of the outputs of each layer
        alpha: learning rate
        lambtha: L2 regularization parameter
        L: number of layers in the network
    """
    m = Y.shape[1]
    dZ = cache['A{}'.format(L)] - Y

    for l in reversed(range(1, L + 1)):
        A_prev = cache['A{}'.format(l - 1)]
        W = weights['W{}'.format(l)]

        dW = (1 / m) * (np.matmul(dZ, A_prev.T) + lambtha * W)
        db = (1 / m) * np.sum(dZ, axis=1, keepdims=True)

        weights['W{}'.format(l)] -= alpha * dW
        weights['b{}'.format(l)] -= alpha * db

        if l > 1:
            dA_prev = np.matmul(W.T, dZ)
            dZ = dA_prev * (1 - A_prev ** 2)  # tanh derivative

