#!/usr/bin/env python3
import numpy as np

def l2_reg_gradient_descent(Y, weights, cache, alpha, lambtha, L):
    """Updates the weights and biases using gradient descent
    with L2 regularization

    Args:
        Y: one-hot numpy.ndarray of shape (classes, m) with correct labels
        weights: dictionary of weights and biases
        cache: dictionary of all activations of the network
        alpha: learning rate
        lambtha: L2 regularization parameter
        L: number of layers
    """
    m = Y.shape[1]
    dZ = cache[f'A{L}'] - Y

    for l in reversed(range(1, L + 1)):
        A_prev = cache[f'A{l - 1}']
        W = weights[f'W{l}']

        dW = (1 / m) * (np.matmul(dZ, A_prev.T) + lambtha * W)
        db = (1 / m) * np.sum(dZ, axis=1, keepdims=True)

        weights[f'W{l}'] -= alpha * dW
        weights[f'b{l}'] -= alpha * db

        if l > 1:
            dA_prev = np.matmul(W.T, dZ)
            dZ = dA_prev * (1 - A_prev ** 2)  # Derivative of tanh

