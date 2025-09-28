#!/usr/bin/env python3
"""L2 regularization cost function"""
import numpy as np


def l2_reg_cost(cost, lambtha, weights, L, m):
    """Calculates the cost of a neural network with L2 regularization

    Args:
        cost: cost without regularization
        lambtha: regularization parameter
        weights: dict of weights and biases
        L: number of layers
        m: number of data points

    Returns:
        Total cost with L2 regularization
    """
    l2_sum = 0
    for l in range(1, L + 1):
        W = weights['W{}'.format(l)]
        l2_sum += np.sum(np.square(W))

    l2_cost = cost + (lambtha / (2 * m)) * l2_sum
    return l2_cost
