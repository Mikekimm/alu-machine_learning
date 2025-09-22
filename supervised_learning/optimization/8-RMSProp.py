#!/usr/bin/env python3
""" Training with RMSProp (TensorFlow 2.x compatible)
"""

import tensorflow as tf

# Disable eager execution to use TF 1.x-style graph-based ops
tf.compat.v1.disable_eager_execution()


def create_RMSProp_op(loss, alpha, beta2, epsilon):
    """
    Creates the training operation using RMSProp optimization algorithm.

    Args:
        loss: the loss tensor of the neural network
        alpha: learning rate
        beta2: RMSProp decay rate (like momentum)
        epsilon: small number to avoid division by zero

    Returns:
        A training operation that minimizes the loss using RMSProp.
    """
    optimizer = tf.compat.v1.train.RMSPropOptimizer(learning_rate=alpha, decay=beta2, epsilon=epsilon)
    return optimizer.minimize(loss)

