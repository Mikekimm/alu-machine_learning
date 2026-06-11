#!/usr/bin/env python3
"""Multi-head attention module"""
import tensorflow as tf
sdp_attention = __import__('5-sdp_attention').sdp_attention


class MultiHeadAttention(tf.keras.layers.Layer):
    """Multi-head attention layer"""

    def __init__(self, dm, h):
        super().__init__()

        self.h = h
        self.dm = dm

        self.depth = dm // h

        self.Wq = tf.keras.layers.Dense(dm)
        self.Wk = tf.keras.layers.Dense(dm)
        self.Wv = tf.keras.layers.Dense(dm)

        self.linear = tf.keras.layers.Dense(dm)

    def call(self, Q, K, V, mask):
        """Forward pass"""

        batch_size = tf.shape(Q)[0]

        Q = self.Wq(Q)
        K = self.Wk(K)
        V = self.Wv(V)

        def split_heads(x):
            x = tf.reshape(x, (batch_size, -1, self.h, self.depth))
            return tf.transpose(x, perm=[0, 2, 1, 3])

        Q = split_heads(Q)
        K = split_heads(K)
        V = split_heads(V)

        scaled_attention, weights = sdp_attention(Q, K, V, mask)

        scaled_attention = tf.transpose(scaled_attention, perm=[0, 2, 1, 3])

        concat_attention = tf.reshape(
            scaled_attention,
            (batch_size, -1, self.dm)
        )

        output = self.linear(concat_attention)

        return output, weights
