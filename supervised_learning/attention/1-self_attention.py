#!/usr/bin/env python3
"""
Self Attention mechanism for machine translation.
"""

import tensorflow as tf


class SelfAttention(tf.keras.layers.Layer):
    """
    Self Attention layer.
    """

    def __init__(self, units):
        """
        Class constructor.

        Args:
            units (int): number of hidden units
        """
        super(SelfAttention, self).__init__()

        self.W = tf.keras.layers.Dense(units)
        self.U = tf.keras.layers.Dense(units)
        self.V = tf.keras.layers.Dense(1)

    def call(self, s_prev, hidden_states):
        """
        Forward pass.

        Args:
            s_prev (tensor): previous decoder hidden state (batch, units)
            hidden_states (tensor): encoder outputs (batch, input_seq_len, units)

        Returns:
            context (tensor): (batch, units)
            weights (tensor): (batch, input_seq_len, 1)
        """

        # expand decoder hidden state
        s_prev_expanded = tf.expand_dims(s_prev, 1)

        # score computation
        score = self.V(tf.nn.tanh(
            self.W(s_prev_expanded) + self.U(hidden_states)
        ))

        # attention weights
        weights = tf.nn.softmax(score, axis=1)

        # context vector
        context = tf.reduce_sum(weights * hidden_states, axis=1)

        return context, weights
