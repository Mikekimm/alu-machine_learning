#!/usr/bin/env python3
"""
RNN Encoder for machine translation using GRU.
"""

import tensorflow as tf


class RNNEncoder(tf.keras.layers.Layer):
    """
    RNN Encoder class that encodes input sequences.
    """

    def __init__(self, vocab, embedding, units, batch):
        """
        Class constructor.

        Args:
            vocab (int): size of input vocabulary
            embedding (int): embedding dimension
            units (int): number of GRU hidden units
            batch (int): batch size
        """
        super(RNNEncoder, self).__init__()

        self.batch = batch
        self.units = units

        self.embedding = tf.keras.layers.Embedding(
            input_dim=vocab,
            output_dim=embedding
        )

        self.gru = tf.keras.layers.GRU(
            units,
            return_sequences=True,
            return_state=True,
            recurrent_initializer='glorot_uniform'
        )

    def initialize_hidden_state(self):
        """
        Initializes hidden state to zeros.

        Returns:
            tensor of shape (batch, units)
        """
        return tf.zeros((self.batch, self.units))

    def call(self, x, initial):
        """
        Forward pass of encoder.

        Args:
            x (tensor): input tensor (batch, input_seq_len)
            initial (tensor): initial hidden state (batch, units)

        Returns:
            outputs, hidden
        """
        x = self.embedding(x)

        outputs, hidden = self.gru(
            x,
            initial_state=initial
        )

        return outputs, hidden
