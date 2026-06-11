#!/usr/bin/env python3
"""
RNN Decoder with Self-Attention for machine translation.
"""

import tensorflow as tf
SelfAttention = __import__('1-self_attention').SelfAttention


class RNNDecoder(tf.keras.layers.Layer):
    """
    RNN Decoder with attention mechanism.
    """

    def __init__(self, vocab, embedding, units, batch):
        """
        Class constructor.

        Args:
            vocab (int): size of output vocabulary
            embedding (int): embedding dimension
            units (int): GRU hidden units
            batch (int): batch size
        """
        super(RNNDecoder, self).__init__()

        self.units = units
        self.batch = batch

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

        self.F = tf.keras.layers.Dense(vocab)

        self.attention = SelfAttention(units)

    def call(self, x, s_prev, hidden_states):
        """
        Forward pass.

        Args:
            x (tensor): previous word (batch, 1)
            s_prev (tensor): previous hidden state (batch, units)
            hidden_states (tensor): encoder outputs (batch, input_seq_len, units)

        Returns:
            y (tensor): output probabilities (batch, vocab)
            s (tensor): new hidden state (batch, units)
        """

        # Attention
        context, _ = self.attention(s_prev, hidden_states)

        # Embed input word
        x = self.embedding(x)

        # Concatenate context and embedding
        x = tf.concat([tf.expand_dims(context, 1), x], axis=-1)

        # GRU output
        output, s = self.gru(x, initial_state=s_prev)

        # Remove time dimension
        output = tf.reshape(output, (-1, output.shape[2]))

        # Dense layer to vocab
        y = self.F(output)

        return y, s
