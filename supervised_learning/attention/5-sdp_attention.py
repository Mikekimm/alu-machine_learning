#!/usr/bin/env python3
"""Scaled Dot Product Attention"""
import tensorflow as tf
import numpy as np


def sdp_attention(Q, K, V, mask=None):
    """
    Calculates scaled dot product attention.

    Q: query tensor (..., seq_len_q, dk)
    K: key tensor (..., seq_len_v, dk)
    V: value tensor (..., seq_len_v, dv)
    mask: optional (..., seq_len_q, seq_len_v)

    Returns:
        output: (..., seq_len_q, dv)
        weights: (..., seq_len_q, seq_len_v)
    """

    matmul_qk = tf.matmul(Q, K, transpose_b=True)

    dk = tf.cast(tf.shape(K)[-1], tf.float32)
    scaled_logits = matmul_qk / tf.math.sqrt(dk)

    if mask is not None:
        scaled_logits += (mask * -1e9)

    weights = tf.nn.softmax(scaled_logits, axis=-1)

    output = tf.matmul(weights, V)

    return output, weights
