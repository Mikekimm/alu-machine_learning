#!/usr/bin/env python3
"""Positional encoding for transformer models."""
import numpy as np


def positional_encoding(max_seq_len, dm):
    """
    Calculates positional encoding for a transformer.

    max_seq_len: maximum sequence length
    dm: model depth

    Returns: numpy.ndarray of shape (max_seq_len, dm)
    """
    PE = np.zeros((max_seq_len, dm))

    position = np.arange(max_seq_len)[:, np.newaxis]
    div_term = np.power(
        10000,
        (2 * (np.arange(dm) // 2)) / dm
    )

    angles = position / div_term

    PE[:, 0::2] = np.sin(angles[:, 0::2])
    PE[:, 1::2] = np.cos(angles[:, 1::2])

    return PE
