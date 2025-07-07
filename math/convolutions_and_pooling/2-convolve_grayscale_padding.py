#!/usr/bin/env python3
""" Convolution with padding
"""
import numpy as np


def convolve_grayscale_padding(images, kernel, padding):
    """ Convolve grayscale images with custom padding

    Args:
        images (_type_): _description_
        kernel (_type_): _description_
        padding (_type_): _description_

    Returns:
        _type_: _description_
    """
    kh, kw = kernel.shape
    m, hm, wm = images.shape
    ph, pw = padding
    padded = np.pad(images, ((0, 0), (ph, ph), (pw, pw)), 'constant')
    ch = hm + (2 * ph) - kh + 1
    cw = wm + (2 * pw) - kw + 1
    convoluted = np.zeros((m, ch, cw))
    for h in range(ch):
        for w in range(cw):
            square = padded[:, h: h + kh, w: w + kw]
            insert = np.sum(square * kernel, axis=1).sum(axis=1)
            convoluted[:, h, w] = insert
    return convoluted
