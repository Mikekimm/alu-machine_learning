#!/usr/bin/env python3
"""Neural Style Transfer - initialization module"""

import numpy as np
import tensorflow as tf


class NST:
    """Performs Neural Style Transfer tasks"""

    style_layers = [
        'block1_conv1',
        'block2_conv1',
        'block3_conv1',
        'block4_conv1',
        'block5_conv1'
    ]

    content_layer = 'block5_conv2'

    def __init__(self, style_image, content_image, alpha=1e4, beta=1):
        """Class constructor"""

        if not isinstance(style_image, np.ndarray) or style_image.shape[-1] != 3:
            raise TypeError("style_image must be a numpy.ndarray with shape (h, w, 3)")

        if not isinstance(content_image, np.ndarray) or content_image.shape[-1] != 3:
            raise TypeError("content_image must be a numpy.ndarray with shape (h, w, 3)")

        if not isinstance(alpha, (int, float)) or alpha < 0:
            raise TypeError("alpha must be a non-negative number")

        if not isinstance(beta, (int, float)) or beta < 0:
            raise TypeError("beta must be a non-negative number")

        tf.enable_eager_execution()

        self.alpha = float(alpha)
        self.beta = float(beta)

        self.style_image = self.scale_image(style_image)
        self.content_image = self.scale_image(content_image)

    @staticmethod
    def scale_image(image):
        """Rescales image to max size 512 while keeping aspect ratio"""

        if not isinstance(image, np.ndarray) or image.shape[-1] != 3:
            raise TypeError("image must be a numpy.ndarray with shape (h, w, 3)")

        h, w, _ = image.shape

        max_dim = 512
        scale = max_dim / max(h, w)

        new_h = int(h * scale)
        new_w = int(w * scale)

        image = tf.image.resize(image, (new_h, new_w), method='bicubic')

        image = image / 255.0

        image = tf.expand_dims(image, axis=0)

        return image
