# ================================================================
# MIT License
# Copyright (c) 2021 edwardyehuang (https://github.com/edwardyehuang)
# ================================================================

import tensorflow as tf

from iseg.layers.common_layers import extract_spatial_patches

from tensorflow.python.keras.utils import conv_utils


def adaptive_average_pooling_2d(inputs, size, name=None):

    name = "adaptive_pooling_2d" if name is None else name

    with tf.name_scope(name=name):
        x, _, _ = extract_spatial_patches(inputs, size=size, inverse_slice=True)  # [N, row, ph, col, pw, channels]
        x = tf.reduce_mean(x, (2, 4))

        return x
