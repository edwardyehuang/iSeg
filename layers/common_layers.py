# ================================================================
# MIT License
# Copyright (c) 2021 edwardyehuang (https://github.com/edwardyehuang)
# ================================================================

import tensorflow as tf


def extract_spatial_patches(x, size=4, use_mean_padding_value=False, padding_direction=0, inverse_slice=False):

    inputs_shape = tf.shape(x)
    batch_size = inputs_shape[0]
    height = inputs_shape[1]
    width = inputs_shape[2]
    channels = x.shape[-1]

    if isinstance(size, tuple):
        size = list(size)

    if not isinstance(size, list):
        size = [size, size]

    patch_size_h = size[0]
    patch_size_w = size[1]

    num_row = height // patch_size_h
    num_col = width // patch_size_w

    r_h = height % patch_size_h
    r_w = width % patch_size_w

    pad_h = tf.where(r_h == 0, 0, patch_size_h - r_h)
    pad_w = tf.where(r_w == 0, 0, patch_size_w - r_w)

    num_row = tf.where(pad_h > 0, num_row + 1, num_row)
    num_col = tf.where(pad_w > 0, num_col + 1, num_col)

    padding_value = 0

    if use_mean_padding_value:
        raise NotImplementedError()

    possible_paddings_arr = [
        [[0, 0], [0, pad_h], [0, pad_w], [0, 0]],
        [[0, 0], [pad_h, 0], [0, pad_w], [0, 0]],
        [[0, 0], [pad_h, 0], [pad_w, 0], [0, 0]],
        [[0, 0], [0, pad_h], [pad_w, 0], [0, 0]],
    ]

    x = tf.pad(x, paddings=possible_paddings_arr[padding_direction], constant_values=padding_value)

    if not inverse_slice:
        x = tf.reshape(x, [batch_size, num_row, patch_size_h, num_col, patch_size_w, channels])
    else:
        x = tf.reshape(x, [batch_size, patch_size_h, num_row, patch_size_w, num_col, channels])

    return x, pad_h, pad_w
