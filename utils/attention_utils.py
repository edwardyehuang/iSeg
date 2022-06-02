# ================================================================
# MIT License
# Copyright (c) 2021 edwardyehuang (https://github.com/edwardyehuang)
# ================================================================

import tensorflow as tf

import numpy as np

from iseg.layers.model_builder import get_tensor_shape


def flatten_hw(inputs):

    x = tf.reshape(inputs, (tf.shape(inputs)[0], tf.shape(inputs)[1] * tf.shape(inputs)[2], inputs.shape[-1]))

    return x


def transpose_hw_c(x):

    return tf.transpose(x, (0, 2, 1))


def get_attention(query, key, apply_scale=False, numeric_stable=False):

    if numeric_stable:
        x_dtype = query.dtype
        query = tf.cast(query, tf.float32)
        key = tf.cast(key, tf.float32)

    x = tf.matmul(query, key)

    if apply_scale:
        x /= tf.sqrt(tf.cast(query.shape[-1], x.dtype))

    x = tf.nn.softmax(x)

    if numeric_stable:
        x = tf.cast(x, x_dtype)

    return x


def compute_2d_self_attention(query, key, scale=False):

    query = flatten_hw(query)
    key = transpose_hw_c(flatten_hw(key))
    attention_map = tf.matmul(query, key)

    if scale:
        attention_map /= tf.sqrt(tf.cast(query.shape[-1], attention_map.dtype))

    attention_map = tf.nn.softmax(attention_map)

    return attention_map


def get_axial_attention(query, key, axis=1, apply_scale=False):

    if axis == 1:
        query = tf.transpose(query, [0, 2, 1, 3])  # [N, W, H, C]
        key = tf.transpose(key, [0, 2, 3, 1])  # [N, W, C, H]
    elif axis == 2:
        key = tf.transpose(key, [0, 1, 3, 2])  # [N, H, C, W]
    else:
        raise ValueError("axis must be 1 or 2")

    return get_attention(query, key, apply_scale=apply_scale)  # [N, W, H, H] or [N, H, W, W]
