# ================================================================
# MIT License
# Copyright (c) 2022 edwardyehuang (https://github.com/edwardyehuang)
# ================================================================

import tensorflow as tf

def drop_path(inputs, drop_prob, training):

    if (not training) or (drop_prob == 0.0):
        return inputs

    # Compute keep_prob
    keep_prob = 1.0 - drop_prob

    # Compute drop_connect tensor
    random_tensor = keep_prob
    shape = (tf.shape(inputs)[0],) + (1,) * (len(inputs.shape) - 1)
    random_tensor += tf.random.uniform(shape, dtype=inputs.dtype)
    binary_tensor = tf.floor(random_tensor)
    output = tf.math.divide(inputs, keep_prob) * binary_tensor

    return output