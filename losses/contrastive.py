# ================================================================
# MIT License
# Copyright (c) 2021 edwardyehuang (https://github.com/edwardyehuang)
# ================================================================

import tensorflow as tf


def contrastive_loss(y_true, y_pred, margin=1.0):

    y_pred = tf.convert_to_tensor(y_pred)
    y_true = tf.dtypes.cast(y_true, y_pred.dtype)

    return y_true * tf.math.square(y_pred) + (1.0 - y_true) * tf.math.square(tf.math.maximum(margin - y_pred, 0.0))
