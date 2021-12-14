# ================================================================
# MIT License
# Copyright (c) 2021 edwardyehuang (https://github.com/edwardyehuang)
# ================================================================

import tensorflow as tf


def smooth_l1_loss(y_true, y_pred):

    y_true = tf.cast(y_true, tf.float32)
    y_pred = tf.cast(y_pred, tf.float32)

    diff = tf.math.abs(y_true - y_pred)

    less_than_one = tf.cast(diff < 1.0, tf.float32)

    loss = less_than_one * 0.5 * tf.square(diff) + (1.0 - less_than_one) * (diff - 0.5)

    return tf.reduce_mean(loss)
