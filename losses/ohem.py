# ================================================================
# MIT License
# Copyright (c) 2021 edwardyehuang (https://github.com/edwardyehuang)
# ================================================================

# WIP DO NOT USE

import tensorflow as tf


def ohem_selector(loss, y_true, y_pred, batch_size=4, thresh=None, min_kept=100000):

    # losses [NHW]
    min_kept = tf.constant(min_kept, dtype=tf.int32)
    batch_min_kept = tf.multiply(min_kept, batch_size)

    if thresh is not None:
        seg_prob = tf.nn.softmax(y_pred)
        seg_prob = tf.math.multiply(seg_prob, tf.cast(y_true, dtype=seg_prob.dtype))
        seg_prob = tf.math.reduce_max(seg_prob, axis=-1)

        non_zeros_count = tf.math.count_nonzero(seg_prob, dtype=batch_min_kept.dtype)

        if non_zeros_count > 0:
            batch_min_kept = tf.math.minimum(batch_min_kept, non_zeros_count - 1)
            tf.assert_less(batch_min_kept, tf.shape(seg_prob)[0], "batch_min_kept must less than num of fatten loss")
            min_threshold = tf.sort(seg_prob, direction="DESCENDING")[batch_min_kept]
        else:
            min_threshold = tf.constant(0, dtype=seg_prob.dtype)

        threshold = tf.maximum(min_threshold, thresh)

        loss = tf.where(seg_prob < threshold, loss, 0)
    else:
        sorted_loss = tf.sort(loss, direction="DESCENDING")
        threshold = sorted_loss[batch_min_kept]
        loss = tf.where(loss > threshold, loss, 0)

    return loss


def get_ohem_fn(thresh=None, min_kept=100000):
    def wrapper_fn(y_true, y_pred, loss, batch_size):
        return ohem_selector(loss, y_true, y_pred, batch_size, thresh, min_kept)

    return wrapper_fn
