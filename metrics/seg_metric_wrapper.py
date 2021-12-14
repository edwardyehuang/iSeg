# ================================================================
# MIT License
# Copyright (c) 2021 edwardyehuang (https://github.com/edwardyehuang)
# ================================================================

import tensorflow as tf


class SegMetricWrapper(tf.keras.metrics.Metric):
    def __init__(self, metric, num_class=21, ignore_label=255, name=None):
        super(SegMetricWrapper, self).__init__(name=name)

        self.metric = metric
        self.num_class = num_class
        self.ignore_label = ignore_label

        self.__pre_compute_fn_list = []

    def add_pre_compute_fn(self, fn):

        if fn is None:
            return

        self.__pre_compute_fn_list.append(fn)

    def update_state(self, y_true, y_pred, sample_weight=None):

        for pre_compute_fn in self.__pre_compute_fn_list:
            y_true, y_pred = pre_compute_fn(y_true, y_pred)

        y_true = tf.cast(y_true, tf.dtypes.int32)

        y_pred = tf.reshape(y_pred, shape=[-1, self.num_class])  # [NHW, C]
        y_pred = tf.argmax(y_pred, axis=-1)

        y_true = tf.reshape(y_true, shape=[-1])  # [NHW]

        not_ignore_mask = tf.math.not_equal(y_true, self.ignore_label)
        not_ignore_mask = tf.cast(not_ignore_mask, tf.dtypes.float32)

        if self.ignore_label == 0:
            y_true -= 1
            y_true = tf.where(tf.equal(y_true, -1), tf.zeros_like(y_true), y_true)
        else:
            y_true = tf.where(tf.equal(y_true, self.ignore_label), tf.zeros_like(y_true), y_true)

        self.metric.update_state(y_true, y_pred, not_ignore_mask)

    def result(self):

        return self.metric.result()

    def reset_states(self):

        self.metric.reset_states()
