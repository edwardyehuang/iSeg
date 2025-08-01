# ================================================================
# MIT License
# Copyright (c) 2021 edwardyehuang (https://github.com/edwardyehuang)
# ================================================================

import tensorflow as tf


def process_seg_metric_inputs(
    y_true, 
    y_pred,
    num_class=21,
    ignore_label=255,
    use_class_prob_as_pred=True,
):
    
    y_true = tf.cast(y_true, tf.dtypes.int32)

    # resize y_true to match y_pred shape
    if len(y_pred.shape) == 4:
        if len(y_true.shape) == 3:
            y_true = tf.expand_dims(y_true, axis=-1)

        y_true = tf.image.resize(y_true, size=tf.shape(y_pred)[1:3], method="nearest")


    if use_class_prob_as_pred:
        y_pred = tf.reshape(y_pred, shape=[-1, num_class])  # [NHW, C]
        y_pred = tf.argmax(y_pred, axis=-1)
    else:
        y_pred = tf.reshape(y_pred, shape=[-1]) # [NHW]

    y_true = tf.reshape(y_true, shape=[-1])  # [NHW]

    not_ignore_mask = tf.math.not_equal(y_true, ignore_label)
    not_ignore_mask = tf.cast(not_ignore_mask, tf.dtypes.float32)

    if ignore_label == 0:
        y_true -= 1
        y_true = tf.where(tf.equal(y_true, -1), tf.zeros_like(y_true), y_true)
    else:
        y_true = tf.where(tf.equal(y_true, ignore_label), tf.zeros_like(y_true), y_true)

    return y_true, y_pred, not_ignore_mask



class SegMetricWrapper(tf.keras.metrics.Metric):
    def __init__(self, metric, num_class=21, ignore_label=255, name=None):
        super().__init__(name=name)

        self.metric = metric
        self.num_class = num_class
        self.ignore_label = ignore_label

        self._pre_compute_fn_list = []

    def add_pre_compute_fn(self, fn):

        if fn is None:
            return

        self._pre_compute_fn_list.append(fn)

    def update_state(self, y_true, y_pred, sample_weight=None):

        for pre_compute_fn in self._pre_compute_fn_list:
            y_true, y_pred = pre_compute_fn(y_true, y_pred)


        y_true, y_pred, not_ignore_mask = process_seg_metric_inputs(
            y_true=y_true,
            y_pred=y_pred,
            num_class=self.num_class,
            ignore_label=self.ignore_label,
        )

        self.metric.update_state(y_true, y_pred, not_ignore_mask)

    def result(self):

        return self.metric.result()

    def reset_states(self):

        self.metric.reset_states()
