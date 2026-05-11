# ================================================================
# MIT License
# Copyright (c) 2026 edwardyehuang (https://github.com/edwardyehuang)
# ================================================================

import tensorflow as tf
import keras


class ClassificationMetricWrapper(keras.metrics.Metric):
    def __init__(self, metric, num_class=1000, name=None):
        super().__init__(name=name)

        self.metric = metric
        self.num_class = num_class


    def update_state(self, y_true, y_pred, sample_weight=None):
        
        self._update_state_internal(y_true, y_pred, sample_weight=sample_weight)


    @tf.autograph.experimental.do_not_convert
    def _update_state_internal(self, y_true, y_pred, sample_weight=None):

        y_true, y_pred = self._prepare_inputs(y_true, y_pred)

        self.metric.update_state(y_true, y_pred, sample_weight=sample_weight)


    def result(self):
        return self.metric.result()

    def reset_states(self):
        if hasattr(self.metric, "reset_states"):
            self.metric.reset_states()
        else:
            self.metric.reset_state()

    def _prepare_inputs(self, y_true, y_pred):
        y_pred = tf.cast(y_pred, tf.float32)
        y_pred = tf.reshape(y_pred, [-1, self.num_class])

        if y_true.dtype.is_floating:
            y_true = tf.cast(y_true, tf.float32)

            if y_true.shape.rank == 1:
                y_true = tf.cast(y_true, tf.int32)
                y_true = tf.reshape(y_true, [-1])
            else:
                y_true = tf.reshape(y_true, [-1, self.num_class])
                y_true = tf.argmax(y_true, axis=-1, output_type=tf.int32)
        else:
            y_true = tf.cast(y_true, tf.int32)
            y_true = tf.reshape(y_true, [-1])

        return y_true, y_pred


def classification_top1_metric(num_class=1000, ignore_label=255, name=None):
    del ignore_label

    metric_name = "acc" if name is None or name == "" else f"{name}acc"

    metric = keras.metrics.SparseCategoricalAccuracy(name=metric_name)

    return ClassificationMetricWrapper(metric, num_class=num_class, name=metric_name)


def classification_topk_metric(num_class=1000, ignore_label=255, name=None, k=5):
    del ignore_label

    metric_name = f"top{k}" if name is None or name == "" else f"{name}top{k}"

    metric = keras.metrics.SparseTopKCategoricalAccuracy(k=k, name=metric_name)

    return ClassificationMetricWrapper(metric, num_class=num_class, name=metric_name)


def attach_classification_metrics(model, top_k=5):
    def custom_metrics(num_class, ignore_label):
        metrics = [classification_top1_metric(num_class=num_class, ignore_label=ignore_label, name="")]

        if top_k is not None and int(top_k) > 1:
            metrics.append(
                classification_topk_metric(
                    num_class=num_class,
                    ignore_label=ignore_label,
                    name="",
                    k=int(top_k),
                )
            )

        return {model._index_to_output_key(0): metrics}

    model.custom_metrics = custom_metrics

    return model
