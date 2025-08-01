# ================================================================
# MIT License
# Copyright (c) 2021 edwardyehuang (https://github.com/edwardyehuang)
# ================================================================

# This code is motified from offical TensorFlow repo, may be deleted in future (since a bug is fixed)

import tensorflow as tf
import keras
import numpy as np

from iseg.metrics.confusion_matrix import confusion_matrix
from iseg.utils.version_utils import is_keras3


def get_class_confusion_matrix (y_true, y_pred, num_class=21, sample_weight=None, dtype=tf.float32):

    y_true = tf.cast(y_true, dtype)
    y_pred = tf.cast(y_pred, dtype)

    # Flatten the input if its rank > 1.
    if y_pred.shape.ndims > 1:
        y_pred = tf.reshape(y_pred, [-1])

    if y_true.shape.ndims > 1:
        y_true = tf.reshape(y_true, [-1])

    if sample_weight is not None:
        sample_weight = tf.cast(sample_weight, dtype)
        if sample_weight.shape.ndims > 1:
            sample_weight = tf.reshape(sample_weight, [-1])

    # Accumulate the prediction to current confusion matrix.
    return confusion_matrix(y_true, y_pred, num_class, weights=sample_weight, dtype=dtype)



def get_per_class_miou (cm, dtype=tf.float32, row_axis=-2, col_axis=-1):

    """Compute the mean intersection-over-union via the confusion matrix."""
    sum_over_row = tf.cast(tf.reduce_sum(cm, axis=row_axis), dtype=dtype) # [..., num_classes]
    sum_over_col = tf.cast(tf.reduce_sum(cm, axis=col_axis), dtype=dtype) # [..., num_classes]
    true_positives = tf.cast(tf.linalg.diag_part(cm), dtype=dtype) # [..., num_classes]

    # sum_over_row + sum_over_col =
    #     2 * true_positives + false_positives + false_negatives.
    denominator = sum_over_row + sum_over_col - true_positives # [..., num_classes]

    num_valid_entries = tf.reduce_sum(
        tf.cast(tf.not_equal(denominator, 0), dtype=dtype), 
        axis=-1,
    ) # [...]

    iou = tf.math.divide_no_nan(true_positives, denominator)  # [..., num_classes]

    return iou, num_valid_entries


def per_class_miou_to_mean_miou(iou, num_valid_entries):

    return tf.math.divide_no_nan(tf.reduce_sum(iou, name="mean_iou"), num_valid_entries)



class MeanIOU(keras.metrics.Metric):
    def __init__(self, num_classes, name=None, dtype=None):
        """Initializes `PerClassIoU`.

        Arguments:
        num_classes: The possible number of labels the prediction task can have.
            This value must be provided, since a confusion matrix of dimension =
            [num_classes, num_classes] will be allocated.
        name: (Optional) string name of the metric instance.
        dtype: (Optional) data type of the metric result.

        """

        if is_keras3() and dtype is None:
            dtype = tf.float32

        super(MeanIOU, self).__init__(name=name, dtype=dtype)
        self.num_classes = num_classes

        # Variable to accumulate the predictions in the confusion matrix.
        self.total_cm = self.add_weight(
            name="total_confusion_matrix", 
            shape=(num_classes, num_classes), 
            initializer=tf.compat.v1.zeros_initializer
        )

    def update_state(self, y_true, y_pred, sample_weight=None):
        """Accumulates the confusion matrix statistics.

        Args:
        y_true: The ground truth values.
        y_pred: The predicted values.
        sample_weight: Optional weighting of each example. Defaults to 1. Can be a
            `Tensor` whose rank is either 0, or the same rank as `y_true`, and must
            be broadcastable to `y_true`.

        Returns:
        IOU per class.
        """

        current_cm = get_class_confusion_matrix(
            y_true, y_pred, self.num_classes, sample_weight, dtype=self._dtype
        )

        return self.total_cm.assign_add(current_cm)

    @tf.autograph.experimental.do_not_convert
    def result(self):
        
        iou, num_valid_entries = self.per_class_result()

        iou = per_class_miou_to_mean_miou(iou, num_valid_entries)

        return iou
    

    def per_class_result (self):

        return get_per_class_miou(self.total_cm, dtype=self._dtype)


    def reset_states(self):
        tf.keras.backend.set_value(self.total_cm, np.zeros((self.num_classes, self.num_classes)))

    def get_config(self):
        config = {"num_classes": self.num_classes}
        base_config = super(MeanIOU, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))
