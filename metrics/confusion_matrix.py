# ================================================================
# MIT License
# Copyright (c) 2021 edwardyehuang (https://github.com/edwardyehuang)
# ================================================================

# This code is motified from offical TensorFlow repo, may be deleted in future (since a bug is fixed)

import tensorflow as tf
from tensorflow.python.framework import ops
from tensorflow.python.ops import control_flow_ops


def remove_squeezable_dimensions(labels, predictions, expected_rank_diff=0, name=None):
    """Squeeze last dim if ranks differ from expected by exactly 1.
    In the common case where we expect shapes to match, `expected_rank_diff`
    defaults to 0, and we squeeze the last dimension of the larger rank if they
    differ by 1.
    But, for example, if `labels` contains class IDs and `predictions` contains 1
    probability per class, we expect `predictions` to have 1 more dimension than
    `labels`, so `expected_rank_diff` would be 1. In this case, we'd squeeze
    `labels` if `rank(predictions) - rank(labels) == 0`, and
    `predictions` if `rank(predictions) - rank(labels) == 2`.
    This will use static shape if available. Otherwise, it will add graph
    operations, which could result in a performance hit.
    Args:
        labels: Label values, a `Tensor` whose dimensions match `predictions`.
        predictions: Predicted values, a `Tensor` of arbitrary dimensions.
        expected_rank_diff: Expected result of `rank(predictions) - rank(labels)`.
        name: Name of the op.
    Returns:
        Tuple of `labels` and `predictions`, possibly with last dim squeezed.
    """
    with ops.name_scope(name, "remove_squeezable_dimensions", [labels, predictions]):
        predictions = tf.convert_to_tensor(predictions)
        labels = tf.convert_to_tensor(labels)
        predictions_shape = predictions.get_shape()
        predictions_rank = predictions_shape.ndims
        labels_shape = labels.get_shape()
        labels_rank = labels_shape.ndims
        if (labels_rank is not None) and (predictions_rank is not None):
            # Use static rank.
            rank_diff = predictions_rank - labels_rank
            if rank_diff == expected_rank_diff + 1 and predictions_shape.dims[-1].is_compatible_with(1):
                predictions = tf.squeeze(predictions, [-1])
            elif rank_diff == expected_rank_diff - 1 and labels_shape.dims[-1].is_compatible_with(1):
                labels = tf.squeeze(labels, [-1])

            return labels, predictions

        # Use dynamic rank.
        rank_diff = tf.rank(predictions) - tf.rank(labels)
        if (predictions_rank is None) or (predictions_shape.dims[-1].is_compatible_with(1)):
            predictions = tf.cond(
                tf.equal(expected_rank_diff + 1, rank_diff), lambda: tf.squeeze(predictions, [-1]), lambda: predictions
            )

        if (labels_rank is None) or (labels_shape.dims[-1].is_compatible_with(1)):
            labels = tf.cond(
                tf.equal(expected_rank_diff - 1, rank_diff), lambda: tf.squeeze(labels, [-1]), lambda: labels
            )

        return labels, predictions


def confusion_matrix(labels, predictions, num_classes=None, weights=None, dtype=tf.int32, name=None):
    """Computes the confusion matrix from predictions and labels.
    The matrix columns represent the prediction labels and the rows represent the
    real labels. The confusion matrix is always a 2-D array of shape `[n, n]`,
    where `n` is the number of valid labels for a given classification task. Both
    prediction and labels must be 1-D arrays of the same shape in order for this
    function to work.
    If `num_classes` is `None`, then `num_classes` will be set to one plus the
    maximum value in either predictions or labels. Class labels are expected to
    start at 0. For example, if `num_classes` is 3, then the possible labels
    would be `[0, 1, 2]`.
    If `weights` is not `None`, then each prediction contributes its
    corresponding weight to the total value of the confusion matrix cell.
    For example:
    ```python
        tf.math.confusion_matrix([1, 2, 4], [2, 2, 4]) ==>
            [[0 0 0 0 0]
            [0 0 1 0 0]
            [0 0 1 0 0]
            [0 0 0 0 0]
            [0 0 0 0 1]]
    ```
    Note that the possible labels are assumed to be `[0, 1, 2, 3, 4]`,
    resulting in a 5x5 confusion matrix.
    Args:
        labels: 1-D `Tensor` of real labels for the classification task.
        predictions: 1-D `Tensor` of predictions for a given classification.
        num_classes: The possible number of labels the classification task can
                    have. If this value is not provided, it will be calculated
                    using both predictions and labels array.
        weights: An optional `Tensor` whose shape matches `predictions`.
        dtype: Data type of the confusion matrix.
        name: Scope name.
    Returns:
        A `Tensor` of type `dtype` with shape `[n, n]` representing the confusion
        matrix, where `n` is the number of possible labels in the classification
        task.
    Raises:
        ValueError: If both predictions and labels are not 1-D vectors and have
        mismatched shapes, or if `weights` is not `None` and its shape doesn't
        match `predictions`.
    """
    with ops.name_scope(name, "confusion_matrix", (predictions, labels, num_classes, weights)) as name:
        labels, predictions = remove_squeezable_dimensions(
            tf.convert_to_tensor(labels, name="labels"), tf.convert_to_tensor(predictions, name="predictions")
        )
    predictions = tf.cast(predictions, tf.int32)
    labels = tf.cast(labels, tf.int32)

    # Sanity checks - underflow or overflow can cause memory corruption.
    labels = control_flow_ops.with_dependencies(
        [tf.debugging.assert_non_negative(labels, message="`labels` contains negative values")], labels
    )
    predictions = control_flow_ops.with_dependencies(
        [tf.debugging.assert_non_negative(predictions, message="`predictions` contains negative values")], predictions
    )

    if num_classes is None:
        num_classes = tf.maximum(tf.reduce_max(predictions), tf.reduce_max(labels)) + 1
    else:
        num_classes_int32 = tf.cast(num_classes, tf.int32)
        labels = control_flow_ops.with_dependencies(
            [tf.debugging.assert_less(labels, num_classes_int32, message="`labels` out of bound")], labels
        )
        predictions = control_flow_ops.with_dependencies(
            [tf.debugging.assert_less(predictions, num_classes_int32, message="`predictions` out of bound")],
            predictions,
        )

    if weights is not None:
        weights = tf.convert_to_tensor(weights, name="weights")
        predictions.get_shape().assert_is_compatible_with(weights.get_shape())
        weights = tf.cast(weights, dtype)

    shape = tf.stack([num_classes, num_classes])
    indices = tf.stack([labels, predictions], axis=1)
    values = tf.ones_like(predictions, dtype) if weights is None else weights

    return tf.scatter_nd(indices=indices, updates=values, shape=tf.cast(shape, tf.int32))
