# ================================================================
# MIT License
# Copyright (c) 2026 edwardyehuang (https://github.com/edwardyehuang)
# ================================================================

import tensorflow as tf
import keras

from iseg.losses.seg_loss_base import SegLossBase
from iseg.utils.tensor_utils import get_stable_float_dtype_for_loss

class ClassificationSoftLabelCrossentropyLoss(SegLossBase):
    def __init__(
        self,
        num_class=1000,
        ignore_label=255,
        batch_size=2,
        reduction=False,
        class_weights=None,
        from_logits=True,
        label_smoothing=0.0,
        name=None,
        **kwargs,
    ):
        del kwargs

        super().__init__(
            num_class=num_class,
            ignore_label=ignore_label,
            batch_size=batch_size,
            reduction=reduction,
            from_logits=from_logits,
            class_weights=class_weights,
            name=name,
        )

        self.label_smoothing = float(label_smoothing)

        self.class_weights_tensor = None

        if self.class_weights is not None:
            class_weights_tensor = tf.cast(self.class_weights, tf.float32)
            self.class_weights_tensor = tf.reshape(class_weights_tensor, [1, -1])

    def internal_preprocess(self, y_true, y_pred):
        # Override SegLossBase preprocessing because classification labels are not spatial maps.
        float_dtype = get_stable_float_dtype_for_loss()

        y_pred = tf.cast(y_pred, float_dtype)
        y_pred = tf.reshape(y_pred, [-1, self.num_class])

        return y_true, y_pred, None

    @tf.autograph.experimental.do_not_convert
    def compute_loss_forwards(self, y_true, y_pred, valid_mask=None):
        del valid_mask

        y_true = self._to_soft_targets(y_true)

        if self.label_smoothing > 0 + 1e-6:
            smooth = self.label_smoothing / tf.cast(self.num_class, y_true.dtype)
            y_true = y_true * (1.0 - self.label_smoothing) + smooth

        loss = keras.losses.categorical_crossentropy(
            y_true,
            y_pred,
            from_logits=self.from_logits,
        )

        if self.class_weights_tensor is not None:
            sample_weights = tf.reduce_sum(y_true * self.class_weights_tensor, axis=-1)
            loss = loss * sample_weights

        return loss

    def _to_soft_targets(self, y_true):
        float_dtype = get_stable_float_dtype_for_loss()

        if y_true.dtype.is_floating:
            y_true = tf.cast(y_true, float_dtype)

            if y_true.shape.rank == 1:
                y_true = tf.cast(y_true, tf.int32)
                y_true = tf.one_hot(y_true, self.num_class, dtype=float_dtype)
            else:
                y_true = tf.reshape(y_true, [-1, self.num_class])

            return y_true

        y_true = tf.cast(y_true, tf.int32)

        if y_true.shape.rank is not None and y_true.shape.rank > 1 and y_true.shape[-1] == 1:
            y_true = tf.squeeze(y_true, axis=-1)

        y_true = tf.reshape(y_true, [-1])

        if self.ignore_label >= 0:
            valid_mask = tf.not_equal(y_true, self.ignore_label)
            y_true = tf.where(valid_mask, y_true, tf.zeros_like(y_true))

        return tf.one_hot(y_true, self.num_class, dtype=float_dtype)
