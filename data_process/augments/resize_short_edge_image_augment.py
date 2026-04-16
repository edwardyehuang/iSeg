# ================================================================
# MIT License
# Copyright (c) 2026 edwardyehuang (https://github.com/edwardyehuang)
# ================================================================

import tensorflow as tf

from iseg.data_process.augments.data_augment_base import DataAugmentationBase


class ResizeShortEdgeImageAugment(DataAugmentationBase):
    def __init__(
        self,
        short_edge,
        max_long_edge=None,
        interpolation=tf.image.ResizeMethod.BILINEAR,
        name=None,
    ):
        super().__init__(name=name)

        self.short_edge = int(short_edge)
        self.max_long_edge = int(max_long_edge) if max_long_edge is not None else None
        self.interpolation = interpolation

    def call(self, image, label):
        image_shape = tf.shape(image)

        height = image_shape[0]
        width = image_shape[1]

        short_side = tf.minimum(height, width)
        scale = tf.cast(self.short_edge, tf.float32) / tf.cast(short_side, tf.float32)

        new_height = tf.cast(tf.round(tf.cast(height, tf.float32) * scale), tf.int32)
        new_width = tf.cast(tf.round(tf.cast(width, tf.float32) * scale), tf.int32)

        if self.max_long_edge is not None:
            max_long_edge = tf.constant(self.max_long_edge, dtype=tf.int32)
            new_long_edge = tf.maximum(new_height, new_width)

            def _clamp_long_edge():
                clamp_scale = tf.cast(max_long_edge, tf.float32) / tf.cast(new_long_edge, tf.float32)
                clamped_height = tf.cast(tf.round(tf.cast(new_height, tf.float32) * clamp_scale), tf.int32)
                clamped_width = tf.cast(tf.round(tf.cast(new_width, tf.float32) * clamp_scale), tf.int32)

                return clamped_height, clamped_width

            new_height, new_width = tf.cond(
                new_long_edge > max_long_edge,
                _clamp_long_edge,
                lambda: (new_height, new_width),
            )

        image = tf.image.resize(image, [new_height, new_width], method=self.interpolation)
        image.set_shape([None, None, 3])

        return image, label
