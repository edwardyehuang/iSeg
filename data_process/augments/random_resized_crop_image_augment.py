# ================================================================
# MIT License
# Copyright (c) 2026 edwardyehuang (https://github.com/edwardyehuang)
# ================================================================

import tensorflow as tf

from iseg.data_process.augments.data_augment_base import DataAugmentationBase


class RandomResizedCropImageAugment(DataAugmentationBase):
    def __init__(
        self,
        target_height,
        target_width,
        area_range=(0.08, 1.0),
        aspect_ratio_range=(3.0 / 4.0, 4.0 / 3.0),
        max_attempts=10,
        interpolation=tf.image.ResizeMethod.BILINEAR,
        name=None,
    ):
        super().__init__(name=name)

        self.target_height = target_height
        self.target_width = target_width
        self.area_range = (float(area_range[0]), float(area_range[1]))
        self.aspect_ratio_range = (float(aspect_ratio_range[0]), float(aspect_ratio_range[1]))
        self.max_attempts = int(max_attempts)
        self.interpolation = interpolation

    def call(self, image, label):
        image_size = tf.shape(image)

        # Sample a random crop region over the whole image, then resize.
        begin, size, _ = tf.image.sample_distorted_bounding_box(
            image_size=image_size,
            bounding_boxes=tf.constant([[[0.0, 0.0, 1.0, 1.0]]], dtype=tf.float32),
            min_object_covered=0.0,
            aspect_ratio_range=self.aspect_ratio_range,
            area_range=self.area_range,
            max_attempts=self.max_attempts,
            use_image_if_no_bounding_boxes=True,
        )

        image = tf.slice(image, begin, size)
        image = tf.image.resize(image, [self.target_height, self.target_width], method=self.interpolation)
        image.set_shape([self.target_height, self.target_width, 3])

        return image, label
