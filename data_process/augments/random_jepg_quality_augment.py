# ================================================================
# MIT License
# Copyright (c) 2025 edwardyehuang (https://github.com/edwardyehuang)
# ================================================================

import tensorflow as tf

from iseg.data_process.augments.data_augment_base import DataAugmentationBase, random_execute_helper


class RandomJEPGQualityAugment(DataAugmentationBase):
    def __init__(self, name=None):

        super().__init__(name=name)


    def call(self, image, label):
        
        image = random_execute_helper(
            self.execute_prob, lambda: tf.image.random_jpeg_quality(
                image, min_jpeg_quality=10, max_jpeg_quality=100
            ), lambda: image
        )

        image = tf.clip_by_value(image, clip_value_min=0.0, clip_value_max=256.0)

        return image, label