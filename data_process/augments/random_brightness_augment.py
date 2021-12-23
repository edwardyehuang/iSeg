# ================================================================
# MIT License
# Copyright (c) 2021 edwardyehuang (https://github.com/edwardyehuang)
# ================================================================

import tensorflow as tf
import iseg.data_process.utils as dataprocess

from iseg.data_process.augments.data_augment_base import DataAugmentationBase, random_execute_helper


class RandomBrightnessAugment(DataAugmentationBase):
    def __init__(self, max_delta=32, execute_prob=0.5, name=None):

        super().__init__(name=name)

        self.max_delta = max_delta
        self.execute_prob = execute_prob

    def call(self, image, label):

        image = random_execute_helper(
            self.execute_prob, lambda: tf.image.random_brightness(image, self.max_delta), lambda: image
        )

        image = tf.clip_by_value(image, clip_value_min=0.0, clip_value_max=256.0)

        return image, label
