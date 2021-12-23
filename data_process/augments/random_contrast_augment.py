# ================================================================
# MIT License
# Copyright (c) 2021 edwardyehuang (https://github.com/edwardyehuang)
# ================================================================

import tensorflow as tf
import iseg.data_process.utils as dataprocess

from iseg.data_process.augments.data_augment_base import DataAugmentationBase, random_execute_helper


class RandomContrastAugment(DataAugmentationBase):
    def __init__(self, lower=0.5, upper=1.5, execute_prob=0.5, name=None):

        super().__init__(name=name)

        self.lower = lower
        self.upper = upper

        self.execute_prob = execute_prob

    def call(self, image, label):

        image = random_execute_helper(
            self.execute_prob, lambda: tf.image.random_contrast(image, self.lower, self.upper), lambda: image
        )

        return image, label
