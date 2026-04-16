# ================================================================
# MIT License
# Copyright (c) 2026 edwardyehuang (https://github.com/edwardyehuang)
# ================================================================

import tensorflow as tf

from iseg.data_process.augments.data_augment_base import DataAugmentationBase


class RandomFlipImageAugment(DataAugmentationBase):
    def __init__(self, prob_of_flip=0.5, name=None):
        super().__init__(name=name)

        self.prob_of_flip = prob_of_flip

    def call(self, image, label):
        random_value = tf.random.uniform([])

        image = tf.cond(
            random_value <= self.prob_of_flip,
            lambda: tf.image.flip_left_right(image),
            lambda: image,
        )

        return image, label
