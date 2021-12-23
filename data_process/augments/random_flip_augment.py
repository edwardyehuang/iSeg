# ================================================================
# MIT License
# Copyright (c) 2021 edwardyehuang (https://github.com/edwardyehuang)
# ================================================================

import tensorflow as tf
import iseg.data_process.utils as dataprocess

from iseg.data_process.augments.data_augment_base import DataAugmentationBase


class RandomFlipAugment(DataAugmentationBase):
    def __init__(self, prob_of_flip=0.5, name=None):

        super().__init__(name=name)

        self.prob_of_flip = prob_of_flip

    def call(self, image, label, reversed_label=None):

        random_value = tf.random.uniform([])

        if label is not None and random_value <= self.prob_of_flip:
            image = tf.image.flip_left_right(image)

            if reversed_label is not None:
                label = reversed_label
            else:
                label = tf.image.flip_left_right(label)

        return image, label
