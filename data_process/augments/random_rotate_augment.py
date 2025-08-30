# ================================================================
# MIT License
# Copyright (c) 2021 edwardyehuang (https://github.com/edwardyehuang)
# ================================================================

import tensorflow as tf
import iseg.data_process.utils as dataprocess

from iseg.data_process.augments.data_augment_base import DataAugmentationBase


class RandomRotateAugment(DataAugmentationBase):
    def __init__(self, prob_of_rotate=0.5, name=None):

        super().__init__(name=name)

        self.prob_of_rotate = prob_of_rotate


    def call(self, image, label):

        random_value = tf.random.uniform([])

        return tf.cond(
            random_value <= self.prob_of_rotate,
            lambda: self._execute_branch(image, label),
            lambda: (image, label),
        )
    
    
    def _execute_branch(self, image, label):

        num_rotate = tf.random.uniform([], minval=1, maxval=3, dtype=tf.int32)

        image = tf.image.rot90(image, k=num_rotate)

        if label is not None:
            label = tf.image.rot90(label, k=num_rotate)

        return image, label
