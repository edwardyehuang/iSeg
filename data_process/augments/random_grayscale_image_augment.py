# ================================================================
# MIT License
# Copyright (c) 2026 edwardyehuang (https://github.com/edwardyehuang)
# ================================================================

import tensorflow as tf

from iseg.data_process.augments.data_augment_base import DataAugmentationBase, random_execute_helper


class RandomGrayscaleImageAugment(DataAugmentationBase):
    def __init__(self, execute_prob=0.2, name=None):
        super().__init__(name=name)

        self.execute_prob = execute_prob

    def call(self, image, label):

        image = random_execute_helper(
            self.execute_prob,
            lambda: tf.image.grayscale_to_rgb(tf.image.rgb_to_grayscale(image)),
            lambda: image,
        )

        return image, label
