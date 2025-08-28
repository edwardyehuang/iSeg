# ================================================================
# MIT License
# Copyright (c) 2025 edwardyehuang (https://github.com/edwardyehuang)
# ================================================================

import tensorflow as tf

from iseg.data_process.augments.data_augment_base import DataAugmentationBase, random_execute_helper


class RandomNoisyEvalAugment(DataAugmentationBase):
    def __init__(self, noise_level=0, name=None):

        super().__init__(name=name)
        self.noise_level = noise_level


    def call(self, image, label):
        
        if self.noise_level <= 0 + 1e-3:
            return image, label
        
        noise = tf.random.normal(shape=tf.shape(image), mean=0.0, stddev=self.noise_level, dtype=image.dtype)

        image = tf.add(image, noise)

        image = tf.clip_by_value(image, clip_value_min=0.0, clip_value_max=256.0)

        return image, label