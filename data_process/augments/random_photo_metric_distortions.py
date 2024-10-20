# ================================================================
# MIT License
# Copyright (c) 2021 edwardyehuang (https://github.com/edwardyehuang)
# ================================================================

import tensorflow as tf

from iseg.data_process.augments.data_augment_base import DataAugmentationBase, random_execute_helper
from iseg.data_process.augments.random_contrast_augment import RandomContrastAugment
from iseg.data_process.augments.random_saturation_augment import RandomSaturationAugment
from iseg.data_process.augments.random_hue_augment import RandomHueAugment


class RandomPhotoMetricDistortions(DataAugmentationBase):
    def __init__(self, name=None):

        super().__init__(name=name)

        self.random_contrast = RandomContrastAugment(0.75, 1.25, execute_prob=0.5)
        self.random_saturation = RandomSaturationAugment(0.75, 1.25, execute_prob=0.5)
        self.random_hue = RandomHueAugment(0.1, execute_prob=1.0)

    def call(self, image, label):

        image, label = self.contrast_first_forward(image, label)
        image = tf.clip_by_value(image, clip_value_min=0.0, clip_value_max=256.0)

        return image, label
    

    def contrast_first_forward (self, image, label):

        image, label = self.random_contrast(image, label)
        image, label = self.random_saturation(image, label)
        image, label = self.random_hue(image, label)

        return image, label
    

    def contrast_last_forward (self, image, label):

        image, label = self.random_saturation(image, label)
        image, label = self.random_hue(image, label)
        image, label = self.random_contrast(image, label)

        return image, label
