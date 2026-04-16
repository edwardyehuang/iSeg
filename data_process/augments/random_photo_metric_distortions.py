# ================================================================
# MIT License
# Copyright (c) 2021 edwardyehuang (https://github.com/edwardyehuang)
# ================================================================

import tensorflow as tf

from iseg.data_process.augments.data_augment_base import DataAugmentationBase, random_execute_helper
from iseg.data_process.augments.random_brightness_augment import RandomBrightnessAugment
from iseg.data_process.augments.random_contrast_augment import RandomContrastAugment
from iseg.data_process.augments.random_saturation_augment import RandomSaturationAugment
from iseg.data_process.augments.random_hue_augment import RandomHueAugment


class RandomPhotoMetricDistortions(DataAugmentationBase):
    def __init__(
        self,
        brightness_max_delta=32,
        brightness_prob=0.5,
        contrast_lower=0.75,
        contrast_upper=1.25,
        contrast_prob=0.5,
        saturation_lower=0.75,
        saturation_upper=1.25,
        saturation_prob=0.5,
        hue_max_delta=0.1,
        hue_prob=1.0,
        include_brightness=False,
        random_order=False,
        execute_prob=1.0,
        name=None,
    ):

        super().__init__(name=name)

        self.include_brightness = include_brightness
        self.random_order = random_order
        self.execute_prob = execute_prob

        self.random_brightness = RandomBrightnessAugment(brightness_max_delta, execute_prob=brightness_prob)
        self.random_contrast = RandomContrastAugment(contrast_lower, contrast_upper, execute_prob=contrast_prob)
        self.random_saturation = RandomSaturationAugment(saturation_lower, saturation_upper, execute_prob=saturation_prob)
        self.random_hue = RandomHueAugment(hue_max_delta, execute_prob=hue_prob)

    def call(self, image, label):

        return random_execute_helper(
            self.execute_prob,
            lambda: self._execute_branch(image, label),
            lambda: (image, label),
        )


    def _execute_branch(self, image, label):

        if self.random_order:
            image, label = tf.cond(
                tf.random.uniform([]) <= 0.5,
                lambda: self.contrast_first_forward(image, label),
                lambda: self.contrast_last_forward(image, label),
            )
        else:
            image, label = self.contrast_first_forward(image, label)

        image = tf.clip_by_value(image, clip_value_min=0.0, clip_value_max=256.0)

        return image, label
    

    def contrast_first_forward (self, image, label):

        if self.include_brightness:
            image, label = self.random_brightness(image, label)

        image, label = self.random_contrast(image, label)
        image, label = self.random_saturation(image, label)
        image, label = self.random_hue(image, label)

        return image, label
    

    def contrast_last_forward (self, image, label):

        if self.include_brightness:
            image, label = self.random_brightness(image, label)

        image, label = self.random_saturation(image, label)
        image, label = self.random_hue(image, label)
        image, label = self.random_contrast(image, label)

        return image, label
