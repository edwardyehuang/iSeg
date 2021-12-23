# ================================================================
# MIT License
# Copyright (c) 2021 edwardyehuang (https://github.com/edwardyehuang)
# ================================================================

import tensorflow as tf
import iseg.data_process.utils as dataprocess

from iseg.data_process.augments.data_augment_base import DataAugmentationBase


class RandomScaleAugment(DataAugmentationBase):
    def __init__(self, min_scale_factor=0.5, max_scale_factor=2.0, scale_factor_step_size=0.1, name=None):

        super().__init__(name=name)

        self.min_scale_factor = min_scale_factor
        self.max_scale_factor = max_scale_factor
        self.scale_factor_step_size = scale_factor_step_size

    def call(self, image, label):

        scale = dataprocess.get_random_scale(self.min_scale_factor, self.max_scale_factor, self.scale_factor_step_size)
        image, label = dataprocess.randomly_scale_image_and_label(image, label, scale)
        image.set_shape([None, None, 3])

        return image, label
