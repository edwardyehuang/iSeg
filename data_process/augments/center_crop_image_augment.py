# ================================================================
# MIT License
# Copyright (c) 2026 edwardyehuang (https://github.com/edwardyehuang)
# ================================================================

import tensorflow as tf

from iseg.data_process.augments.data_augment_base import DataAugmentationBase


class CenterCropImageAugment(DataAugmentationBase):
    def __init__(self, crop_height, crop_width, name=None):
        super().__init__(name=name)

        self.crop_height = crop_height
        self.crop_width = crop_width

    def call(self, image, label):
        image = tf.image.resize_with_crop_or_pad(image, self.crop_height, self.crop_width)
        image.set_shape([self.crop_height, self.crop_width, 3])

        return image, label
