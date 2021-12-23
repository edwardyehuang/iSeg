# ================================================================
# MIT License
# Copyright (c) 2021 edwardyehuang (https://github.com/edwardyehuang)
# ================================================================

import tensorflow as tf
import iseg.data_process.utils as dataprocess

from iseg.data_process.augments.data_augment_base import DataAugmentationBase


class RandomCropAugment(DataAugmentationBase):
    def __init__(self, crop_height=513, crop_width=513, name=None):

        super().__init__(name=name)

        self.crop_height = crop_height
        self.crop_width = crop_width

    def call(self, image, label):

        if label is not None:
            image, label = dataprocess.random_crop([image, label], self.crop_height, self.crop_width)

        image.set_shape([self.crop_height, self.crop_width, 3])
        label.set_shape([self.crop_height, self.crop_width, label.shape[-1]])

        return image, label
