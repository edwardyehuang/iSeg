# ================================================================
# MIT License
# Copyright (c) 2021 edwardyehuang (https://github.com/edwardyehuang)
# ================================================================

import tensorflow as tf
import iseg.data_process.utils as dataprocess

from iseg.data_process.augments.data_augment_base import DataAugmentationBase


class PadAugment(DataAugmentationBase):
    def __init__(
        self, target_height, target_width, image_pad_value=[127.5, 127.5, 127.5], label_pad_value=255, name=None
    ):

        super().__init__(name=name)

        self.target_height = target_height
        self.target_width = target_width

        self.image_pad_value = image_pad_value
        self.label_pad_value = label_pad_value

    def call(self, image, label):

        image_shape = tf.shape(image)
        image_height = image_shape[0]
        image_width = image_shape[1]

        target_height = image_height + tf.maximum(self.target_height - image_height, 0)
        target_width = image_width + tf.maximum(self.target_width - image_width, 0)

        image = dataprocess.pad_to_bounding_box(
            image, 0, 0, target_height, target_width, pad_value=self.image_pad_value
        )

        if label is not None:

            label = dataprocess.pad_to_bounding_box(
                label, 0, 0, target_height, target_width, pad_value=self.label_pad_value
            )

        return image, label
