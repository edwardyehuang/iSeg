# ================================================================
# MIT License
# Copyright (c) 2021 edwardyehuang (https://github.com/edwardyehuang)
# ================================================================

import tensorflow as tf
import iseg.data_process.utils as dataprocess

from iseg.data_process.augments.data_augment_base import DataAugmentationBase


def pad_to_odd(image, label=None, image_pad_value=[127.5, 127.5, 127.5], label_pad_value=255):

    image_shape = tf.shape(image)
    image_height = image_shape[0]
    image_width = image_shape[1]

    image_height += tf.cast(image_height % 2 == 0, tf.int32)
    image_width += tf.cast(image_width % 2 == 0, tf.int32)

    image = dataprocess.pad_to_bounding_box(image, 0, 0, image_height, image_width, pad_value=image_pad_value)

    if label is not None:
        label = dataprocess.pad_to_bounding_box(label, 0, 0, image_height, image_width, pad_value=label_pad_value)

    return image, label


class PadToOddAugment(DataAugmentationBase):
    def __init__(self, image_pad_value=[127.5, 127.5, 127.5], label_pad_value=255, name=None):

        super().__init__(name=name)

        self.image_pad_value = image_pad_value
        self.label_pad_value = label_pad_value

    def call(self, image, label):

        return pad_to_odd(
            image=image, label=label, image_pad_value=self.image_pad_value, label_pad_value=self.label_pad_value
        )
