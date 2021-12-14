# ================================================================
# MIT License
# Copyright (c) 2021 edwardyehuang (https://github.com/edwardyehuang)
# ================================================================

import tensorflow as tf
import iseg.data_process.utils as dataprocess

from iseg.data_process.arguments.data_argument_base import DataArgumentBase


class RandomHueArgument(DataArgumentBase):
    def __init__(self, max_delta=0.1, name=None):

        super().__init__(name=name)

        self.max_delta = max_delta

    def call(self, image, label):

        image = tf.image.random_hue(image, self.max_delta)

        return image, label
