# ================================================================
# MIT License
# Copyright (c) 2021 edwardyehuang (https://github.com/edwardyehuang)
# ================================================================

import tensorflow as tf
import iseg.data_process.utils as dataprocess

from iseg.data_process.arguments.data_argument_base import DataArgumentBase

class ResizeArgument(DataArgumentBase):

    def __init__(self, min_resize_value, max_resize_value, name = None):
        super().__init__(name = name)

        self.min_resize_value = min_resize_value
        self.min_resize_value = max_resize_value


    def call(self, image, label):
        
        return dataprocess.resize_to_range(image, label, min_size = self.min_resize_value, max_size = self.max_resize_value)