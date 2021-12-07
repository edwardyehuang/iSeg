# ================================================================
# MIT License
# Copyright (c) 2021 edwardyehuang (https://github.com/edwardyehuang)
# ================================================================

import tensorflow as tf
import iseg.data_process.utils as dataprocess

from iseg.data_process.arguments.data_argument_base import DataArgumentBase

class LambdaArgument(DataArgumentBase):

    def __init__(self, fn, name = None):
        super().__init__(name = name)

        self.fn = fn


    def call(self, *args):
        
        return self.fn(*args)