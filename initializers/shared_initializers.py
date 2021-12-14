# ================================================================
# MIT License
# Copyright (c) 2021 edwardyehuang (https://github.com/edwardyehuang)
# ================================================================

import tensorflow as tf


class SharedInitializer(tf.keras.initializers.Initializer):
    def __init__(self, initializer, transpose_perm=None):

        self.__initializer = initializer
        self.__value = None

        self.__transpose_perm = transpose_perm

    def __call__(self, shape, dtype=None, **kwargs):

        if self.__value is None:
            self.__value = self.__initializer(shape, dtype, **kwargs)

        y = self.__value

        if self.__transpose_perm:
            y = tf.transpose(y, self.__transpose_perm)

        return y

    def transpose(self, perm=None):

        return SharedInitializer(self, transpose_perm=perm)
