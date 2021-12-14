# ================================================================
# MIT License
# Copyright (c) 2021 edwardyehuang (https://github.com/edwardyehuang)
# ================================================================

import tensorflow as tf

from tensorflow.python.keras import initializers
from tensorflow.python.keras import regularizers


class DenseExt(tf.keras.layers.Layer):
    def __init__(
        self,
        units,
        use_bias=False,
        kernel_initializer="glorot_uniform",
        kernel_regularizer=None,
        bias_initializer="zeros",
        bias_regularizer=None,
        name=None,
    ):

        super(DenseExt, self).__init__(name="DenseExt" if name is None else name)

        self.units = units
        self.kernel_initializer = initializers.get(kernel_initializer)
        self.kernel_regularizer = regularizers.get(kernel_regularizer)

        self.use_bias = use_bias

        if self.use_bias:
            self.bias_initializer = bias_initializer
            self.bias_regularizer = bias_regularizer

    def build(self, input_shape):

        self.kernel = self.add_weight(
            "kernel",
            shape=[int(input_shape[-1]), self.units],
            initializer=self.kernel_initializer,
            regularizer=self.kernel_regularizer,
            dtype=self.dtype,
            trainable=True,
        )

        if self.use_bias:

            self.bias = self.add_weight(
                "bias",
                shape=[
                    self.units,
                ],
                initializer=self.bias_initializer,
                regularizer=self.bias_regularizer,
                dtype=self.dtype,
                trainable=True,
            )

    def call(self, inputs):

        # tf.assert_equal(inputs.shape[-1], self.kernel.shape[0], message = "inputs channel not equal to kenrel")

        x = tf.expand_dims(inputs, axis=-1)  # [N, In_C] => [N, In_C, 1]
        w = tf.expand_dims(self.kernel, axis=0)  # [In_C, Out_C] => [1, In_C, Out_C]

        x = tf.multiply(x, w)  # [N, InC, Out_C]
        x = tf.reduce_sum(x, axis=1)  # [N, Out_C]

        if self.use_bias:
            x += self.bias

        return x

        """
        # Caused error when eval on Pascal Context. Not sure why

        return tf.matmul(inputs, self.kernel, name = "dense_matmul")

        """
