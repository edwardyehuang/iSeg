# ================================================================
# MIT License
# Copyright (c) 2021 edwardyehuang (https://github.com/edwardyehuang)
# ================================================================

# This code is motified from https://github.com/keras-team/keras/blob/master/keras/applications/resnet.py
# The modifications are refer to https://github.com/tensorflow/models/blob/master/research/deeplab/core/resnet_v1_beta.py
# and "Bag of Tricks for Image Classification with Convolutional Neural Networks", CVPR2019

import tensorflow as tf

from iseg.layers.normalizations import normalization
from tensorflow.python.keras.utils import conv_utils

BN_EPSILON = 1.001e-5


def conv2d_same_fn(*args, **kwargs):

    return tf.keras.layers.Conv2D(*args, **kwargs)


class BlockType1(tf.keras.Model):
    def __init__(
        self, filters, kernel_size=3, stride=1, conv_shortcut=True, use_bias=True, norm_method=None, name=None
    ):

        super(BlockType1, self).__init__(name=name)

        self.conv_shortcut = conv_shortcut

        if self.conv_shortcut:
            self.shortcut_conv = tf.keras.layers.Conv2D(
                4 * filters, kernel_size=1, strides=stride, use_bias=use_bias, name=name + "_0_conv"
            )
            self.shortcut_bn = normalization(epsilon=BN_EPSILON, method=norm_method, name=name + "_0_bn")

        self.conv1_conv = tf.keras.layers.Conv2D(
            filters, kernel_size=1, strides=stride, use_bias=use_bias, name=name + "_1_conv"
        )
        self.conv1_bn = normalization(epsilon=BN_EPSILON, method=norm_method, name=name + "_1_bn")

        self.conv2_conv = conv2d_same_fn(filters, kernel_size, padding="SAME", use_bias=use_bias, name=name + "_2_conv")
        self.conv2_bn = normalization(epsilon=BN_EPSILON, method=norm_method, name=name + "_2_bn")

        self.conv3_conv = tf.keras.layers.Conv2D(4 * filters, kernel_size=1, use_bias=use_bias, name=name + "_3_conv")
        self.conv3_bn = normalization(epsilon=BN_EPSILON, method=norm_method, name=name + "_3_bn")

    @property
    def strides(self):
        return self.conv1_conv.strides[0]

    @strides.setter
    def strides(self, value):

        value = conv_utils.normalize_tuple(value, self.shortcut_conv.rank, "strides")

        self.conv1_conv.strides = value

        if self.conv_shortcut:
            self.shortcut_conv.strides = value

    @property
    def atrous_rates(self):
        return self.conv2_conv.dilation_rate[0]

    @atrous_rates.setter
    def atrous_rates(self, value):

        value = conv_utils.normalize_tuple(value, self.conv2_conv.rank, "dilation_rate")

        if self.conv2_conv.built:
            raise ValueError("conv has been built")

        self.conv2_conv.dilation_rate = value

    def call(self, inputs, training=None, **kwargs):

        if self.conv_shortcut:
            shortcut = self.shortcut_conv(inputs)
            shortcut = self.shortcut_bn(shortcut, training=training)
        else:
            shortcut = inputs

        x = self.conv1_conv(inputs)
        x = self.conv1_bn(x, training=training)
        x = tf.nn.relu(x)

        tf.assert_equal(x.shape.rank, 4)

        x = self.conv2_conv(x, training=training)
        x = self.conv2_bn(x, training=training)
        x = tf.nn.relu(x)

        x = self.conv3_conv(x)
        x = self.conv3_bn(x, training=training)

        x = tf.add(shortcut, x)
        x = tf.nn.relu(x)

        return x


class BlockType2(tf.keras.Model):
    def __init__(
        self,
        filters,
        kernel_size=3,
        stride=1,
        conv_shortcut=True,
        use_bias=False,
        norm_method=None,
        downsample_method="avg",
        name=None,
    ):

        super(BlockType2, self).__init__(name=name)

        self.conv_shortcut = conv_shortcut
        self.downsample_method = downsample_method

        if self.conv_shortcut:
            self.shortcut_conv = tf.keras.layers.Conv2D(
                4 * filters, kernel_size=1, strides=stride, use_bias=use_bias, name=name + "_0_conv"
            )
            self.shortcut_bn = normalization(epsilon=BN_EPSILON, method=norm_method, name=name + "_0_bn")

        self.conv1_conv = tf.keras.layers.Conv2D(filters, kernel_size=1, use_bias=use_bias, name=name + "_1_conv")
        self.conv1_bn = normalization(epsilon=BN_EPSILON, method=norm_method, name=name + "_1_bn")

        self.conv2_conv = conv2d_same_fn(
            filters, kernel_size, strides=stride, padding="SAME", use_bias=use_bias, name=name + "_2_conv"
        )
        self.conv2_bn = normalization(epsilon=BN_EPSILON, method=norm_method, name=name + "_2_bn")

        self.conv3_conv = tf.keras.layers.Conv2D(4 * filters, kernel_size=1, use_bias=use_bias, name=name + "_3_conv")
        self.conv3_bn = normalization(epsilon=BN_EPSILON, method=norm_method, name=name + "_3_bn")

    @property
    def strides(self):
        return self.conv2_conv.strides[0]

    @strides.setter
    def strides(self, value):

        if not isinstance(value, tuple):
            value = (value, value)

        self.conv2_conv.strides = value

        if self.conv_shortcut:
            self.shortcut_conv.strides = value

    @property
    def atrous_rates(self):
        return self.conv2_conv.dilation_rate[0]

    @atrous_rates.setter
    def atrous_rates(self, value):

        if not isinstance(value, tuple):
            value = (value, value)

        self.conv2_conv.dilation_rate = value

    def call(self, inputs, training=None, **kwargs):

        if self.conv_shortcut:
            shortcut = self.shortcut_conv(inputs)
            shortcut = self.shortcut_bn(shortcut, training=training)
        elif self.strides > 1:
            if "avg" in self.downsample_method:
                shortcut = tf.nn.avg_pool2d(inputs, self.conv2_conv.strides, self.conv2_conv.strides, "SAME")
            elif "max" in self.downsample_method:
                shortcut = tf.nn.max_pool2d(inputs, self.conv2_conv.strides, self.conv2_conv.strides, "SAME")
            else:
                raise ValueError("Only max or avg are supported")
        else:
            shortcut = inputs

        x = self.conv1_conv(inputs)
        x = self.conv1_bn(x, training=training)
        x = tf.nn.relu(x)

        x = self.conv2_conv(x, training=training)
        x = self.conv2_bn(x, training=training)
        x = tf.nn.relu(x)

        x = self.conv3_conv(x)
        x = self.conv3_bn(x, training=training)

        x = tf.add(shortcut, x)
        x = tf.nn.relu(x)

        return x
