# ================================================================
# MIT License
# Copyright (c) 2021 edwardyehuang (https://github.com/edwardyehuang)
# ================================================================

# This code is motified from https://github.com/keras-team/keras/blob/master/keras/applications/xception.py
# The modification is refer to https://github.com/tensorflow/models/blob/master/research/deeplab/core/xception.py

import tensorflow as tf
from iseg.layers.normalizations import normalization


class XceptionDepthWiseConv(tf.keras.Model):
    def __init__(
        self, block_index, conv_index, filters, strides=(1, 1), activation=False, weight_decay=0.0, momentun=0.9
    ):

        super(XceptionDepthWiseConv, self).__init__(name="XceptionDepthWiseConv")

        block_index = str(block_index)
        conv_index = str(conv_index)

        self.activation = activation

        prefix = "block" + block_index + "_separable_conv" + conv_index

        depthwise_conv_name = prefix + "_depthwise"
        depthwise_bn_name = depthwise_conv_name + "_BN"

        self.depthwise_conv = tf.keras.layers.DepthwiseConv2D(
            (3, 3), strides=strides, padding="same", use_bias=False, name=depthwise_conv_name
        )
        self.depthwise_bn = normalization(name=depthwise_bn_name)

        pointwise_conv_name = prefix + "_pointwise"
        pointwise_bn_name = pointwise_conv_name + "_BN"

        self.pointwise_conv = tf.keras.layers.Conv2D(
            filters, (1, 1), padding="same", use_bias=False, name=pointwise_conv_name
        )
        self.pointwise_bn = normalization(name=pointwise_bn_name)

    def call(self, inputs, training=None, **kwargs):

        x = inputs

        if not self.activation:
            x = tf.nn.relu(x)

        x = self.depthwise_conv(x)
        x = self.depthwise_bn(x, training=training)

        if self.activation:
            x = tf.nn.relu(x)

        x = self.pointwise_conv(x)
        x = self.pointwise_bn(x, training=training)

        if self.activation:
            x = tf.nn.relu(x)

        return x

    @property
    def strides(self):
        return self.depthwise_conv.strides

    @strides.setter
    def strides(self, value):
        self.depthwise_conv.strides = value

    @property
    def atrous_rates(self):
        return self.depthwise_conv.dilation_rate

    @atrous_rates.setter
    def atrous_rates(self, value):
        self.depthwise_conv.dilation_rate = value


class XceptionBlock(tf.keras.Model):
    def __init__(self, block_index, filters_list, strides, skip_connection=0, activation=False):

        super(XceptionBlock, self).__init__(name="XceptionBlock")

        length = len(filters_list)

        if not isinstance(strides, tuple):
            strides = (strides, strides)

        self.convs = []

        for i in range(length):
            conv_strides = (1, 1) if i < length - 1 else strides
            self.convs.append(
                XceptionDepthWiseConv(block_index, i + 1, filters_list[i], strides=conv_strides, activation=activation)
            )

        self.skip_connection = skip_connection

        if skip_connection == 2:

            shortcut_name = "block" + str(block_index) + "_shortcut"

            self.shortcut = tf.keras.layers.Conv2D(
                filters_list[-1], (1, 1), strides=strides, padding="same", use_bias=False, name=shortcut_name
            )
            self.shortcut_bn = normalization(name=shortcut_name + "_BN")

    def call(self, inputs, training=None, **kwargs):

        x = inputs

        for i in range(len(self.convs)):
            x = self.convs[i](x, training=training)

            if i == 1:
                skip_result = x

        if self.skip_connection == 1:
            x = x + inputs
        elif self.skip_connection == 2:
            shortcut_result = self.shortcut(inputs)
            shortcut_result = self.shortcut_bn(shortcut_result, training=training)

            x = x + shortcut_result

        return x

    @property
    def strides(self):
        return self.convs[-1].strides

    @strides.setter
    def strides(self, value):

        if not isinstance(value, tuple):
            value = (value, value)

        self.convs[-1].strides = value

        if self.skip_connection == 2:
            self.shortcut.strides = value

    @property
    def atrous_rates(self):
        return self.convs[-1].atrous_rates

    @atrous_rates.setter
    def atrous_rates(self, value):
        for conv in self.convs:
            conv.atrous_rates = value

        if self.skip_connection == 2:
            self.shortcut.dilation_rate = value


class Xception(tf.keras.Model):
    def __init__(self, return_endpoints=False, name=None):

        super(Xception, self).__init__(name=name)

        self._xception_blocks = []

        # Block 1

        self.block1_conv1 = tf.keras.layers.Conv2D(
            32, (3, 3), strides=(2, 2), padding="same", use_bias=False, name="block1_conv1"
        )
        self.block1_conv1_bn = normalization(name="block1_conv1_BN")

        self.block1_conv2 = tf.keras.layers.Conv2D(64, (3, 3), padding="same", use_bias=False, name="block1_conv2")
        self.block1_conv2_bn = normalization(name="block1_conv2_BN")

        self.return_endpoints = return_endpoints

    def call(self, inputs, training=None, **kwargs):

        endpoints = []

        x = self.block1_conv1(inputs)
        x = self.block1_conv1_bn(x, training=training)
        x = tf.nn.relu(x)

        endpoints += [x]

        x = self.block1_conv2(x)
        x = self.block1_conv2_bn(x, training=training)
        x = tf.nn.relu(x)

        for i in range(len(self._xception_blocks)):

            if self._xception_blocks[i].strides[0] > 1:
                endpoints += [x]

            x = self._xception_blocks[i](x, training=training)

        assert endpoints[-1] != x
        endpoints += [x]

        if self.return_endpoints:
            return endpoints
        else:
            return x

    def add_xception_block(self, filters_list, strides, skip_connection=0, activation=False, repeat=1):
        index = len(self._xception_blocks) + 2

        for i in range(repeat):
            self._xception_blocks.append(
                XceptionBlock(
                    block_index=index + i,
                    filters_list=filters_list,
                    strides=strides,
                    skip_connection=skip_connection,
                    activation=activation,
                )
            )

    @property
    def xception_blocks(self):
        return self._xception_blocks


def xception65(return_endpoints=False):

    model = Xception(return_endpoints=return_endpoints)

    model.add_xception_block([128, 128, 128], strides=2, skip_connection=2)  # Block2
    model.add_xception_block([256, 256, 256], strides=2, skip_connection=2)  # Block3
    model.add_xception_block([728, 728, 728], strides=2, skip_connection=2)  # Block4
    model.add_xception_block([728, 728, 728], strides=1, skip_connection=1, repeat=16)  # Block5
    model.add_xception_block([728, 1024, 1024], strides=2, skip_connection=2)  # Block 2nd last
    model.add_xception_block([1536, 1536, 2048], strides=1, skip_connection=0, activation=True)  # Block last

    return model


def build_atrous_xception(model: Xception, output_stride=32):

    current_os = 2
    current_atrous_rates = 1

    blocks = model.xception_blocks

    for block in blocks:
        if current_os >= output_stride:
            block.atrous_rates = (current_atrous_rates, current_atrous_rates)
            current_atrous_rates *= block.strides[0]

            if block.strides[0] > 1:
                block.strides = 1

        else:
            current_os *= block.strides[0]

    return model
