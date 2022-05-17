# ================================================================
# MIT License
# Copyright (c) 2021 edwardyehuang (https://github.com/edwardyehuang)
# ================================================================

# WIP DO NOT USE

import tensorflow as tf

from tensorflow.python.keras.utils import conv_utils

from iseg.layers.normalizations import normalization


class MobileNetV2(tf.keras.Model):
    def __init__(self, alpha=1.0, return_endpoints=False, name=None):
        super().__init__(name=name)

        first_block_filters = _make_divisible(32 * alpha, 8)

        self.conv1 = tf.keras.layers.Conv2D(
            first_block_filters, (3, 3), strides=(2, 2), padding="same", use_bias=False, name="Conv1"
        )
        self.bn_conv1 = normalization(name="bn_Conv1")

        self.blocks = []

        self.__add_blocks(16, alpha, stride=1, expansion=1, repeated=1)
        self.__add_blocks(24, alpha, stride=2, expansion=6, repeated=2)
        self.__add_blocks(32, alpha, stride=2, expansion=6, repeated=3)
        self.__add_blocks(64, alpha, stride=2, expansion=6, repeated=4)
        self.__add_blocks(96, alpha, stride=1, expansion=6, repeated=3)
        self.__add_blocks(160, alpha, stride=2, expansion=6, repeated=3)
        self.__add_blocks(320, alpha, stride=1, expansion=6, repeated=1)

        last_block_filters = _make_divisible(1280 * alpha, 8) if alpha > 1.0 else 1280

        self.last_block_conv = tf.keras.layers.Conv2D(last_block_filters, (1, 1), use_bias=False, name="Conv_1")
        self.last_block_conv_bn = normalization(name="Conv_1_bn")

        self.return_endpoints = return_endpoints

    def call(self, inputs, training=None):

        endpoints = []

        x = self.conv1(inputs)
        x = self.bn_conv1(x, training=training)
        x = tf.nn.relu6(x)

        for block in self.blocks:

            if block.orginal_stride > 1:
                endpoints += [x]

            x = block(x, training=training)

        x = self.last_block_conv(x)
        x = self.last_block_conv_bn(x, training=training)
        x = tf.nn.relu6(x)

        endpoints += [x]

        if self.return_endpoints:
            return endpoints
        else:
            return x

    def __add_blocks(self, filters, alpha, stride=1, expansion=1, repeated=1):

        for i in range(repeated):

            block_id = len(self.blocks)
            adjusted_stride = stride if i == 0 else 1

            self.blocks.append(
                InvertedResBlock(
                    filters=filters, alpha=alpha, stride=adjusted_stride, expansion=expansion, block_id=block_id
                )
            )


class InvertedResBlock(tf.keras.Model):
    def __init__(self, expansion, stride, alpha, filters, block_id):
        super().__init__()

        self.expansion = expansion
        self.orginal_stride = stride
        self.alpha = alpha
        self.filters = filters
        self.block_id = block_id

        self.prefix = "block_{}_".format(self.block_id)

        if not self.block_id:
            self.prefix = "expanded_conv_"

        self.depthwise = tf.keras.layers.DepthwiseConv2D(
            (3, 3),
            strides=stride,
            use_bias=False,
            padding="same" if stride == 1 else "valid",
            name=self.prefix + "depthwise",
        )

        self.depthwise_bn = normalization(name=self.prefix + "depthwise_BN")

    def build(self, input_shape):

        self.in_channels = input_shape[-1]
        pointwise_conv_filters = int(self.filters * self.alpha)
        self.pointwise_filters = _make_divisible(pointwise_conv_filters, 8)

        if self.block_id:
            self.expand_conv = tf.keras.layers.Conv2D(
                self.expansion * self.in_channels, (1, 1), padding="same", use_bias=False, name=self.prefix + "expand"
            )
            self.expand_bn = normalization(name=self.prefix + "expand_BN")
            
        self.pad = tf.keras.layers.ZeroPadding2D(padding=correct_pad(input_shape, 3), name=self.prefix + "pad")

        self.project = tf.keras.layers.Conv2D(
            self.pointwise_filters, (1, 1), padding="same", use_bias=False, name=self.prefix + "project"
        )
        self.project_bn = normalization(name=self.prefix + "project_BN")

        # super().build(input_shape)

    @property
    def strides(self):
        return self.depthwise.strides[0]

    @strides.setter
    def strides(self, value):

        value = conv_utils.normalize_tuple(value, self.depthwise.rank, "strides")

        self.depthwise.strides = value
        self.depthwise.padding = "same" if value[0] == 1 else "valid"
        

    @property
    def atrous_rates(self):
        return self.depthwise.dilation_rate[0]

    @atrous_rates.setter
    def atrous_rates(self, value):

        value = conv_utils.normalize_tuple(value, self.depthwise.rank, "dilation_rate")

        self.depthwise.dilation_rate = value

    def call(self, inputs, training=None):

        x = inputs

        if self.block_id:
            x = self.expand_conv(x)
            x = self.expand_bn(x, training=training)
            x = tf.nn.relu6(x)

        if self.strides == 2:
            x = self.pad(x)

        x = self.depthwise(x)
        x = self.depthwise_bn(x, training=training)
        x = tf.nn.relu6(x)

        x = self.project(x)
        x = self.project_bn(x, training=training)

        if self.in_channels == self.pointwise_filters and self.orginal_stride == 1:
            x = inputs + x

        return x


def correct_pad(inputs_shape, kernel_size):

    img_dim = 2 if tf.keras.backend.image_data_format() == "channels_first" else 1
    input_size = inputs_shape[img_dim : (img_dim + 2)]
    if isinstance(kernel_size, int):
        kernel_size = (kernel_size, kernel_size)
    if input_size[0] is None:
        adjust = (1, 1)
    else:
        adjust = (1 - input_size[0] % 2, 1 - input_size[1] % 2)
    correct = (kernel_size[0] // 2, kernel_size[1] // 2)
    return ((correct[0] - adjust[0], correct[0]), (correct[1] - adjust[1], correct[1]))


def _make_divisible(v, divisor, min_value=None):
    if min_value is None:
        min_value = divisor
    new_v = max(min_value, int(v + divisor / 2) // divisor * divisor)
    # Make sure that round down does not go down by more than 10%.
    if new_v < 0.9 * v:
        new_v += divisor

    return new_v


def build_atrous_mobilenetv2(net: MobileNetV2, output_stride=32):

    current_os = 2
    current_dilation_rate = 1

    for block in net.blocks:
        block: InvertedResBlock = block

        if block.strides > 1:
            if current_os >= output_stride:
                current_dilation_rate *= block.strides
                block.strides = 1
                block.atrous_rates = current_dilation_rate
            else:
                current_os *= block.strides
        else:
            block.atrous_rates = current_dilation_rate

    return net
