# ================================================================
# MIT License
# Copyright (c) 2023 edwardyehuang (https://github.com/edwardyehuang)
# ================================================================

# This code implemented https://arxiv.org/pdf/2301.00808.pdf

import tensorflow as tf
import numpy as np

from iseg.utils.drops import drop_path


class GlobalResponseNormlizationLayer(tf.keras.layers.Layer):

    def __init__(self, trainable=True, epsilon=1e-6, name=None):
        super().__init__(trainable=trainable, name=name)

        self.epsilon = epsilon


    def build(self, input_shape):

        channels = input_shape[-1]

        self.gamma = self.add_weight(
            name="gamma",
            shape=[1, 1, 1, channels],
            trainable=self.trainable,
            initializer=tf.zeros_initializer(),
            dtype=tf.float32,
        )

        self.beta = self.add_weight(
            name="beta",
            shape=[1, 1, 1, channels],
            trainable=self.trainable,
            initializer=tf.zeros_initializer(),
            dtype=tf.float32,
        )

        super().build(input_shape)

    def call (self, inputs, training=None):

        x = inputs
        x = tf.cast(x, tf.float32)

        beta = tf.cast(self.beta, tf.float32)
        gamma = tf.cast(self.gamma, tf.float32)

        gx = tf.pow((tf.reduce_sum(tf.pow(x, 2), axis=(1, 2), keepdims=True) + self.epsilon), 0.5)
        # gx = tf.nn.l2_normalize(x, axis=(1, 2), epsilon=1e-6)
        nx = gx / (tf.reduce_mean(gx, axis=-1, keepdims=True) + self.epsilon)

        x = gamma * (x * nx) + beta + x

        return tf.cast(x, inputs.dtype)


class Block(tf.keras.Model):
    def __init__(self, filters, drop_path_prob=0.0, name=None):

        super().__init__(name=name)

        self.drop_path_prob = drop_path_prob
        self.filters = filters

        self.dwconv = tf.keras.layers.DepthwiseConv2D(kernel_size=7, padding="same", name=f"{self.name}/dwconv")

        self.norm = tf.keras.layers.LayerNormalization(epsilon=1e-6, name=f"{self.name}/norm")

        self.pwconv1 = tf.keras.layers.Dense(units=4 * filters, name=f"{self.name}/pwconv1")
        self.grn = GlobalResponseNormlizationLayer(trainable=self.trainable, name=f"{self.name}/grn")
        self.pwconv2 = tf.keras.layers.Dense(units=filters, name=f"{self.name}/pwconv2")

    def build(self, input_shape):

        super().build(input_shape)

    def call(self, inputs, training=None):

        x = inputs
        x = self.dwconv(x)
        x = self.norm(x)
        x = self.pwconv1(x)
        x = tf.nn.gelu(x)
        x = self.grn(x)
        x = self.pwconv2(x)

        if self.drop_path_prob != 0.0 and training:
            x = drop_path(x, drop_prob=self.drop_path_prob, training=training)
        x += tf.cast(inputs, x.dtype)

        return x


class DownSampleLayer(tf.keras.Model):
    def __init__(self, filters=96, strides=2, swap=False, name=None):
        super().__init__(name=name)

        self.swap = swap
        names = ["1", "0"] if swap else ["0", "1"]

        self.norm = tf.keras.layers.LayerNormalization(epsilon=1e-6, name=f"{self.name}/{names[0]}")
        self.conv = tf.keras.layers.Conv2D(
            filters=filters, kernel_size=strides, strides=strides, padding="same", name=f"{self.name}/{names[1]}"
        )

    def build(self, input_shape):

        super().build(input_shape)

    def call(self, inputs, training=None):

        x = inputs

        if self.swap:
            x = self.norm(self.conv(x))
        else:
            x = self.conv(self.norm(x))

        return x


class Stage(tf.keras.Model):
    def __init__(self, filters=96, depth=3, drop_path_probs=[], name=None):

        super().__init__(name=name)

        assert len(drop_path_probs) == 0 or len(drop_path_probs) == depth

        self.blocks = [
            Block(
                filters=filters,
                drop_path_prob=drop_path_probs[i],
                name=f"{self.name}/{i}",
            )
            for i in range(depth)
        ]

    def build(self, input_shape):

        super().build(input_shape)

    def call(self, inputs, training=None):

        x = inputs

        for block in self.blocks:
            x = block(x, training=training)

        return x


class ConvNeXtV2(tf.keras.Model):
    def __init__(
        self,
        depths=[3, 3, 9, 3],
        filters_list=[96, 192, 384, 768],
        drop_path_rate=0.0,
        return_endpoints=False,
        name=None,
    ):
        super().__init__(name=name)

        self.return_endpoints = return_endpoints

        num_stage = len(depths)
        assert num_stage == len(filters_list)

        drop_path_rates = np.linspace(0.0, drop_path_rate, sum(depths))

        self.downsample_blocks = []
        self.stages = []

        cur = 0

        for i in range(num_stage):
            self.downsample_blocks += [
                DownSampleLayer(
                    filters=filters_list[i], strides=4 if i == 0 else 2, swap=i == 0, name=f"downsample_layers/{i}"
                )
            ]

            self.stages += [
                Stage(
                    filters=filters_list[i],
                    depth=depths[i],
                    drop_path_probs=drop_path_rates[cur : cur + depths[i]],
                    name=f"stages/{i}",
                )
            ]

            cur += depths[i]

    def build(self, input_shape):

        super().build(input_shape)

    def call(self, inputs, training=None):

        x = inputs

        endpoints = [None]

        for i in range(len(self.stages)):
            x = self.downsample_blocks[i](x, training=training)
            x = self.stages[i](x, training=training)

            endpoints += [x]

        if self.return_endpoints:
            return endpoints

        return x


def convnext_v2_nano(return_endpoints=False):

    return ConvNeXtV2(depths=[2, 2, 8, 2], filters_list=[80, 160, 320, 640], return_endpoints=return_endpoints,)


def convnext_v2_tiny(return_endpoints=False):

    return ConvNeXtV2(depths=[3, 3, 9, 3], filters_list=[96, 192, 384, 768], return_endpoints=return_endpoints,)


def convnext_v2_large(return_endpoints=False):

    return ConvNeXtV2(depths=[3, 3, 27, 3], filters_list=[192, 384, 768, 1536], return_endpoints=return_endpoints,)


def convnext_v2_huge(return_endpoints=False):

    return ConvNeXtV2(depths=[3, 3, 27, 3], filters_list=[352, 704, 1408, 2816], return_endpoints=return_endpoints,)


def build_dilated_convnext(model: ConvNeXtV2, output_stride=32):

    num_stages = len(model.stages)

    current_os = 1
    current_dilation = 1

    for i in range(num_stages):

        if current_os >= output_stride:
            current_dilation *= model.downsample_blocks[i].conv.strides[0]
            model.downsample_blocks[i].conv.strides = (1, 1)
            model.downsample_blocks[i].conv.dilation_rate = (current_dilation, current_dilation)

            for block in model.stages[i].blocks:
                block.dwconv.strides = (1, 1)
                block.dwconv.dilation_rate = (current_dilation, current_dilation)

        else:
            current_os *= model.downsample_blocks[i].conv.strides[0]

    return model
