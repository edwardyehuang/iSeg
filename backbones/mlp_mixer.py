# ================================================================
# MIT License
# Copyright (c) 2021 edwardyehuang (https://github.com/edwardyehuang)
# ================================================================

import iseg.static_strings as ss
import tensorflow as tf

from iseg.layers.model_builder import get_tensor_shape


class MLPBlock(tf.keras.Model):
    def __init__(self, filters, name=None):
        super().__init__(name=name)

        self.filters = filters

    def build(self, input_shape):

        self.dense0 = tf.keras.layers.Dense(self.filters, activation=tf.nn.gelu, name="dense0")
        self.dense1 = tf.keras.layers.Dense(input_shape[-1], name="dense1")

    def call(self, inputs, training=None):

        x = self.dense0(inputs)
        x = self.dense1(x)

        return x


class MixerBlock(tf.keras.Model):
    def __init__(self, token_filters, channel_filters, name=None):
        super().__init__(name=name)

        self.norm_0 = tf.keras.layers.LayerNormalization()
        self.norm_1 = tf.keras.layers.LayerNormalization()

        self.token_mixing = MLPBlock(token_filters, name="token_mixing")
        self.channel_mixing = MLPBlock(channel_filters, name="channel_mixing")

    def call(self, inputs, training=None):

        x = self.norm_0(inputs, training=training)  # [N, HW, C]

        x = tf.transpose(x, [0, 2, 1])  # [N, C, HW]
        x = self.token_mixing(x)  # [N, C, HW]

        x = tf.transpose(x, [0, 2, 1])  # [N, HW, C]

        x += inputs
        idenity = x

        x = self.norm_1(x, training=training)
        x = self.channel_mixing(x)

        return idenity + x


class MLPMixer(tf.keras.Model):
    def __init__(self, filters=768, patch_size=16, num_blocks=12, token_filters=384, channel_filters=3072, name=None):

        super().__init__(name=name)

        self.filters = filters
        self.patch_size = patch_size
        self.num_blocks = num_blocks
        self.token_filters = token_filters
        self.channel_filters = channel_filters

    def build(self, input_shape):

        self.stem = tf.keras.layers.Conv2D(self.filters, self.patch_size, self.patch_size, name="stem")

        self.blocks = [MixerBlock(self.token_filters, self.channel_filters) for _ in self.num_blocks]
        self.pre_head_layer_norm = tf.keras.layers.LayerNormalization(name="pre_head_layer_norm")

    def call(self, inputs, training=None):

        x = self.stem(inputs)

        batch_size, height, width, channels = get_tensor_shape(x)

        x = tf.reshape(inputs, [batch_size, height * width, channels])

        for block in self.blocks:
            x = block(x, training=training)

        x = self.pre_head_layer_norm(x, training=training)

        return x


def mixer_b16():

    return MLPMixer(
        filters=768, patch_size=16, num_blocks=12, token_filters=384, channel_filters=3072, name="Mixer-B_16"
    )


def mixer_l16():

    return MLPMixer(
        filters=1024, patch_size=16, num_blocks=24, token_filters=512, channel_filters=4096, name="Mixer-L_16"
    )
