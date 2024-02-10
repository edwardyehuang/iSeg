# ================================================================
# MIT License
# Copyright (c) 2021 edwardyehuang (https://github.com/edwardyehuang)
# ================================================================

import tensorflow as tf

from iseg.utils.common import get_tensor_shape
from iseg.utils.attention_utils import *
from iseg.initializers.shared_initializers import SharedInitializer
from iseg.vis.vismanager import get_visualization_manager


class SelfAttention(tf.keras.Model):
    def __init__(
        self,
        guided_filters=64,
        filters=512,
        shared_querykey_weights=False,
        shared_querykey=False,
        attention_dropout_rate=0,
        feature_dropout_rate=0,
        apply_scale=False,
        name=None,
    ):

        super().__init__(name=name)

        self.guided_filters = guided_filters
        self.apply_scale = apply_scale

        query_initializer = key_initializer = "glorot_uniform"

        if shared_querykey_weights:
            query_initializer = SharedInitializer(tf.keras.initializers.GlorotUniform())
            key_initializer = query_initializer

        self.query_conv = tf.keras.layers.Conv2D(
            guided_filters, 1, kernel_initializer=query_initializer, name="query_conv"
        )

        if shared_querykey:
            self.key_conv = self.query_conv
        else:
            self.key_conv = tf.keras.layers.Conv2D(
                guided_filters, 1, kernel_initializer=key_initializer, name="key_conv"
            )

        self.value_conv = tf.keras.layers.Conv2D(filters, 1, name="value_conv")

        self.vis_manager = get_visualization_manager()

        self.attention_dropout = tf.keras.layers.Dropout(rate=attention_dropout_rate, name="attention_dropout")
        self.feature_dropout = tf.keras.layers.Dropout(rate=feature_dropout_rate, name="feature_dropout")

    def call(self, inputs, training=None):

        batch_size, height, width, channels = get_tensor_shape(inputs)

        query = self.query_conv(inputs, training=training)
        key = self.key_conv(inputs, training=training)
        value = self.value_conv(inputs, training=training)

        query = flatten_hw(query)
        key = transpose_hw_c(flatten_hw(key))

        attention = get_attention(query, key, apply_scale=self.apply_scale)

        if self.vis_manager.recording:
            self.vis_manager.easy_add(tf.reshape(attention, (-1, height, width, height, width)), name="attention_map")

        value = flatten_hw(value)

        attention = self.attention_dropout(attention, training=training)

        value = tf.matmul(attention, value)
        value = tf.reshape(value, [batch_size, height, width, value.shape[-1]])

        value = self.feature_dropout(value, training=training)

        return value
