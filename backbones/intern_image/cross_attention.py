# ===================================================================
# MIT License
# Copyright (c) 2024 edwardyehuang (https://github.com/edwardyehuang)
# ===================================================================

import tensorflow as tf

from iseg.utils import get_tensor_shape
from iseg.backbones.intern_image.utils import extract_qkv
from iseg.utils.keras3_utils import Keras3_Model_Wrapper

class CrossAttention(Keras3_Model_Wrapper):

    def __init__(
        self,
        filters=None,
        num_heads=8,
        use_qkv_bias=False,
        qk_scale=None,
        attention_dropout_rate=0.,
        projection_dropout_rate=0.,
        attention_head_filters=None,
        name=None
    ):
        super().__init__(name=name)

        self.filters = filters
        self.num_heads = num_heads
        self.use_qkv_bias = use_qkv_bias

        self.qk_scale = qk_scale
        self.attention_dropout_rate = attention_dropout_rate
        self.projection_dropout_rate = projection_dropout_rate

        self.attention_head_filters = attention_head_filters


    def build(self, input_shape):

        input_channels = input_shape[-1]

        if self.filters is None:
            self.filters = input_channels

        head_filters = input_channels // self.num_heads

        if self.attention_head_filters is None:
            head_filters = self.attention_head_filters

        all_head_filters = head_filters * self.num_heads

        self.scale = self.qk_scale or head_filters ** -0.5

        assert all_head_filters == input_channels

        self.q = tf.keras.layers.Dense(
            head_filters, 
            use_bias=self.use_qkv_bias, 
            name=f"{self.name}/q",
        )

        self.k = tf.keras.layers.Dense(
            head_filters, 
            use_bias=self.use_qkv_bias, 
            name=f"{self.name}/k",
        )

        self.v = tf.keras.layers.Dense(
            head_filters, 
            use_bias=self.use_qkv_bias, 
            name=f"{self.name}/v",
        )

        self.attention_dropout = tf.keras.layers.Dropout(
            self.attention_dropout_rate, 
            name=f"{self.name}/attention_dropout",
        )

        self.projection = tf.keras.layers.Dense(
            all_head_filters, 
            name=f"{self.name}/projection"
        )

        self.projection_dropout = tf.keras.layers.Dropout(
            self.projection_dropout_rate, 
            name=f"{self.name}/projection_dropout"
        )

        super.build(input_shape)


    def call (self, inputs, training=None):

        x = inputs

        q, k, v = extract_qkv(x)
        
        batch_size, height, width, channels = get_tensor_shape(q)

        q, k, v = self.q(q), self.k(k), self.v(v)
        hw = height * width

        q = tf.reshape(q, [batch_size, hw, self.num_heads, q.shape[-1]])
        q = tf.transpose(q, [0, 2, 1, 3]) # [batch_size, num_heads, hw, head_filters]

        k = tf.reshape(k, [batch_size, hw, self.num_heads, k.shape[-1]])
        k = tf.transpose(k, [0, 2, 3, 1]) # [batch_size, num_heads, head_filters, hw]

        v = tf.reshape(v, [batch_size, hw, self.num_heads, v.shape[-1]])
        v = tf.transpose(v, [0, 2, 1, 3]) # [batch_size, num_heads, hw, head_filters]

        q *= self.scale
        a = tf.matmul(q, k) # [batch_size, num_heads, hw, hw]
        a = tf.nn.softmax(a, axis=-1)
        a = self.attention_dropout(a, training=training)

        x = tf.matmul(a, v) # [batch_size, num_heads, hw, head_filters]

        x = tf.transpose(x, [0, 2, 1, 3]) # [batch_size, hw, num_heads, head_filters]
        x = tf.reshape(x, [batch_size, height, width, -1])

        x = self.projection(x)
        x = self.projection_dropout(x, training=training)

        return x


    