import tensorflow as tf
import numpy as np
import math

from iseg.initializers.shared_initializers import SharedInitializer
from iseg.utils import get_tensor_shape
from iseg.utils.keras_ops import replace_inf, replace_nan_or_inf
from iseg import check_numerics
from iseg.utils.keras3_utils import Keras3_Model_Wrapper, is_keras3

def safed_softmax (x):
    t = x.dtype
    x = tf.cast(x, tf.float32)
    x = tf.nn.softmax(x)
    x = tf.cast(x, t)

    return x


class MultiHeadSelfAttentionLayer (Keras3_Model_Wrapper):

    def __init__(
        self, 
        filters=-1,
        num_heads=4,
        apply_linear=True,
        apply_scale=True,
        shared_qk_weights=True,
        shared_qk=False,
        trainable=True,
        linear_func=tf.keras.layers.Conv2D,
        name=None
    ):

        super().__init__(trainable=trainable, name=name)

        self.filters = filters
        self.num_heads = num_heads
        self.apply_linear = apply_linear
        self.apply_scale = apply_scale
        self.shared_qk_weights = shared_qk_weights

        self.shared_qk = shared_qk

        self.linear_func = linear_func

    
    def build (self, input_shape):

        channels = input_shape[-1]

        qk_filters = channels if self.filters == -1 else self.filters

        if self.shared_qk_weights:
            if is_keras3():
                q_kernel_initializer = k_kernel_initializer = tf.keras.initializers.GlorotUniform()
            else:
                q_kernel_initializer = k_kernel_initializer = SharedInitializer(tf.keras.initializers.GlorotUniform())
        else:
            q_kernel_initializer = tf.keras.initializers.GlorotUniform()
            k_kernel_initializer = tf.keras.initializers.GlorotUniform()

        if self.apply_linear:
            self.query_conv = self.linear_func(
                qk_filters,
                (1, 1), 
                kernel_initializer=q_kernel_initializer,
                trainable=self.trainable,
                name="query_conv"
                )
            
            if not self.shared_qk:  
                self.key_conv = self.linear_func(
                    qk_filters, 
                    (1, 1), 
                    kernel_initializer=k_kernel_initializer,
                    trainable=self.trainable,
                    name="key_conv"
                )

            self.value_conv = self.linear_func(
                channels, 
                (1, 1), 
                trainable=self.trainable,
                name="value_conv"
            )

        super().build(input_shape)


    def compute_attetnion (self, query, key, value, training=None):

        batch_size, height, width, _ = get_tensor_shape(value)

        query = replace_nan_or_inf(query, tf.keras.backend.epsilon())
        key = replace_nan_or_inf(key, tf.keras.backend.epsilon())

        query = check_numerics(query, "query contains NaN/Inf", level=1)
        key = check_numerics(key, "keys contains NaN/Inf", level=1)

        query = tf.reshape(query, [batch_size, height * width, self.num_heads, query.shape[-1] // self.num_heads]) # [N, H*W, heads, C//heads]
        query = tf.transpose(query, [0, 2, 1, 3]) # [N, heads, H*W, C//heads]
        key = tf.reshape(key, [batch_size, height * width, self.num_heads, key.shape[-1] // self.num_heads]) # [N, H*W, heads, C//heads]
        key = tf.transpose(key, [0, 2, 3, 1]) # [N, heads, C//heads, H*W]
        value = tf.reshape(value, [batch_size, height * width, self.num_heads, value.shape[-1] // self.num_heads]) # [N, H*W, heads, C//heads]
        value = tf.transpose(value, [0, 2, 1, 3]) # [N, heads, H*W, C//heads]

        attention_map = tf.matmul(query, key) # [N, heads, H*W, H*W]
        attention_map = replace_inf(attention_map)
        attention_map = check_numerics(attention_map, "attention_map contains NaN/Inf", level=1)

        if self.apply_scale:
            attention_map /= tf.sqrt(tf.cast(query.shape[-1], attention_map.dtype))

        attention_map = safed_softmax(attention_map) # [N, heads, H*W, H*W]

        attention_map = replace_nan_or_inf(attention_map, nan_value=tf.keras.backend.epsilon())
        
        attention_map = tf.clip_by_value(attention_map, tf.keras.backend.epsilon(), 1.0 - tf.keras.backend.epsilon())

        attention_map = check_numerics(attention_map, "softmax attention_map contains NaN/Inf", level=1) # [N, heads, H*W, H*W]

        x = tf.matmul(attention_map, value) # [N, heads, H*W, C//heads]
        x = tf.transpose(x, [0, 2, 1, 3]) # [N, H*W, heads, C//heads]
        x = tf.reshape(x, [batch_size, height, width, x.shape[-1] * self.num_heads]) # [N, H, W, C]

        return x


    def call (self, inputs, training=None):

        x = inputs # [N, H, W, C]

        if self.apply_linear:
            query = self.query_conv(x) # [N, H, W, C]

            if not self.shared_qk:
                key = self.key_conv(x)
            else:
                key = tf.identity(query, name="shared_key")

            x = self.value_conv(x) # [N, H, W, C]
            
        else:
            query = x
            key = x


        x = self.compute_attetnion(query, key, x, training=training)

        return x