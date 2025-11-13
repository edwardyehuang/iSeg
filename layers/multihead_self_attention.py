import tensorflow as tf
import keras
import numpy as np
import math

from iseg.initializers.shared_initializers import SharedInitializer
from iseg.utils import get_tensor_shape
from iseg.utils.keras_ops import replace_inf, replace_nan_or_inf
from iseg import check_numerics
from iseg.utils.version_utils import is_keras3
from iseg.utils.keras3_utils import Keras3_Model_Wrapper

from iseg.utils.op_utils import safed_softmax


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
        use_dense_for_linear=False,
        use_jit_compile=False,
        name=None
    ):

        super().__init__(trainable=trainable, name=name)

        self.filters = filters
        self.num_heads = num_heads
        self.apply_linear = apply_linear
        self.apply_scale = apply_scale
        self.shared_qk_weights = shared_qk_weights

        self.shared_qk = shared_qk

        self.use_dense_for_linear = use_dense_for_linear

        if is_keras3():
            self.use_jit_compile = False

        if tf.distribute.has_strategy():
            strategy = tf.distribute.get_strategy()
            if isinstance(strategy, tf.distribute.TPUStrategy):
                use_jit_compile = False

        self.use_jit_compile = use_jit_compile

    
    def build (self, input_shape):

        linear_func = keras.layers.Dense if self.use_dense_for_linear else keras.layers.Conv2D

        channels = input_shape[-1]

        qk_filters = channels if self.filters == -1 else self.filters

        if self.shared_qk_weights:
            if is_keras3():
                q_kernel_initializer = k_kernel_initializer = keras.initializers.RandomUniform()
            else:
                q_kernel_initializer = k_kernel_initializer = SharedInitializer(keras.initializers.RandomUniform())
        else:
            q_kernel_initializer = keras.initializers.RandomUniform()
            k_kernel_initializer = keras.initializers.RandomUniform()

        if self.apply_linear:
            self.query_conv = linear_func(
                qk_filters,
                (1, 1), 
                kernel_initializer=q_kernel_initializer,
                trainable=self.trainable,
                name="query_conv"
                )
            
            if not self.shared_qk:  
                self.key_conv = linear_func(
                    qk_filters, 
                    (1, 1), 
                    kernel_initializer=k_kernel_initializer,
                    trainable=self.trainable,
                    name="key_conv"
                )

            self.value_conv = linear_func(
                channels, 
                (1, 1), 
                trainable=self.trainable,
                name="value_conv"
            )

        super().build(input_shape)

    
    def compute_attention_internal (self, query, key, value):

        batch_size, height, width, _ = get_tensor_shape(query)

        query = replace_nan_or_inf(query, keras.backend.epsilon())
        key = replace_nan_or_inf(key, keras.backend.epsilon())

        query = check_numerics(query, "query contains NaN/Inf", level=1)
        key = check_numerics(key, "keys contains NaN/Inf", level=1)

        query = tf.reshape(query, [batch_size, -1, self.num_heads, query.shape[-1] // self.num_heads]) # [N, H*W, heads, C//heads]
        query = tf.transpose(query, [0, 2, 1, 3]) # [N, heads, H*W, C//heads]
        key = tf.reshape(key, [batch_size, -1, self.num_heads, key.shape[-1] // self.num_heads]) # [N, H*W, heads, C//heads]
        key = tf.transpose(key, [0, 2, 3, 1]) # [N, heads, C//heads, H*W]
        value = tf.reshape(value, [batch_size, -1, self.num_heads, value.shape[-1] // self.num_heads]) # [N, H*W, heads, C//heads]
        value = tf.transpose(value, [0, 2, 1, 3]) # [N, heads, H*W, C//heads]

        attention_map = tf.matmul(query, key) # [N, heads, H*W, H*W]
        attention_map = replace_inf(attention_map)
        attention_map = check_numerics(attention_map, "attention_map contains NaN/Inf", level=1)

        if self.apply_scale:
            attention_map /= tf.sqrt(tf.cast(query.shape[-1], attention_map.dtype))

        attention_map = safed_softmax(attention_map) # [N, heads, H*W, H*W]

        attention_map = replace_nan_or_inf(attention_map, nan_value=keras.backend.epsilon())
        
        attention_map = tf.clip_by_value(attention_map, keras.backend.epsilon(), 1.0 - keras.backend.epsilon())

        attention_map = check_numerics(attention_map, "softmax attention_map contains NaN/Inf", level=1) # [N, heads, H*W, H*W]

        x = tf.matmul(attention_map, value) # [N, heads, H*W, C//heads]
        x = tf.transpose(x, [0, 2, 1, 3]) # [N, H*W, heads, C//heads]
        x = tf.reshape(x, [batch_size, height, width, x.shape[-1] * self.num_heads]) # [N, H, W, C]

        x = replace_nan_or_inf(x, keras.backend.epsilon())

        return x
    
    @tf.function(jit_compile=True, autograph=False)
    def compute_attention_xla (self, query, key, value):
        return self.compute_attention_internal(query, key, value)


    def compute_attention (self, query, key, value):

        if self.use_jit_compile:
            return self.compute_attention_xla(query, key, value)
        else:
            return self.compute_attention_internal(query, key, value)



    def call (self, inputs, key=None, value=None, training=None):

        query = inputs # [N, H, W, C]

        if key is None:
            key = query

        if value is None:
            value = key

        if self.apply_linear:
            query = self.query_conv(query) # [N, H, W, C]

            if not self.shared_qk:
                key = self.key_conv(key)
            else:
                key = tf.identity(query, name="shared_key")

            value = self.value_conv(value) # [N, H, W, C]


        x = self.compute_attention(query, key, value)

        return x