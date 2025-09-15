# ====================================================================
# MIT License
# Copyright (c) 2024 edwardyehuang (https://github.com/edwardyehuang)
# ====================================================================

import tensorflow as tf

from iseg.utils import get_tensor_shape
from iseg.backbones.eva.rotar_embedding_cat import RotaryEmbeddingCat, apply_rot_embed_cat
from iseg.utils.keras3_utils import Keras3_Model_Wrapper, _N
from iseg.utils.keras_ops import replace_nan_or_inf
from iseg.utils.op_utils import safed_softmax

LAYER_NORM_EPSILON = 1e-6


class EvaAttention (Keras3_Model_Wrapper):

    def __init__(
        self,
        num_heads=8,
        qkv_bias=True,
        qkv_fused=True,
        attention_dropout_rate=0.0,
        projection_dropout_rate=0.0,
        attention_head_filters=None,
        use_norm=True,
        trainable=True,
        name=None,
    ):
        
        super().__init__(trainable=trainable, name=name)

        self.num_heads = num_heads
        self.qkv_bias = qkv_bias
        self.qkv_fused = qkv_fused

        self.attention_dropout_rate = attention_dropout_rate
        self.projection_dropout_rate = projection_dropout_rate

        self.attention_head_filters = attention_head_filters

        self.use_norm = use_norm
        

    def build(self, input_shape):

        input_channels = input_shape[-1]

        head_filters = input_channels // self.num_heads

        if self.attention_head_filters is not None:
            head_filters = self.attention_head_filters

        all_head_filters = head_filters * self.num_heads

        self.attention_scale = head_filters ** -0.5

        if self.qkv_fused:
            self.qkv = tf.keras.layers.Dense(
                all_head_filters * 3,
                use_bias=False,
                name=_N(f"{self.name}/qkv"),
            )

            self.q_bias = self.add_weight(
                name="q_bias",
                shape=[all_head_filters],
                initializer=tf.keras.initializers.Zeros(),
                trainable=True,
            )

            self.k_bias = tf.zeros_like(self.q_bias, name="k_bias")

            self.v_bias = self.add_weight(
                name="v_bias",
                shape=[all_head_filters],
                initializer=tf.keras.initializers.Zeros(),
                trainable=True,
            )

        else:
            self.q_proj = tf.keras.layers.Dense(all_head_filters, use_bias=self.qkv_bias, name=_N(f"{self.name}/q_proj"))
            self.k_proj = tf.keras.layers.Dense(all_head_filters, use_bias=False, name=_N(f"{self.name}/k_proj"))
            self.v_proj = tf.keras.layers.Dense(all_head_filters, use_bias=self.qkv_bias, name=_N(f"{self.name}/v_proj"))

        self.attention_dropout = tf.keras.layers.Dropout(self.attention_dropout_rate, name="attention_dropout")

        if self.use_norm:
            self.norm = tf.keras.layers.LayerNormalization(epsilon=LAYER_NORM_EPSILON, name=_N(f"{self.name}/norm"))
        else:
            self.norm = tf.identity

        self.projection_dropout = tf.keras.layers.Dropout(self.projection_dropout_rate, name="projection_dropout")
        self.projection = tf.keras.layers.Dense(input_channels, use_bias=True, name=_N(f"{self.name}/proj"))

        super().build(input_shape)


    def call(self, inputs, rope=None, training=None):

        x = inputs

        batch_size, hw, channels = get_tensor_shape(x)

        if self.qkv_fused:
            qkv = self.qkv(x)

            qkv_bias = tf.concat(
                [
                    tf.cast(self.q_bias, qkv.dtype),
                    tf.cast(self.k_bias, qkv.dtype),
                    tf.cast(self.v_bias, qkv.dtype)
                ], 
                axis=0
            )

            qkv = tf.nn.bias_add(qkv, qkv_bias)
            qkv = tf.reshape(qkv, [batch_size, hw, 3, self.num_heads, qkv.shape[-1] // (self.num_heads * 3)])
            qkv = tf.transpose(qkv, [2, 0, 3, 1, 4]) # [3, batch_size, num_heads, hw, head_filters]
            q, k, v = tf.unstack(qkv, 3, axis=0) # [batch_size, num_heads, hw, head_filters]
        else:
            q = self.q_proj(x)
            k = self.k_proj(x)
            v = self.v_proj(x)

            q = tf.reshape(q, [batch_size, hw, self.num_heads, q.shape[-1] // self.num_heads])
            k = tf.reshape(k, [batch_size, hw, self.num_heads, k.shape[-1] // self.num_heads])
            v = tf.reshape(v, [batch_size, hw, self.num_heads, v.shape[-1] // self.num_heads])

            q = tf.transpose(q, [0, 2, 1, 3])
            k = tf.transpose(k, [0, 2, 1, 3])
            v = tf.transpose(v, [0, 2, 1, 3]) # [batch_size, num_heads, hw, head_filters]

        q = replace_nan_or_inf(q, tf.keras.backend.epsilon())
        k = replace_nan_or_inf(k, tf.keras.backend.epsilon())
        v = replace_nan_or_inf(v, tf.keras.backend.epsilon())


        if rope is not None:
            q0 = q[:, :, :1, :] # [batch_size, num_heads, 1, head_filters]
            q1 = apply_rot_embed_cat(q[:, :, 1:, :], rope)
            q = tf.concat([q0, q1], axis=2) # [batch_size, num_heads, hw, head_filters]

            k0 = k[:, :, :1, :] # [batch_size, num_heads, 1, head_filters]
            k1 = apply_rot_embed_cat(k[:, :, 1:, :], rope)
            k = tf.concat([k0, k1], axis=2) # [batch_size, num_heads, hw, head_filters]

        q *= self.attention_scale # [batch_size, num_heads, hw, head_filters]
        k = tf.transpose(k, [0, 1, 3, 2]) # [batch_size, num_heads, head_filters, hw]

        attention = tf.matmul(q, k) # [batch_size, num_heads, hw, hw]

        attention = replace_nan_or_inf(attention, tf.keras.backend.epsilon())

        attention = safed_softmax(attention)

        attention = replace_nan_or_inf(attention, tf.keras.backend.epsilon())

        attention = self.attention_dropout(attention, training=training)
        y = tf.matmul(attention, v) # [batch_size, num_heads, hw, head_filters]

        y = replace_nan_or_inf(y, tf.keras.backend.epsilon())

        y = tf.transpose(y, [0, 2, 1, 3]) # [batch_size, hw, num_heads, head_filters]
        y = tf.reshape(y, [batch_size, hw, channels]) # [batch_size, hw, channels]

        y = self.norm(y)
        y = self.projection(y)
        y = self.projection_dropout(y, training=training)

        y = replace_nan_or_inf(y, tf.keras.backend.epsilon())

        return y