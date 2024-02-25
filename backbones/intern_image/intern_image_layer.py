# ===================================================================
# MIT License
# Copyright (c) 2024 edwardyehuang (https://github.com/edwardyehuang)
# ===================================================================

import tensorflow as tf

from iseg.layers.dcn_v3.dcn_v3 import DeformableConvolutionV3
from iseg.backbones.intern_image.mlp_layer import MLPLayer

from iseg.utils.drops import drop_path

from iseg.utils.keras3_utils import Keras3_Model_Wrapper

LAYER_NROM_EPSILON = 1e-6

class InternImageLayer (Keras3_Model_Wrapper):

    def __init__(
        self, 
        groups,
        mlp_ratio=4,
        dropout_rate=0.,
        drop_path_rate=0.,
        activation=tf.nn.gelu,
        use_post_norm=False,
        layer_scale=False,
        offset_scale=1.0,
        depthwise_kernel_size=None,
        use_res_post_norm=False,
        center_feature_scale=False,
        trainable=True,
        name=None
    ):
        super().__init__(trainable=trainable, name=name)

        self.groups = groups
        self.mlp_ratio = mlp_ratio
        self.dropout_rate = dropout_rate
        self.drop_path_rate = drop_path_rate

        self.activation = activation
        self.use_post_norm = use_post_norm

        self.layer_scale = layer_scale
        self.offset_scale = offset_scale

        self.depthwise_kernel_size = depthwise_kernel_size
        self.use_res_post_norm = use_res_post_norm
        self.center_feature_scale = center_feature_scale


    
    def build(self, input_shape):

        input_channels = input_shape[-1]
        
        self.norm1 = tf.keras.layers.LayerNormalization(
            epsilon=LAYER_NROM_EPSILON, name=f"{self.name}/norm1",
        )

        self.dcn = DeformableConvolutionV3(
            filters=input_channels,
            kernel_size=3,
            depthwise_kernel_size=self.depthwise_kernel_size,
            strides=1,
            padding="same",
            dilation_rate=1,
            groups=self.groups,
            offset_scale=self.offset_scale,
            activation=self.activation,
            center_feature_scale=self.center_feature_scale,
            name=f"{self.name}/dcn"
        )

        self.norm2 = tf.keras.layers.LayerNormalization(
            epsilon=LAYER_NROM_EPSILON, name=f"{self.name}/norm2"
        )

        self.mlp = MLPLayer(
            hidden_filters=input_channels * self.mlp_ratio,
            activation=self.activation,
            dropout_rate=self.dropout_rate,
            name=f"{self.name}/mlp"
        )

        if self.layer_scale is not None:

            assert not self.use_res_post_norm, "use_res_post_norm and layer_scale can not be used at the same time"

            self.gamma1 = self.add_weight(
                name="gamma1",
                shape=[input_channels],
                dtype=tf.float32,
                initializer=tf.ones_initializer(),
                trainable=self.trainable,
            )

            self.gamma2 = self.add_weight(
                name="gamma2",
                shape=[input_channels],
                dtype=tf.float32,
                initializer=tf.ones_initializer(),
                trainable=self.trainable,
            )
        else:
            self.gamma1 = 1.0
            self.gamma2 = 1.0

        if self.use_res_post_norm:
            self.res_post_norm1 = tf.keras.layers.LayerNormalization(
                epsilon=LAYER_NROM_EPSILON, name=f"{self.name}/res_post_norm1",
            )

            self.res_post_norm2 = tf.keras.layers.LayerNormalization(
                epsilon=LAYER_NROM_EPSILON, name=f"{self.name}/res_post_norm2"
            )
  
        super().build(input_shape)


    def call (self, inputs, training=None):

        residual = x = inputs

        if self.use_post_norm:
            x = self.dcn(x, training=training)
            x = self.norm1(x)
            x *= tf.cast(self.gamma1, x.dtype)
            x = drop_path(x, self.drop_path_rate, training=training)

            residual = x = tf.cast(residual, x.dtype) + x

            x = self.mlp(x, training=training)
            x = self.norm2(x)
            x *= tf.cast(self.gamma2, x.dtype)
            x = drop_path(x, self.drop_path_rate, training=training)

            x += residual


        elif self.use_res_post_norm:

            x = self.norm1(x)
            x = self.dcn(x, training=training)
            x = self.res_post_norm1(x)
            x = drop_path(x, self.drop_path_rate, training=training)

            residual = x = tf.cast(residual, x.dtype) + x

            x = self.norm2(x)
            x = self.mlp(x, training=training)
            x = self.res_post_norm2(x)
            x = drop_path(x, self.drop_path_rate, training=training)

            x += residual

        else: # pre_norm

            x = self.norm1(x)
            x = self.dcn(x, training=training)
            x *= tf.cast(self.gamma1, x.dtype)
            x = drop_path(x, self.drop_path_rate, training=training)

            residual = x = tf.cast(residual, x.dtype) + x

            x = self.norm2(x)
            x = self.mlp(x, training=training)
            x *= tf.cast(self.gamma2, x.dtype)
            x = drop_path(x, self.drop_path_rate, training=training)

            x += residual

        return x
