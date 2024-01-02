# ===================================================================
# MIT License
# Copyright (c) 2024 edwardyehuang (https://github.com/edwardyehuang)
# ===================================================================


import tensorflow as tf

from iseg.layers.dcn_v3.op import dcnv3_op

from iseg.layers.model_builder import get_tensor_shape_v2

LAYER_NORM_EPSILON = 1e-6

class DeformableConvolutionV3 (tf.keras.Model):

    def __init__(
        self, 
        filters=64,
        kernel_size=3,
        depthwise_kernel_size=None,
        strides=1,
        padding="SAME",
        dilation_rate=1,
        groups=4,
        offset_scale=1.0,
        activation=tf.nn.gelu,
        center_feature_scale=False,
        name=None
    ):
        super().__init__(name=name)

        assert filters % groups == 0, "filters must be divisible by groups"

        filters_per_group = filters // groups

        depthwise_kernel_size = depthwise_kernel_size or kernel_size

        self.offset_scale = offset_scale
        self.filters = filters
        self.kernel_size = kernel_size
        self.depthwise_kernel_size = depthwise_kernel_size
        self.strides = strides
        self.padding = padding
        self.dilation_rate = dilation_rate
        self.activation = activation

        if self.activation is None:
            self.activation = tf.identity

        self.groups = groups
        self.filters_per_group = filters_per_group

        self.center_feature_scale = center_feature_scale


    def build(self, input_shape):

        input_channel = input_shape[-1]

        self.dw_conv = tf.keras.layers.DepthwiseConv2D(
            kernel_size=self.depthwise_kernel_size,
            strides=1,
            padding=self.padding.lower(),
            name=f"{self.name}/dw_conv"
        )

        self.dw_norm = tf.keras.layers.LayerNormalization(
            epsilon=LAYER_NORM_EPSILON,
            name=f"{self.name}/dw_conv_norm"
        )

        self.offset = tf.keras.layers.Dense(
            units=2 * self.groups * self.kernel_size * self.kernel_size,
            kernel_initializer=tf.zeros_initializer(),
            bias_initializer=tf.zeros_initializer(),
            name=f"{self.name}/offset"
        )

        self.mask = tf.keras.layers.Dense(
            units=self.groups * self.kernel_size * self.kernel_size,
            kernel_initializer=tf.zeros_initializer(),
            bias_initializer=tf.zeros_initializer(),
            name=f"{self.name}/mask"
        )

        self.input_proj = tf.keras.layers.Dense(
            input_channel,
            name=f"{self.name}/input_proj"
        )

        self.output_proj = tf.keras.layers.Dense(
            self.filters,
            name=f"{self.name}/output_proj"
        )

        if self.center_feature_scale:
            self.center_feature_scale_proj = tf.keras.layers.Dense(
                self.groups,
                name=f"{self.name}/center_feature_scale_proj"
            )

        super().build(input_shape)


    def call(self, inputs, training=False):
        
        batch_size, height, width, channels = get_tensor_shape_v2(inputs)

        x = inputs
        
        x_proj = self.input_proj(x) # [N, H, W, C]

        x1 = self.dw_conv(x) # [N, H, W, C]
        x1 = self.dw_norm(x1) # [N, H, W, C]
        x1 = self.activation(x1) # [N, H, W, C]
        offset = self.offset(x1) # [N, H, W, 2 * groups * kernel_size * kernel_size]

        mask = self.mask(x1) # [N, H, W, groups * kernel_size * kernel_size]
        mask = tf.reshape(mask, [batch_size, height, width, self.groups, -1]) # [N, H, W, groups, kernel_size * kernel_size]
        mask = tf.nn.softmax(mask, axis=-1)
        mask = tf.reshape(mask, [batch_size, height, width, -1]) # [N, H, W, groups * kernel_size * kernel_size]

        x = dcnv3_op(
            x=x_proj,
            offset=offset,
            mask=mask,
            kernel_size=[self.kernel_size, self.kernel_size],
            strides=[self.strides, self.strides],
            padding=self.padding,
            dilation_rate=[self.dilation_rate, self.dilation_rate],
            groups=self.groups,
            group_channels=self.filters_per_group,
            offset_scale=self.offset_scale
        ) # [N, H, W, C]

        if self.center_feature_scale:
            center_feature_scale = self.center_feature_scale_proj(x1) # [N, H, W, groups]
            center_feature_scale = tf.expand_dims(center_feature_scale, axis=-1) # [N, H, W, groups, 1]
            center_feature_scale *= tf.ones(
                [batch_size, height, width, self.groups, self.filters_per_group], 
                dtype=center_feature_scale.dtype
            ) # [N, H, W, groups, C // groups]
            center_feature_scale = tf.reshape(center_feature_scale, [batch_size, height, width, channels])    
            x = x * (1 - center_feature_scale) + x_proj * center_feature_scale

        x = self.output_proj(x) # [N, H, W, C]

        return x
