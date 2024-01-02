# ===================================================================
# MIT License
# Copyright (c) 2024 edwardyehuang (https://github.com/edwardyehuang)
# ===================================================================

import tensorflow as tf

LAYER_NORM_EPSILON = 1e-6

class DownsampleLayer(tf.keras.Model):

    def __init__(self, name=None):
        super().__init__(name=name)

    
    def build (self, input_shape):

        input_channel = input_shape[-1]

        self.conv = tf.keras.layers.Conv2D(
            input_channel * 2, 
            kernel_size=3, 
            strides=2, 
            padding='same', 
            use_bias=False, 
            name=f"{self.name}/conv"
        )

        self.norm = tf.keras.layers.LayerNormalization(
            epsilon=LAYER_NORM_EPSILON, name=f"{self.name}/norm"
        )

        super().build(input_shape)


    def call(self, inputs, training=False):

        x = self.conv(inputs)
        x = self.norm(x)

        return x