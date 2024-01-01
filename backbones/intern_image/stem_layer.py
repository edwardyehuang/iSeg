# ===================================================================
# MIT License
# Copyright (c) 2024 edwardyehuang (https://github.com/edwardyehuang)
# ===================================================================

import tensorflow as tf

from iseg.layers.normalizations import normalization

LAYER_NORM_EPSILON = 1e-6

class StemLayer (tf.keras.Model):

    def __init__(
        self, 
        filters=96,
        activation=tf.nn.gelu,
        name=None,
    ):

        super().__init__(name=name)

        self.filters = filters
        self.activation = activation


    def build(self, input_shape):

        self.conv1 = tf.keras.layers.Conv2D(
            self.filters // 2, 
            kernel_size=3, 
            strides=2, 
            padding="same", 
            name=f"{self.name}/conv1",
        )

        self.norm1 = tf.keras.layers.LayerNormalization(
            epsilon=LAYER_NORM_EPSILON, name=f"{self.name}/norm1",
        )

        self.conv2 = tf.keras.layers.Conv2D(
            self.filters, 
            kernel_size=3, 
            strides=2, 
            padding="same", 
            name=f"{self.name}/conv2"
        )

        self.norm2 = tf.keras.layers.LayerNormalization(
            epsilon=LAYER_NORM_EPSILON, name=f"{self.name}/norm2"
        )


        super().build(input_shape)


    def call(self, inputs, training=None):

        x = self.conv1(inputs)
        x = self.norm1(x)
        x = self.activation(x)

        x = self.conv2(x)
        x = self.norm2(x)
        
        return x