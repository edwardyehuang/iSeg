# ====================================================================
# MIT License
# Copyright (c) 2024 edwardyehuang (https://github.com/edwardyehuang)
# ====================================================================

import tensorflow as tf

from iseg.utils.keras3_utils import Keras3_Model_Wrapper, _N

LAYER_NORM_EPSILON = 1e-6

class SwiGLU (Keras3_Model_Wrapper):

    def __init__(
        self,
        hidden_filters=None,
        output_filters=None,
        activation=tf.nn.silu,
        use_bias=True,
        use_norm=True,
        dropout_rate=0.0,
        trainable=True, 
        name=None
    ):
        super().__init__(trainable=trainable, name=name)

        self.hidden_filters = hidden_filters
        self.output_filters = output_filters

        self.activation = activation
        self.use_bias = use_bias
        self.use_norm = use_norm

        self.dropout_rate = dropout_rate


    def build(self, input_shape):

        input_channels = input_shape[-1]

        output_filters = self.output_filters or input_channels
        hidden_filters = self.hidden_filters or input_channels

        self.fc1_g = tf.keras.layers.Dense(
            hidden_filters,
            use_bias=self.use_bias,
            name=_N(f"{self.name}/fc1_g"),
        )

        self.fc1_x = tf.keras.layers.Dense(
            hidden_filters,
            use_bias=self.use_bias,
            name=_N(f"{self.name}/fc1_x"),
        )

        self.drop1 = tf.keras.layers.Dropout(
            rate=self.dropout_rate,
            name="drop1",
        )

        if self.use_norm:
            self.norm = tf.keras.layers.LayerNormalization(
                epsilon=LAYER_NORM_EPSILON,
                name=_N(f"{self.name}/norm"),
            )
        else:
            self.norm = tf.identity

        self.fc2 = tf.keras.layers.Dense(
            output_filters,
            use_bias=self.use_bias,
            name=_N(f"{self.name}/fc2"),
        )

        self.drop2 = tf.keras.layers.Dropout(
            rate=self.dropout_rate,
            name="drop2",
        )

        super().build(input_shape)

    
    def call(self, inputs, training=None):

        x = inputs

        x_gate = self.fc1_g(x)
        x = self.fc1_x(x)
        x = self.activation(x_gate) * x
        x = self.drop1(x, training=training)

        x = self.norm(x)

        x = self.fc2(x)
        x = self.drop2(x, training=training)

        return x


