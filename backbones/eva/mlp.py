# ====================================================================
# MIT License
# Copyright (c) 2024 edwardyehuang (https://github.com/edwardyehuang)
# ====================================================================

import tensorflow as tf

LAYER_NORM_EPSILON = 1e-6

class Mlp(tf.keras.Model):

    def __init__(
        self,
        hidden_filters=None,
        output_filters=None,
        activation=tf.nn.gelu,
        use_bias=True,
        use_norm=True,
        dropout_rate=0.0,
        use_conv=False,
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

        self.use_conv = use_conv


    def build_fc (self, filters, name=None):

        if self.use_conv:
            return tf.keras.layers.Conv2D(
                filters,
                kernel_size=1,
                use_bias=self.use_bias,
                name=name,
            )
        else:
            return tf.keras.layers.Dense(
                filters,
                use_bias=self.use_bias,
                name=name,
            )


    def build(self, input_shape):

        input_channels = input_shape[-1]

        output_filters = self.output_filters or input_channels
        hidden_filters = self.hidden_filters or input_channels

        self.fc1 = self.build_fc(hidden_filters, name=f"{self.name}/fc1")

        self.drop1 = tf.keras.layers.Dropout(
            rate=self.dropout_rate,
            name="drop1",
        )

        if self.use_norm:
            self.norm = tf.keras.layers.LayerNormalization(
                epsilon=LAYER_NORM_EPSILON,
                name=f"{self.name}/norm",
            )
        else:
            self.norm = tf.identity

        self.fc2 = self.build_fc(output_filters, name=f"{self.name}/fc2")

        self.drop2 = tf.keras.layers.Dropout(
            rate=self.dropout_rate,
            name="drop2",
        )

        super().build(input_shape)

    
    def call(self, inputs, training=None):

        x = inputs

        x = self.fc1(x) # [N, H, W, C]
        x = self.activation(x)
        x = self.drop1(x, training=training)
        x = self.norm(x)
        x = self.fc2(x)
        x = self.drop2(x, training=training)

        return x