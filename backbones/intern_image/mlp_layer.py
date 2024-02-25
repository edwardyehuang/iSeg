# ===================================================================
# MIT License
# Copyright (c) 2024 edwardyehuang (https://github.com/edwardyehuang)
# ===================================================================

import tensorflow as tf

from iseg.utils.keras3_utils import Keras3_Model_Wrapper

class MLPLayer(Keras3_Model_Wrapper):

    def __init__(
        self,
        hidden_filters=None,
        out_filters=None,
        activation=tf.nn.gelu,
        dropout_rate=0.,
        name=None
    ):
        super().__init__(name=name)

        self.hidden_filters = hidden_filters
        self.out_filters = out_filters
        self.activation = activation

        self.dropout_rate = dropout_rate
    

    def build(self, input_shape):

        input_channels = input_shape[-1]
        
        self.fc1 = tf.keras.layers.Dense(
            self.hidden_filters, name=f"{self.name}/fc1",
        )

        self.fc2 = tf.keras.layers.Dense(
            self.out_filters or input_channels, name=f"{self.name}/fc2"
        )

        self.dropout = tf.keras.layers.Dropout(
            self.dropout_rate, name=f"{self.name}/dropout"
        )
        
        super().build(input_shape)

    
    def call(self, inputs, training=False):

        x = self.fc1(inputs)
        x = self.activation(x)

        x = self.dropout(x, training=training)

        x = self.fc2(x)

        x = self.dropout(x, training=training)

        return x