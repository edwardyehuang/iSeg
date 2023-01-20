import tensorflow as tf

from iseg.layers.normalizations import normalization


class SqueezeAndExcitationModule(tf.keras.Model):
    def __init__(self, ratio=16, activation=tf.nn.relu, name=None):
        super().__init__(name=name)

        self.ratio = ratio
        self.activation = activation

    def build(self, input_shape):

        filters = input_shape[-1]

        self.down_conv = tf.keras.layers.Conv2D(int(filters / self.ratio), (1, 1), use_bias=True, name="down_conv")
        self.expand_conv = tf.keras.layers.Conv2D(filters, (1, 1), use_bias=True, name="expand_conv")
        

    def call(self, inputs, training=None):

        x = inputs

        x = tf.math.reduce_mean(x, axis=[1, 2], keepdims=True)
        x = self.down_conv(x)
        x = self.activation(x)

        x = self.expand_conv(x)
        x = tf.nn.sigmoid(x)
        x = tf.cast(x, inputs.dtype)

        return x * inputs