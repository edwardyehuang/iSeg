import tensorflow as tf

from iseg.utils import resize_image
from iseg.layers.model_builder import ConvBnRelu


class SimpleDecoder(tf.keras.Model):
    def __init__(self, low_level_filters=48, mlp_filters=256, name=None):

        super().__init__(name=name)

        self.low_level_filters = low_level_filters

        self.low_level_entry_conv = ConvBnRelu(self.low_level_filters, (1, 1), name="low_level_entry_conv")

        self.finetune_conv0 = ConvBnRelu(mlp_filters, (3, 3), name="finetune_conv0")
        self.finetune_conv1 = ConvBnRelu(mlp_filters, (3, 3), name="finetune_conv1")


    def call(self, inputs, training=None, **kwargs):

        low_level_features, result_features = tuple(inputs)

        low_level_features = tf.identity(low_level_features, name="low_level_features")
        low_level_features = self.low_level_entry_conv(low_level_features, training=training)

        x = resize_image(result_features, size=tf.shape(low_level_features)[1:3])
        x = tf.cast(x, dtype=low_level_features.dtype)

        x = tf.concat([low_level_features, x], axis=-1)

        x = self.finetune_conv0(x, training=training)
        x = self.finetune_conv1(x, training=training)

        return x
