import tensorflow as tf
import keras

from iseg.utils.keras3_utils import Keras3_Layer_Wrapper


class RMSNormalization(Keras3_Layer_Wrapper):
    def __init__(self, epsilon=1e-6, **kwargs):
        super().__init__(**kwargs)
        self.epsilon = epsilon

    def build(self, input_shape):
        self.scale = self.add_weight(
            name="scale",
            trainable=True,
            shape=(input_shape[-1],),
            initializer="zeros",
        )
        
        super().build(input_shape)

    def call(self, x):
        # Always compute normalization in float32.
        x = tf.cast(x, tf.float32)
        scale = tf.cast(self.scale, "float32")
        var = tf.reduce_mean(tf.square(x), axis=-1, keepdims=True)
        normed_inputs = x * tf.math.reciprocal(tf.sqrt(var + self.epsilon))
        normed_inputs = normed_inputs * (1 + scale)
        
        return tf.cast(normed_inputs, self.compute_dtype)