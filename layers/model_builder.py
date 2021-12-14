# ================================================================
# MIT License
# Copyright (c) 2021 edwardyehuang (https://github.com/edwardyehuang)
# ================================================================

import tensorflow as tf
import iseg.static_strings as ss

from iseg.core_model import SegFoundation
from iseg.layers.normalizations import normalization
from iseg.utils.common import resize_image


def get_training_value(training=None):

    if training is None:
        training = tf.keras.backend.learning_phase()

    if isinstance(training, int):
        training = bool(training)

    return training


class ConvBnRelu(tf.keras.Model):
    def __init__(
        self,
        filters=256,
        kernel_size=1,
        dilation_rate=(1, 1),
        use_bn=True,
        activation=tf.nn.relu,
        conv_kernel_initializer="glorot_uniform",
        dropout_rate=0,
        trainable=True,
        use_bias=False,
        name=None,
    ):

        super(ConvBnRelu, self).__init__(trainable=trainable, name=name if name is not None else "ConvBnRelu")

        self.conv = tf.keras.layers.Conv2D(
            filters,
            kernel_size,
            padding="same",
            use_bias=use_bias,
            kernel_initializer=conv_kernel_initializer,
            dilation_rate=dilation_rate,
            trainable=trainable,
            name="{}_conv".format(name),
        )

        self.bn = None if not use_bn else normalization(trainable=trainable, name="{}_bn".format(name))

        self.activation = activation

        if self.activation is False:
            self.activation = None

        self.dropout = None

        if dropout_rate > 0:
            self.dropout = tf.keras.layers.Dropout(dropout_rate, name="{}_dropout".format(name))

    def call(self, inputs, training=None):

        x = self.conv(inputs)

        if self.bn is not None:
            x = self.bn(x, training=training)

        if self.activation is not None:
            x = self.activation(x)

        if self.dropout is not None and self.trainable:
            x = self.dropout(x, training=training)

        return x

    def reset_weights(self):

        self.__reset_weight(self.conv.kernel, self.conv.kernel_initializer)

        if self.conv.use_bias:
            self.__reset_weight(self.conv.bias, self.conv.bias_initializer)

        if self.bn is not None:
            self.__reset_weight(self.bn.beta, self.bn.beta_initializer)
            self.__reset_weight(self.bn.gamma, self.bn.gamma_initializer)
            self.__reset_weight(self.bn.moving_mean, self.bn.moving_mean_initializer)
            self.__reset_weight(self.bn.moving_variance, self.bn.moving_variance_initializer)

    def __reset_weight(self, weight, initializer):

        weight.assign(initializer(weight.shape, weight.dtype))


class LnConvGelu(tf.keras.Model):
    def __init__(
        self,
        filters=256,
        kernel_size=1,
        dilation_rate=(1, 1),
        use_ln=True,
        activation=tf.nn.gelu,
        conv_kernel_initializer="glorot_uniform",
        dropout_rate=0,
        trainable=True,
        use_bias=True,
        name=None,
    ):

        super().__init__(trainable=trainable, name=name)

        self.ln = (
            None if not use_ln else tf.keras.layers.LayerNormalization(trainable=trainable, name=f"{self.name}_ln")
        )

        self.conv = tf.keras.layers.Conv2D(
            filters,
            kernel_size,
            padding="same",
            use_bias=use_bias,
            kernel_initializer=conv_kernel_initializer,
            dilation_rate=dilation_rate,
            trainable=trainable,
            activation=activation,
            name=f"{self.name}_conv",
        )

    def call(self, inputs, training=None):

        x = inputs

        if self.ln is not None:
            x = self.ln(x)

        x = self.conv(x)

        return x

    def reset_weights(self):

        self.__reset_weight(self.conv.kernel, self.conv.kernel_initializer)

        if self.conv.use_bias:
            self.__reset_weight(self.conv.bias, self.conv.bias_initializer)

        if self.ln is not None:
            self.__reset_weight(self.ln.beta, self.ln.beta_initializer)
            self.__reset_weight(self.ln.gamma, self.ln.gamma_initializer)

    def __reset_weight(self, weight, initializer):

        weight.assign(initializer(weight.shape, weight.dtype))


class ImageLevelBlock(tf.keras.Model):
    def __init__(self, filters=256, pooling_axis=(1, 2), name=None):
        super(ImageLevelBlock, self).__init__(name="ImageLevelBlock" if name is None else name)

        self.convbnrelu = ConvBnRelu(filters, (1, 1), name="conv")
        self.pooling_axis = pooling_axis

    def call(self, inputs, training=None):

        x = inputs
        inputs_dtype = inputs.dtype
        inputs_size = tf.shape(inputs)[1:3]

        x = tf.reduce_mean(x, axis=self.pooling_axis, keepdims=True, name="pool")
        x = self.convbnrelu(x, training=training)
        x = tf.ones([1, inputs_size[0], inputs_size[1], 1], dtype=x.dtype) * x
        x = tf.cast(x, inputs_dtype)

        # x = tf.ensure_shape(x, [None, inputs_size[0], inputs_size[1], x.shape[-1]])

        return x


class CommonEndBlock(tf.keras.Model):
    def __init__(self, filters=256, num_class=21, dropout_rate=0.1, name=None):

        super().__init__(name=name)

        self.filters = filters
        self.num_class = num_class
        self.dropout_rate = dropout_rate

    def build(self, input_shape):

        self.end_conv = ConvBnRelu(self.filters, dropout_rate=self.dropout_rate, name="end_conv")
        self.logits_conv = tf.keras.layers.Conv2D(self.num_class, (1, 1), name="logits_conv")

    def call(self, inputs, training=False):
        x, orginal_inputs = inputs
        x = self.end_conv(x, training=training)
        x = self.logits_conv(x)
        x = resize_image(x, tf.shape(orginal_inputs)[1:3])

        return tf.cast(x, tf.float32)


def parallel_map(fn, elements, dtype, use_map_fn=False, parallel_iterations=8):

    if not use_map_fn:
        return tf.vectorized_map(fn, elements)
    else:
        return tf.map_fn(fn, elements, dtype, parallel_iterations=parallel_iterations)


def drop_connect(inputs, drop_connect_rate, training=None):

    training = get_training_value(training)

    if not training:
        return inputs

    keep_prob = 1.0 - drop_connect_rate

    # Compute drop_connect tensor
    batch_size = tf.shape(inputs)[0]
    random_tensor = keep_prob
    random_tensor += tf.random_uniform([batch_size, 1, 1, 1], dtype=inputs.dtype)
    binary_tensor = tf.floor(random_tensor)
    output = tf.div(inputs, keep_prob) * binary_tensor
    return output


def get_tensor_shape(x, static_dims=[-1]):

    if not isinstance(static_dims, (list, tuple)):
        static_dims = [static_dims]

    for i in range(len(static_dims)):
        if static_dims[i] < 0:
            static_dims[i] += len(x.shape)

    dynamic_shape = tf.shape(x)

    results = []

    for i in range(len(x.shape)):

        s = dynamic_shape[i]

        if i in static_dims:
            s = x.shape[i]

        results.append(s)

    return tuple(results)
