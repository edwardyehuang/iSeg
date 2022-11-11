# coding=utf-8
# Copyright 2022 The Deeplab2 Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.


import math
import tensorflow as tf
from .attention import Attention


def drop_connect(inputs: tf.Tensor, training: bool, survival_prob: float) -> tf.Tensor:
    """Drops the entire conv with given survival probability [1].

    [1] Deep Networks with Stochastic Depth,
            ECCV 2016.
                Gao Huang, Yu Sun, Zhuang Liu, Daniel Sedra, Kilian Q. Weinberger.

    Args:
        inputs: A tensor with shape [batch_size, height, width, channels].
        training: A boolen, whether in training mode or not.
        survival_prob: A float, 1 - drop_path_rate [1].

    Returns:
        output: A tensor with shape [batch_size, height, width, channels]
    """

    if not training:
        return inputs
    batch_size = tf.shape(inputs)[0]
    random_tensor = survival_prob
    random_tensor += tf.random.uniform([batch_size], dtype=inputs.dtype)
    for _ in range(inputs.shape.rank - 1):
        random_tensor = tf.expand_dims(random_tensor, axis=-1)
    binary_tensor = tf.floor(random_tensor)
    # Unlike the conventional way that we multiply survival_prob at test time, we
    # divide survival_prob at training time, so no additional compute is needed at
    # test time.
    output = inputs / survival_prob * binary_tensor
    return output


def residual_add_with_drop_path(
    residual: tf.Tensor, 
    shortcut: tf.Tensor,
    survival_prob: float, 
    training: bool
) -> tf.Tensor:
    """Combines residual and shortcut."""
    if survival_prob is not None and 0 < survival_prob < 1:
        residual = drop_connect(residual, training, survival_prob)
    return shortcut + residual


class SqueezeAndExcitation(tf.keras.Model):
    """Implementation of Squeeze-and-excitation layer."""

    def __init__(
        self,
        se_filters,
        output_filters,
        activation=tf.keras.activations.swish,
        kernel_initializer=tf.random_normal_initializer(stddev=0.02),
        bias_initializer=tf.zeros_initializer,
        name="se",
    ):

        super().__init__(name=name)

        self._se_reduce = tf.keras.layers.Conv2D(
            se_filters,
            kernel_size=1,
            strides=1,
            padding='same',
            use_bias=True,
            kernel_initializer=kernel_initializer,
            bias_initializer=bias_initializer,
            name=f"{self.name}/reduce_conv2d"
        )
        self._se_expand = tf.keras.layers.Conv2D(
            output_filters,
            kernel_size=1,
            strides=1,
            padding='same',
            use_bias=True,
            kernel_initializer=kernel_initializer,
            bias_initializer=bias_initializer,
            name=f"{self.name}/expand_conv2d"
        )

        self.activation_fn = activation


    def build(self, input_shape):
        
        super().build(input_shape)


    def call(self, inputs: tf.Tensor) -> tf.Tensor:
        _ = inputs.get_shape().with_rank(4)
        se_tensor = tf.reduce_mean(inputs, [1, 2], keepdims=True)
        se_tensor = self._se_expand(self.activation_fn(self._se_reduce(se_tensor)))
        return tf.cast(tf.sigmoid(se_tensor), inputs.dtype) * inputs


class MBConvBlock(tf.keras.Model):

    def __init__(
        self,
        hidden_size,
        kernel_size=3,
        expansion_rate=4,
        se_ratio=0.25,
        block_stride=1,
        pool_size=2,
        norm_class=tf.keras.layers.experimental.SyncBatchNormalization,
        activation=tf.keras.activations.gelu,
        survival_prob=None,
        kernel_initializer=tf.random_normal_initializer(stddev=0.02),
        bias_initializer=tf.zeros_initializer,
        name="mbconv",
    ):
        super().__init__(name=name)
        self._activation_fn = activation
        self._norm_class = norm_class

        self.hidden_size = hidden_size
        self.kernel_size = kernel_size
        self.expansion_rate =expansion_rate
        self.se_ratio = se_ratio

        self.block_stride = block_stride
        self.pool_size = pool_size
        
        self.survival_prob = survival_prob
        self.kernel_initializer = kernel_initializer
        self.bias_initializer = bias_initializer

    def build(self, input_shape: list[int]) -> None:
        input_size = input_shape[-1]
        inner_size = self.hidden_size * self.expansion_rate

        self._shortcut_conv = None
        if input_size != self.hidden_size:
            self._shortcut_conv = tf.keras.layers.Conv2D(
                filters=self.hidden_size,
                kernel_size=1,
                strides=1,
                padding='same',
                kernel_initializer=self.kernel_initializer,
                bias_initializer=self.bias_initializer,
                use_bias=True,
                name=f"{self.name}/shortcut_conv"
            )

        self._pre_norm = self._norm_class(name=f"{self.name}/pre_norm")
        self._expand_conv = tf.keras.layers.Conv2D(
            filters=inner_size,
            kernel_size=1,
            strides=1,
            kernel_initializer=self.kernel_initializer,
            padding='same',
            use_bias=False,
            name=f"{self.name}/expand_conv"
        )
        self._expand_norm = self._norm_class(name=f"{self.name}/expand_norm")
        self._depthwise_conv = tf.keras.layers.DepthwiseConv2D(
            kernel_size=self.kernel_size,
            strides=self.block_stride,
            depthwise_initializer=self.kernel_initializer,
            padding='same',
            use_bias=False,
            name=f"{self.name}/depthwise_conv"
        )
        self._depthwise_norm = self._norm_class(name=f"{self.name}/depthwise_norm")

        self._se = None
        if self.se_ratio is not None:
            se_filters = max(1, int(self.hidden_size * self.se_ratio))
            self._se = SqueezeAndExcitation(
                se_filters=se_filters,
                output_filters=inner_size,
                kernel_initializer=self.kernel_initializer,
                bias_initializer=self.bias_initializer,
                name=f"{self.name}/se"
            )

        self._shrink_conv = tf.keras.layers.Conv2D(
            filters=self.hidden_size,
            kernel_size=1,
            strides=1,
            padding='same',
            kernel_initializer=self.kernel_initializer,
            bias_initializer=self.bias_initializer,
            use_bias=True,
            name=f"{self.name}/shrink_conv"
        )

        super().build(input_shape)


    def _shortcut_downsample(self, inputs, name):
        output = inputs
        if self.block_stride > 1:
            pooling_layer = tf.keras.layers.AveragePooling2D(
                pool_size=self.pool_size,
                strides=self.block_stride,
                padding='same',
                name=name,
            )
            if output.dtype == tf.float32:
                output = pooling_layer(output)
            else:
                # We find that in our code base, the output dtype of pooling is float32
                # no matter whether its input and compute dtype is bfloat16 or
                # float32. So we explicitly cast the output dtype of pooling to be the
                # model compute dtype.
                output = tf.cast(pooling_layer(
                        tf.cast(output, tf.float32)), output.dtype)
        return output

    def _shortcut_branch(self, inputs):
        shortcut = self._shortcut_downsample(inputs, name='shortcut_pool')
        if self._shortcut_conv:
            shortcut = self._shortcut_conv(shortcut)
        return shortcut

    def call(self, inputs: tf.Tensor, training: bool) -> tf.Tensor:
        shortcut = self._shortcut_branch(inputs)
        output = self._pre_norm(inputs, training=training)
        output = self._expand_conv(output)
        output = self._expand_norm(output, training=training)
        output = self._activation_fn(output)
        output = self._depthwise_conv(output)
        output = self._depthwise_norm(output, training=training)
        output = self._activation_fn(output)
        if self._se:
            output = self._se(output)
        output = self._shrink_conv(output)
        output = residual_add_with_drop_path(
                output, tf.cast(shortcut, output.dtype),
                self.survival_prob, training)
        return output


class MOATBlock(tf.keras.Model):
    """Implementation of MOAT block [1].

    [1] MOAT: Alternating Mobile Convolution and Attention
        Brings Strong Vision Models,
        arXiv: ----.----.
            Chenglin Yang, Siyuan Qiao, Qihang Yu, Xiaoding Yuan,
            Yukun Zhu, Alan Yuille, Hartwig Adam, Liang-Chieh Chen.
    """

    def __init__(
        self, 
        hidden_size,
        kernel_size=3,
        expansion_rate=4,
        block_stride=2,
        pool_size=2,
        norm_class=tf.keras.layers.experimental.SyncBatchNormalization,
        activation=tf.keras.activations.gelu,
        head_size=32,
        window_size=None,
        relative_position_embedding_type="2d_multi_head",
        position_embedding_size=7,
        ln_epsilon=1e-5,
        survival_prob=None,
        kernel_initializer= tf.random_normal_initializer(stddev=0.02),
        bias_initializer=tf.zeros_initializer,
        use_checkpointing_for_attention=False,
        name="moat",
    ):
        super().__init__(name=name)
        self._activation_fn = activation
        self._norm_class = norm_class

        self.hidden_size = hidden_size
        self.kernel_size = kernel_size
        self.expansion_rate = expansion_rate
        self.block_stride = block_stride
        self.pool_size = pool_size

        self.head_size = head_size
        self.window_size = window_size

        self.relative_position_embedding_type = relative_position_embedding_type
        self.position_embedding_size = position_embedding_size
        self.ln_epsilon = ln_epsilon
        self.survival_prob = survival_prob
        self.kernel_initializer = kernel_initializer
        self.bias_initializer = bias_initializer

        self.use_checkpointing_for_attention = use_checkpointing_for_attention


    def build(self, input_shape: list[int]) -> None:

        height, width, input_size = input_shape[-3:]
        inner_size = self.hidden_size * self.expansion_rate

        if self.window_size:
            if (isinstance(self.window_size, list) and
                    len(self.window_size)) == 2:
                self._window_height = self.window_size[0]
                self._window_width = self.window_size[1]
            else:
                raise ValueError((
                        'The window size should be a list of two ints',
                        '[height, width], if specified.'))
        else:
            self._window_height = math.ceil(float(height) / self.block_stride)
            self._window_width = math.ceil(float(width) / self.block_stride)

        self._shortcut_conv = None
        if input_size != self.hidden_size:
            self._shortcut_conv = tf.keras.layers.Conv2D(
                filters=self.hidden_size,
                kernel_size=1,
                strides=1,
                padding='same',
                kernel_initializer=self.kernel_initializer,
                bias_initializer=self.bias_initializer,
                use_bias=True,
                name=f"{self.name}/shortcut_conv"
            )

        self._pre_norm = self._norm_class(name=f"{self.name}/pre_norm")
        self._expand_conv = tf.keras.layers.Conv2D(
            filters=inner_size,
            kernel_size=1,
            strides=1,
            kernel_initializer=self.kernel_initializer,
            padding='same',
            use_bias=False,
            name=f"{self.name}/expand_conv",
        )
        self._expand_norm = self._norm_class(name=f"{self.name}/expand_norm")
        self._depthwise_conv = tf.keras.layers.DepthwiseConv2D(
            kernel_size=self.kernel_size,
            strides=self.block_stride,
            depthwise_initializer=self.kernel_initializer,
            padding='same',
            use_bias=False,
            name=f"{self.name}/depthwise_conv",
        )
        self._depthwise_norm = self._norm_class(name=f"{self.name}/depthwise_norm")
        self._shrink_conv = tf.keras.layers.Conv2D(
            filters=self.hidden_size,
            kernel_size=1,
            strides=1,
            padding='same',
            kernel_initializer=self.kernel_initializer,
            bias_initializer=self.bias_initializer,
            use_bias=True,
            name=f"{self.name}/shrink_conv",
        )

        self._attention_norm = tf.keras.layers.LayerNormalization(
            axis=-1,
            epsilon=self.ln_epsilon,
            name=f"{self.name}/attention_norm",
        )

        scale_ratio = None
        if self.relative_position_embedding_type:
            if self.position_embedding_size is None:
                raise ValueError(
                    'The position embedding size need to be specified ' +
                    'if relative position embedding is used.'
                )

            scale_ratio = [
                self._window_height / self.position_embedding_size,
                self._window_width / self.position_embedding_size,
            ]

        self._attention = Attention(
            hidden_size=self.hidden_size,
            head_size=self.head_size,
            relative_position_embedding_type=(
                    self.relative_position_embedding_type),
            scale_ratio=scale_ratio,
            kernel_initializer=self.kernel_initializer,
            bias_initializer=self.bias_initializer,
            name=f"{self.name}/attention"
        )

        super().build(input_shape)
        

    def _make_windows(self, inputs):
        _, height, width, channels = inputs.get_shape().with_rank(4).as_list()

        inputs = tf.reshape(
            inputs,
            (
                -1,
                height // self._window_height, self._window_height,
                width // self._window_width, self._window_width,
                channels
            )
        )
        inputs = tf.transpose(inputs, (0, 1, 3, 2, 4, 5))
        inputs = tf.reshape(
            inputs,
            (-1, self._window_height, self._window_width, channels)
        )
        return inputs

    def _remove_windows(self, inputs, height, width):
        _, _, channels = inputs.get_shape().with_rank(3).as_list()
        inputs = tf.reshape(inputs, [
                -1, height // self._window_height, width // self._window_width,
                self._window_height, self._window_width, channels
        ])
        inputs = tf.transpose(inputs, (0, 1, 3, 2, 4, 5))
        inputs = tf.reshape(inputs, (-1, height, width, channels))
        return inputs

    def _shortcut_downsample(self, inputs, name):
        output = inputs
        if self.block_stride > 1:
            pooling_layer = tf.keras.layers.AveragePooling2D(
                pool_size=self.pool_size,
                strides=self.block_stride,
                padding='same',
                name=name,
            )

            if output.dtype == tf.float32:
                output = pooling_layer(output)
            else:
                # We find that in our code base, the output dtype of pooling is float32
                # no matter whether its input and compute dtype is bfloat16 or
                # float32. So we explicitly cast the output dtype of pooling to be the
                # model compute dtype.
                output = tf.cast(
                    pooling_layer(
                        tf.cast(output, tf.float32)
                    ), 
                    output.dtype
                )

        return output

    def _shortcut_branch(self, inputs):
        shortcut = self._shortcut_downsample(inputs, name='shortcut_pool')
        if self._shortcut_conv:
            shortcut = self._shortcut_conv(shortcut)
        return shortcut

    def call(self, inputs: tf.Tensor, training: bool) -> tf.Tensor:
        mbconv_shortcut = self._shortcut_branch(inputs)
        output = self._pre_norm(inputs, training=training)
        output = self._expand_conv(output)
        output = self._expand_norm(output, training=training)
        output = self._activation_fn(output)
        output = self._depthwise_conv(output)
        output = self._depthwise_norm(output, training=training)
        output = self._activation_fn(output)
        output = self._shrink_conv(output)
        output = residual_add_with_drop_path(
                output, tf.cast(mbconv_shortcut, output.dtype),
                self.survival_prob, training)

        # For classification, the window size is the same as feature map size.
        # For downstream tasks, the window size can be set the same as
        # classification's.
        attention_shortcut = output
        def _func(output):
            output = self._attention_norm(output)
            _, height, width, channels = output.get_shape().with_rank(4).as_list()

            if self.window_size:
                output = self._make_windows(output)
            output = self._attention(output)

            if self.window_size:
                output = self._remove_windows(output, height, width)
            else:
                output = tf.reshape(output, [-1, height, width, channels])

            return output

        func = _func
        if self.use_checkpointing_for_attention:
            func = tf.recompute_grad(_func)

        output = func(output)
        output = residual_add_with_drop_path(
                output, attention_shortcut,
                self.survival_prob, training)
        return output