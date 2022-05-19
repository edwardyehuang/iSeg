# ================================================================
# MIT License
# Copyright (c) 2021 edwardyehuang (https://github.com/edwardyehuang)
# ================================================================

import tensorflow as tf
import copy
import math

from distutils.version import LooseVersion

if LooseVersion(tf.version.VERSION) < LooseVersion("2.7.0"):
    from tensorflow.python.keras.applications import imagenet_utils
else:
    from keras.applications import imagenet_utils

from iseg.layers.normalizations import normalization

DEFAULT_BLOCKS_ARGS = [
    {
        "kernel_size": 3,
        "repeats": 1,
        "filters_in": 32,
        "filters_out": 16,
        "expand_ratio": 1,
        "id_skip": True,
        "strides": 1,
        "se_ratio": 0.25,
    },
    {
        "kernel_size": 3,
        "repeats": 2,
        "filters_in": 16,
        "filters_out": 24,
        "expand_ratio": 6,
        "id_skip": True,
        "strides": 2,
        "se_ratio": 0.25,
    },
    {
        "kernel_size": 5,
        "repeats": 2,
        "filters_in": 24,
        "filters_out": 40,
        "expand_ratio": 6,
        "id_skip": True,
        "strides": 2,
        "se_ratio": 0.25,
    },
    {
        "kernel_size": 3,
        "repeats": 3,
        "filters_in": 40,
        "filters_out": 80,
        "expand_ratio": 6,
        "id_skip": True,
        "strides": 2,
        "se_ratio": 0.25,
    },
    {
        "kernel_size": 5,
        "repeats": 3,
        "filters_in": 80,
        "filters_out": 112,
        "expand_ratio": 6,
        "id_skip": True,
        "strides": 1,
        "se_ratio": 0.25,
    },
    {
        "kernel_size": 5,
        "repeats": 4,
        "filters_in": 112,
        "filters_out": 192,
        "expand_ratio": 6,
        "id_skip": True,
        "strides": 2,
        "se_ratio": 0.25,
    },
    {
        "kernel_size": 3,
        "repeats": 1,
        "filters_in": 192,
        "filters_out": 320,
        "expand_ratio": 6,
        "id_skip": True,
        "strides": 1,
        "se_ratio": 0.25,
    },
]

CONV_KERNEL_INITIALIZER = {
    "class_name": "VarianceScaling",
    "config": {"scale": 2.0, "mode": "fan_out", "distribution": "truncated_normal"},
}


def round_filters(filters, coefficient, divisor=8):
    """Round number of filters based on depth multiplier."""
    filters *= coefficient
    new_filters = max(divisor, int(filters + divisor / 2) // divisor * divisor)
    # Make sure that round down does not go down by more than 10%.
    if new_filters < 0.9 * filters:
        new_filters += divisor

    return int(new_filters)


def round_repeats(repeats, depth_coefficient):
    """Round number of repeats based on depth multiplier."""
    return int(math.ceil(depth_coefficient * repeats))


class Block(tf.keras.Model):
    def __init__(
        self,
        activation=tf.nn.swish,
        drop_rate=0,
        filters_in=32,
        filters_out=16,
        kernel_size=3,
        strides=1,
        expand_ratio=1,
        se_ratio=0,
        id_skip=True,
        name=None,
        **kwargs
    ):

        super(Block, self).__init__(name=name, **kwargs)

        self.strides = strides
        self.kernel_size = kernel_size

        self.activation = activation if activation is not None else tf.nn.swish

        self.filters_in = filters_in
        self.filters_out = filters_out
        self.drop_rate = drop_rate
        self.id_skip = id_skip

        self.output_endpoint = strides > 1

        filters = filters_in * expand_ratio

        if expand_ratio != 1:
            self.expand_conv = tf.keras.layers.Conv2D(
                filters=filters,
                kernel_size=1,
                padding="same",
                use_bias=False,
                kernel_initializer=CONV_KERNEL_INITIALIZER,
                name=name + "expand_conv",
            )

            self.expand_conv_bn = normalization(name=name + "expand_bn")
        else:
            self.expand_conv = None

        conv_pad = "valid" if self.strides == 2 else "same"

        self.dwconv = tf.keras.layers.DepthwiseConv2D(
            kernel_size=kernel_size,
            strides=strides,
            padding=conv_pad,
            use_bias=False,
            depthwise_initializer=CONV_KERNEL_INITIALIZER,
            name=name + "dwconv",
        )
        self.dwconv_bn = normalization(name=name + "bn")

        if 0 < se_ratio <= 1:
            filters_se = max(1, int(filters_in * se_ratio))

            self.se_reduce = tf.keras.layers.Conv2D(
                filters=filters_se,
                kernel_size=1,
                padding="same",
                kernel_initializer=CONV_KERNEL_INITIALIZER,
                name=name + "se_reduce",
            )
            self.se_expand = tf.keras.layers.Conv2D(
                filters=filters,
                kernel_size=1,
                padding="same",
                kernel_initializer=CONV_KERNEL_INITIALIZER,
                name=name + "se_expand",
            )
        else:
            self.se_reduce = None

        self.project_conv = tf.keras.layers.Conv2D(
            filters=filters_out,
            kernel_size=1,
            padding="same",
            use_bias=False,
            kernel_initializer=CONV_KERNEL_INITIALIZER,
            name=name + "project_conv",
        )
        self.project_bn = normalization(name=name + "project_bn")

        if drop_rate > 0:
            self.dropout = tf.keras.layers.Dropout(drop_rate, noise_shape=(None, 1, 1, 1))

    def call(self, inputs, training=None):

        x = inputs

        current_strides = self.dwconv.strides[0]

        if self.expand_conv is not None:
            x = self.expand_conv(x)
            x = self.expand_conv_bn(x, training=training)
            x = self.activation(x)

        if current_strides == 2:
            padding = imagenet_utils.correct_pad(x, self.kernel_size)
            x = tf.keras.backend.spatial_2d_padding(x, padding)

        x = self.dwconv(x)
        x = self.dwconv_bn(x, training=training)
        x = self.activation(x)

        if self.se_reduce is not None:

            se = tf.reduce_mean(x, (1, 2), keepdims=True, name=self.name + "se_squeeze")

            se = self.se_reduce(se)
            se = self.activation(se)

            se = self.se_expand(se)
            se = tf.nn.sigmoid(se)

            x = tf.multiply(x, se, name=self.name + "se_excite")

        x = self.project_conv(x)
        x = self.project_bn(x, training=training)

        if self.id_skip and current_strides == 1 and self.filters_in == self.filters_out:

            if self.drop_rate > 0:
                x = self.dropout(x, training=training)

            x = tf.add(x, inputs, name=self.name + "add")

        return x


class EfficientNet(tf.keras.Model):
    def __init__(
        self,
        width_confficient,
        depth_confficient,
        drop_connect_rate=0.2,
        depth_divisor=8,
        activation=tf.nn.swish,
        blocks_args="default",
        return_endpoints=False,
        name="efficientnet",
        **kwargs
    ):

        super(EfficientNet, self).__init__(name=name)

        self.return_endpoints = return_endpoints

        if blocks_args == "default":
            blocks_args = DEFAULT_BLOCKS_ARGS

        blocks_args = copy.deepcopy(blocks_args)

        self.activation = activation if activation is not None else tf.nn.swish

        self.stem_conv = tf.keras.layers.Conv2D(
            filters=round_filters(32, width_confficient, depth_divisor),
            kernel_size=3,
            strides=2,
            padding="valid",
            use_bias=False,
            kernel_initializer=CONV_KERNEL_INITIALIZER,
            name="stem_conv",
        )

        self.steam_conv_bn = normalization(name="stem_bn")

        b = 0

        self.blocks = []

        blocks_num = float(sum(round_repeats(args["repeats"], depth_confficient) for args in blocks_args))

        for (i, args) in enumerate(blocks_args):
            assert args["repeats"] > 0

            args["filters_in"] = round_filters(args["filters_in"], width_confficient, depth_divisor)
            args["filters_out"] = round_filters(args["filters_out"], width_confficient, depth_divisor)

            for j in range(round_repeats(args.pop("repeats"), depth_confficient)):
                if j > 0:
                    args["strides"] = 1
                    args["filters_in"] = args["filters_out"]

                block = Block(
                    activation=activation,
                    drop_rate=drop_connect_rate * b / blocks_num,
                    name="block{}{}_".format(i + 1, chr(j + 97)),
                    **args
                )

                self.blocks.append(block)

                b += 1

        self.top_conv = tf.keras.layers.Conv2D(
            filters=round_filters(1280, width_confficient, depth_divisor),
            kernel_size=1,
            padding="same",
            use_bias=False,
            kernel_initializer=CONV_KERNEL_INITIALIZER,
            name="top_conv",
        )

        self.top_bn = normalization(name="top_bn")

    def call(self, inputs, training=None, **kwargs):

        endpoints = []

        x = inputs

        x = tf.keras.backend.spatial_2d_padding(x, imagenet_utils.correct_pad(x, 3))

        x = self.stem_conv(x)
        x = self.steam_conv_bn(x, training=training)
        x = self.activation(x)

        for block in self.blocks:

            if block.output_endpoint:
                endpoints += [x]

            x = block(x, training=training)

        x = self.top_conv(x)
        x = self.top_bn(x, training=training)
        x = self.activation(x)

        endpoints += [x]

        if self.return_endpoints:
            return endpoints
        else:
            return x


def EfficientNetB0(return_endpoints=False):

    return EfficientNet(
        width_confficient=1.0,
        depth_confficient=1.0,
        default_size=224,
        drop_connect_rate=0.2,
        return_endpoints=return_endpoints,
        name="efficientnetb0",
    )


def EfficientNetB1(return_endpoints=False):

    return EfficientNet(
        width_confficient=1.0,
        depth_confficient=1.1,
        default_size=240,
        drop_connect_rate=0.2,
        return_endpoints=return_endpoints,
        name="efficientnetb1",
    )


def EfficientNetB2(return_endpoints=False):

    return EfficientNet(
        width_confficient=1.1,
        depth_confficient=1.2,
        default_size=260,
        drop_connect_rate=0.3,
        return_endpoints=return_endpoints,
        name="efficientnetb2",
    )


def EfficientNetB3(return_endpoints=False):

    return EfficientNet(
        width_confficient=1.2,
        depth_confficient=1.4,
        default_size=300,
        drop_connect_rate=0.3,
        return_endpoints=return_endpoints,
        name="efficientnetb3",
    )


def EfficientNetB4(return_endpoints=False):

    return EfficientNet(
        width_confficient=1.4,
        depth_confficient=1.8,
        default_size=380,
        drop_connect_rate=0.4,
        return_endpoints=return_endpoints,
        name="efficientnetb4",
    )


def EfficientNetB5(return_endpoints=False):

    return EfficientNet(
        width_confficient=1.6,
        depth_confficient=2.2,
        default_size=456,
        drop_connect_rate=0.4,
        return_endpoints=return_endpoints,
        name="efficientnetb5",
    )


def EfficientNetB6(return_endpoints=False):

    return EfficientNet(
        width_confficient=1.8,
        depth_confficient=2.6,
        default_size=528,
        drop_connect_rate=0.5,
        return_endpoints=return_endpoints,
        name="efficientnetb6",
    )


def EfficientNetB7(return_endpoints=False):

    return EfficientNet(
        width_confficient=2.0,
        depth_confficient=3.1,
        default_size=600,
        drop_connect_rate=0.5,
        return_endpoints=return_endpoints,
        name="efficientnetb7",
    )


def EfficientNetL2(return_endpoints=False):

    return EfficientNet(
        width_confficient=4.3,
        depth_confficient=5.3,
        default_size=800,
        drop_connect_rate=0.5,
        return_endpoints=return_endpoints,
        name="efficientnetl2",
    )


def build_dilated_efficientnet(efficientnet, output_stride=16):

    current_os = 2
    current_dilation = 1

    for block in efficientnet.blocks:

        if current_os >= output_stride:
            current_dilation *= block.dwconv.strides[0]
            block.dwconv.strides = (1, 1)
            block.dwconv.padding = "same"
            block.dwconv.dilation_rate = (current_dilation, current_dilation)
        else:
            current_os *= block.dwconv.strides[0]

    return efficientnet
