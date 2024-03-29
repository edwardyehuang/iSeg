# ================================================================
# MIT License
# Copyright (c) 2021 edwardyehuang (https://github.com/edwardyehuang)
# ================================================================

# This code is motified from https://github.com/keras-team/keras/blob/master/keras/applications/resnet.py
# The modifications are refer to https://github.com/tensorflow/models/blob/master/research/deeplab/core/resnet_v1_beta.py
# and "Bag of Tricks for Image Classification with Convolutional Neural Networks", CVPR2019

import iseg.static_strings as ss
import tensorflow as tf
from iseg.layers.normalizations import normalization

from iseg.backbones.resnet_blocks import BlockType1, BlockType2
from iseg.backbones.resnet_blocks_small import BlockType2Small

from iseg.backbones.utils.layerwise_decay import decay_layers_lr
from iseg.utils.keras3_utils import Keras3_Model_Wrapper

BN_EPSILON = 1.001e-5

DEFAULT_CONV_FUNC = tf.keras.layers.Conv2D


class Stack(Keras3_Model_Wrapper):
    def __init__(
        self, 
        filters, 
        blocks_count, 
        stride1=2, 
        use_bias=True, 
        norm_method=None, 
        custom_block=None, 
        conv_func=DEFAULT_CONV_FUNC,
        name=None
    ):

        super(Stack, self).__init__(name=name)

        block_func = BlockType1 if custom_block is None else custom_block

        self.blocks = []

        self.blocks.append(
            block_func(
                filters, 
                stride=stride1, 
                use_bias=use_bias, 
                norm_method=norm_method, 
                conv_func=conv_func, 
                name=name + "_block1"
            )
        )

        for i in range(2, blocks_count + 1):
            block = block_func(
                filters, 
                conv_shortcut=False, 
                use_bias=use_bias, 
                norm_method=norm_method, 
                conv_func=conv_func, 
                name=name + "_block" + str(i)
            )
            self.blocks.append(block)

        self.output_endpoint = stride1 > 1

    @property
    def strides(self):
        return self.blocks[0].strides
    

    def build(self, input_shape):
        super().build(input_shape)


    def call(self, inputs, training=None, **kwargs):

        x = inputs

        x_before_stride = tf.identity(x, name="before_stride")

        x = self.blocks[0](x, training=training)

        for block in self.blocks[1:]:
            x = block(x, training=training)

        if self.output_endpoint:
            return x, x_before_stride
        else:
            return x


class Stack2(Keras3_Model_Wrapper):
    def __init__(
        self, 
        filters, 
        blocks_count, 
        stride1=2, 
        use_bias=True, 
        norm_method=None, 
        custom_block=None, 
        conv_func=DEFAULT_CONV_FUNC,
        name=None
    ):

        super(Stack2, self).__init__(name=name)

        block_func = BlockType2 if custom_block is None else custom_block

        self.blocks = []

        if blocks_count > 1:
            self.blocks.append(
                block_func(
                    filters, 
                    stride=1, 
                    conv_shortcut=True, 
                    use_bias=use_bias, 
                    norm_method=norm_method, 
                    conv_func=conv_func, 
                    name=name + "_block1"
                )
            )

            # 2022-08-15 This is correct. But it scared me when I checked it.
            for i in range(2, blocks_count):
                block = block_func(
                    filters, 
                    conv_shortcut=False, 
                    use_bias=use_bias, 
                    norm_method=norm_method, 
                    conv_func=conv_func, 
                    name=name + "_block" + str(i)
                )
                self.blocks.append(block)

            self.blocks.append(
                block_func(
                    filters,
                    stride=stride1,
                    conv_shortcut=False,
                    use_bias=use_bias,
                    norm_method=norm_method,
                    conv_func=conv_func,
                    name=name + "_block" + str(blocks_count),
                )
            )
        else:
            self.blocks = [block_func(
                    filters, stride=stride1, 
                    conv_shortcut=True, 
                    use_bias=use_bias, 
                    norm_method=norm_method, 
                    conv_func=conv_func, 
                    name=name + "_block1"
                )]

        assert len(self.blocks) == blocks_count

        self.output_endpoint = stride1 > 1

    @property
    def strides(self):
        return self.blocks[-1].strides
    
    def build(self, input_shape):
        super().build(input_shape)

    def call(self, inputs, training=None, **kwargs):

        x = inputs

        for block in self.blocks[:-1]:
            x = block(x, training=training)

        x_before_stride = tf.identity(x, name="before_stride")

        x = self.blocks[-1](x, training=training)

        if self.output_endpoint:
            return x, x_before_stride
        else:
            return x


class ResNet(Keras3_Model_Wrapper):
    def __init__(
        self, 
        stacks, 
        use_bias=True, 
        norm_method=None,
        conv1_depth_multiplier=1,
        replace_7x7_conv=False, 
        return_endpoints=False, 
        conv_func=DEFAULT_CONV_FUNC,
        name="resnet"
    ):

        super(ResNet, self).__init__(name=name)

        self.replace_7x7_conv = replace_7x7_conv
        self.conv_func = conv_func

        conv1_fn = self.build_3x3_resnet if self.replace_7x7_conv else self.build_7x7_resnet
        conv1_fn(
            depth_multiplier=conv1_depth_multiplier,
            use_bias=use_bias, 
            norm_method=norm_method
            )

        if not replace_7x7_conv:
            self.poo1_pad = tf.keras.layers.ZeroPadding2D(padding=(1, 1), name="poo1_pad")

        self.poo1_pool = tf.keras.layers.MaxPooling2D(
            3, strides=2, padding="same" if replace_7x7_conv else "valid", name="pool1_pool"
        )

        self.stacks = stacks

        self.return_endpoints = return_endpoints

    ### ResNet 7x7 ###################

    def build_7x7_resnet(self, depth_multiplier=1, use_bias=True, norm_method=None):

        self.conv1_pad = tf.keras.layers.ZeroPadding2D(padding=(3, 3), name="conv1_pad")
        self.conv1_conv = self.conv_func(
            int(64 * depth_multiplier), 7, strides=2, use_bias=use_bias, name="conv1_conv"
            )
        self.conv1_bn = normalization(epsilon=BN_EPSILON, method=norm_method, name="conv1_bn")


    def compute_7x7_resnet(self, inputs, training=None, **kwargs):

        x = self.conv1_pad(inputs)
        x = self.conv1_conv(x)
        x = self.conv1_bn(x, training=training)
        x = tf.nn.relu(x)

        return x

    ### ResNet 3x3 ###################

    def build_3x3_resnet(self, depth_multiplier=1, use_bias=True, norm_method=None):

        if isinstance(depth_multiplier, tuple):
            depth_multiplier = list(depth_multiplier)

        if not isinstance(depth_multiplier, list):
            depth_multiplier = [depth_multiplier] * 3

        self.conv1_1_conv = self.conv_func(
            int(64 * depth_multiplier[0]), 3, strides=2, padding="SAME", use_bias=use_bias, name="conv1_1_conv"
        )
        self.conv1_1_bn = normalization(epsilon=BN_EPSILON, method=norm_method, name="conv1_1_bn")

        self.conv1_2_conv = self.conv_func(
            int(64 * depth_multiplier[1]), 3, strides=1, padding="SAME", use_bias=use_bias, name="conv1_2_conv"
        )
        self.conv1_2_bn = normalization(epsilon=BN_EPSILON, method=norm_method, name="conv1_2_bn")

        self.conv1_3_conv = self.conv_func(
            int(128 * depth_multiplier[2]), 3, strides=1, padding="SAME", use_bias=use_bias, name="conv1_3_conv"
        )
        self.conv1_3_bn = normalization(epsilon=BN_EPSILON, method=norm_method, name="conv1_3_bn")


    def compute_3x3_resnet(self, inputs, training=None, **kwargs):

        x = self.conv1_1_conv(inputs)
        x = self.conv1_1_bn(x, training=training)
        x = tf.nn.relu(x)

        x = self.conv1_2_conv(x)
        x = self.conv1_2_bn(x, training=training)
        x = tf.nn.relu(x)

        x = self.conv1_3_conv(x)
        x = self.conv1_3_bn(x, training=training)
        x = tf.nn.relu(x)

        return x
    

    def decay_lr(
        self, 
        rate=0.99
    ):

        stages = list(self.stacks)

        if self.replace_7x7_conv:
            stems = [
                self.conv1_1_conv, 
                self.conv1_1_bn,
                self.conv1_2_conv, 
                self.conv1_2_bn,
                self.conv1_3_conv, 
                self.conv1_3_bn,
            ]
        else:
            stems = [self.conv1_conv, self.conv1_bn]

        stages = stems + stages

        stages.reverse()
        decay_layers_lr(stages, rate=rate)


    def build(self, input_shape):
        super().build(input_shape)


    ### ResNet Call ###################

    def call(self, inputs, training=None, **kwargs):

        endpoints = []

        conv1_fn = self.compute_3x3_resnet if self.replace_7x7_conv else self.compute_7x7_resnet

        x = conv1_fn(inputs, training=training, **kwargs)

        endpoints += [x]  # OS = 2

        if not self.replace_7x7_conv:
            x = self.poo1_pad(x)

        x = self.poo1_pool(x)

        for stack in self.stacks:

            x = stack(x, training=training)

            if stack.output_endpoint:
                x, value_before_stride = x
                endpoints += [value_before_stride]

        endpoints += [x]

        if self.return_endpoints:
            return endpoints
        else:
            return x


def resnet9(
    use_bias=True,
    norm_method=None,
    replace_7x7_conv=False,
    slim_behaviour=False,
    custom_block=None,
    return_endpoints=False,
    conv_func=DEFAULT_CONV_FUNC,
):

    return get_resnet(
        resnet_name=ss.RESNET9,
        num_of_blocks=[1, 1, 1, 1],
        use_bias=use_bias,
        norm_method=norm_method,
        replace_7x7_conv=replace_7x7_conv,
        conv1_depth_multiplier=0.5,
        slim_behaviour=slim_behaviour,
        custom_block=BlockType2Small if custom_block is None else custom_block,
        return_endpoints=return_endpoints,
        conv_func=conv_func,
    )


def resnet10(
    use_bias=True,
    norm_method=None,
    replace_7x7_conv=False,
    slim_behaviour=False,
    custom_block=None,
    return_endpoints=False,
    conv_func=DEFAULT_CONV_FUNC,
):

    return get_resnet(
        resnet_name=ss.RESNET9,
        num_of_blocks=[1, 1, 1, 1],
        use_bias=use_bias,
        norm_method=norm_method,
        replace_7x7_conv=replace_7x7_conv,
        conv1_depth_multiplier=(0.375, 0.5, 0.5) if replace_7x7_conv else 0.5,
        slim_behaviour=slim_behaviour,
        custom_block=BlockType2Small if custom_block is None else custom_block,
        return_endpoints=return_endpoints,
        conv_func=conv_func,
    )



def resnet18(
    use_bias=True,
    norm_method=None,
    replace_7x7_conv=False,
    slim_behaviour=False,
    custom_block=None,
    return_endpoints=False,
    conv_func=DEFAULT_CONV_FUNC,
):

    return get_resnet(
        resnet_name=ss.RESNET18,
        num_of_blocks=[2, 2, 2, 2],
        use_bias=use_bias,
        norm_method=norm_method,
        replace_7x7_conv=replace_7x7_conv,
        conv1_depth_multiplier=0.5,
        slim_behaviour=slim_behaviour,
        custom_block=BlockType2Small if custom_block is None else custom_block,
        return_endpoints=return_endpoints,
        conv_func=conv_func,
    )


def resnet50(
    use_bias=True,
    norm_method=None,
    replace_7x7_conv=False,
    slim_behaviour=False,
    custom_block=None,
    return_endpoints=False,
    conv_func=DEFAULT_CONV_FUNC,
):

    return get_resnet(
        resnet_name=ss.RESNET50,
        num_of_blocks=[3, 4, 6, 3],
        use_bias=use_bias,
        norm_method=norm_method,
        replace_7x7_conv=replace_7x7_conv,
        slim_behaviour=slim_behaviour,
        custom_block=custom_block,
        return_endpoints=return_endpoints,
        conv_func=conv_func,
    )


def resnet101(
    use_bias=True,
    norm_method=None,
    replace_7x7_conv=False,
    slim_behaviour=False,
    custom_block=None,
    return_endpoints=False,
    conv_func=DEFAULT_CONV_FUNC,
):

    return get_resnet(
        resnet_name=ss.RESNET101,
        num_of_blocks=[3, 4, 23, 3],
        use_bias=use_bias,
        norm_method=norm_method,
        replace_7x7_conv=replace_7x7_conv,
        slim_behaviour=slim_behaviour,
        custom_block=custom_block,
        return_endpoints=return_endpoints,
        conv_func=conv_func,
    )


def resnet152(
    use_bias=True,
    norm_method=None,
    replace_7x7_conv=False,
    slim_behaviour=False,
    custom_block=None,
    return_endpoints=False,
    conv_func=DEFAULT_CONV_FUNC,
):

    return get_resnet(
        resnet_name=ss.RESNET152,
        num_of_blocks=[3, 8, 36, 3],
        use_bias=use_bias,
        norm_method=norm_method,
        replace_7x7_conv=replace_7x7_conv,
        slim_behaviour=slim_behaviour,
        custom_block=custom_block,
        return_endpoints=return_endpoints,
        conv_func=conv_func,
    )


def get_resnet(
    resnet_name=ss.RESNET50,
    num_of_blocks=[3, 4, 6, 3],
    use_bias=True,
    norm_method=None,
    replace_7x7_conv=False,
    slim_behaviour=False,
    conv1_depth_multiplier=1,
    custom_block=None,
    conv_func=DEFAULT_CONV_FUNC,
    return_endpoints=False,
):

    stacks = build_stacks(
        num_of_blocks=num_of_blocks,
        use_bias=use_bias,
        norm_method=norm_method,
        slim_behaviour=slim_behaviour,
        custom_block=custom_block,
        conv_func=conv_func,
    )

    return ResNet(
        stacks,
        use_bias=use_bias,
        norm_method=norm_method,
        replace_7x7_conv=replace_7x7_conv,
        conv1_depth_multiplier=conv1_depth_multiplier,
        return_endpoints=return_endpoints,
        conv_func=conv_func,
        name=resnet_name,
    )


def build_stacks(
    num_of_blocks=[3, 4, 23, 3], 
    use_bias=True, 
    norm_method=None, 
    slim_behaviour=False, 
    custom_block=None,
    conv_func=DEFAULT_CONV_FUNC,
):

    if not slim_behaviour:
        strides = [1, 2, 2, 2]
        stacks_func = Stack
    else:
        strides = [2, 2, 2, 1]
        stacks_func = Stack2
        use_bias = False

    filters_list = [64, 128, 256, 512]
    stacks = []

    for i in range(4):
        stack = stacks_func(
            filters=filters_list[i],
            blocks_count=num_of_blocks[i],
            stride1=strides[i],
            use_bias=use_bias,
            norm_method=norm_method,
            custom_block=custom_block,
            conv_func=conv_func,
            name="conv{}".format(i + 2),
        )

        stacks.append(stack)

    return stacks


def build_atrous_resnet(resnet : ResNet, output_stride=32):

    stacks = resnet.stacks

    if len(stacks) != 4:
        return ValueError("Len of stacks must be 4")
    
    current_os = 4
    
    if output_stride == 2:
        resnet.poo1_pool.strides = (1, 1)

    current_atrous_rate = 1

    for stack in stacks:
        for block in stack.blocks:

            if block.strides > 1:
                if current_os >= output_stride:
                    current_atrous_rate *= 2
                    block.strides = 1
                    block.atrous_rates = block.atrous_rates * current_atrous_rate
                else:
                    current_os *= 2
            else:
                block.atrous_rates = block.atrous_rates * current_atrous_rate

    return resnet


def apply_multi_grid(resnet, block_index=3, grids=[1, 2, 4]):

    stack = resnet.stacks[block_index]

    for i in range(len(stack.blocks)):
        stack.blocks[i].atrous_rates = stack.blocks[i].atrous_rates * grids[i]

    return resnet
