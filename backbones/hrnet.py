# ================================================================
# MIT License
# Copyright (c) 2022 edwardyehuang (https://github.com/edwardyehuang)
# ================================================================

# This code is motified from https://github.com/HRNet/HRNet-Semantic-Segmentation/blob/HRNet-OCR/lib/models/hrnet.py
# with slightly improvement on code structure.


import tensorflow as tf

from iseg.layers.normalizations import normalization


class BasicBlock(tf.keras.Model):

    expansion = 1

    def __init__(self, filters, strides=1, downsample=None, name=None):
        super().__init__(name=name)

        self.conv1 = tf.keras.layers.Conv2D(
            filters, (3, 3), strides=strides, padding="same", use_bias=False, name=f"{name}/conv1"
        )
        self.bn1 = normalization(name=f"{name}/bn1")

        self.conv2 = tf.keras.layers.Conv2D(
            filters, (3, 3), strides=strides, padding="same", use_bias=False, name=f"{name}/conv2"
        )
        self.bn2 = normalization(name=f"{name}/bn2")

        self.downsample = downsample
        self.strides = strides

    def build(self, input_shape):

        super().build(input_shape)

    def call(self, inputs, training=None):

        residual = tf.identity(inputs, name="residual")

        if self.downsample is not None:
            residual = self.downsample(residual, training=training)

        x = self.conv1(inputs)
        x = self.bn1(x, training=training)
        x = tf.nn.relu(x)

        x = self.conv2(x)
        x = self.bn2(x, training=training)

        residual = tf.cast(residual, x.dtype)

        return tf.nn.relu(x + residual)


class Bottleneck(tf.keras.Model):

    expansion = 4

    def __init__(self, filters, strides=1, downsample=None, name=None):
        super().__init__(name=name)

        self.conv1 = tf.keras.layers.Conv2D(filters, (1, 1), use_bias=False, name=f"{name}/conv1")
        self.bn1 = normalization(name=f"{name}/bn1")

        self.conv2 = tf.keras.layers.Conv2D(
            filters, (3, 3), strides=strides, padding="same", use_bias=False, name=f"{name}/conv2"
        )
        self.bn2 = normalization(name=f"{name}/bn2")

        self.conv3 = tf.keras.layers.Conv2D(filters * self.expansion, (1, 1), use_bias=False, name=f"{name}/conv3")
        self.bn3 = normalization(name=f"{name}/bn3")

        self.downsample = downsample
        self.strides = strides

    def build(self, input_shape):

        super().build(input_shape)

    def call(self, inputs, training=None):

        residual = tf.identity(inputs, name="residual")

        if self.downsample is not None:
            residual = self.downsample(residual, training=training)

        x = self.conv1(inputs)
        x = self.bn1(x, training=training)
        x = tf.nn.relu(x)

        x = self.conv2(x)
        x = self.bn2(x, training=training)
        x = tf.nn.relu(x)

        x = self.conv3(x)
        x = self.bn3(x, training=training)

        residual = tf.cast(residual, x.dtype)

        return tf.nn.relu(x + residual)


class ConvBlock(tf.keras.Model):
    def __init__(self, filters, kernel_size=(1, 1), strides=1, use_relu=True, name=None) -> None:
        super().__init__(name=name)

        self.use_relu = use_relu

        self.conv = tf.keras.layers.Conv2D(
            filters, kernel_size, strides=strides, padding="same", use_bias=False, name=f"{name}/0"
        )
        self.norm = normalization(name=f"{name}/1")

    def build(self, input_shape):

        super().build(input_shape)

    def call(self, inputs, training=None):

        x = inputs
        x = self.conv(x)
        x = self.norm(x, training=training)

        if self.use_relu:
            x = tf.nn.relu(x)

        return x


class TransitionBlockStack(tf.keras.Model):
    def __init__(self, filters_list, name=None) -> None:
        super().__init__(name=name)

        self.blocks = []

        for i in range(len(filters_list)):
            self.blocks.append(ConvBlock(filters_list[i], (3, 3), strides=2, use_relu=True, name=f"{name}/{i}"))

    def build(self, input_shape):

        super().build(input_shape)

    def call(self, inputs, training=None):

        x = inputs

        for block in self.blocks:
            x = block(x, training=training)

        return x


class DownSampleBlock(tf.keras.Model):
    def __init__(self, filters=None, strides=1, name=None):
        super().__init__(name=name)

        self.conv = tf.keras.layers.Conv2D(filters, (1, 1), strides=strides, use_bias=False, name=f"{name}/0")
        self.norm = normalization(name=f"{name}/1")

    def build(self, input_shape):

        super().build(input_shape)

    def call(self, inputs, training=None):

        x = self.conv(inputs)
        x = self.norm(x, training=training)

        return x


class HighResolutionLayer(tf.keras.Model):
    def __init__(self, block_func, filters, num_blocks, strides=1, name=None):
        super().__init__(name=name)

        self.filters = filters
        self.block_func = block_func
        self.strides = strides
        self.num_blocks = num_blocks

    def build(self, input_shape):

        channels = input_shape[-1]

        downsample = None

        if self.strides != 1 or channels != self.filters * self.block_func.expansion:
            downsample = DownSampleBlock(
                filters=self.filters * self.block_func.expansion, strides=self.strides, name=f"{self.name}/0/downsample"
            )

        self.hr_blocks = [
            self.block_func(filters=self.filters, strides=self.strides, downsample=downsample, name=f"{self.name}/0")
        ]

        for i in range(1, self.num_blocks):
            self.hr_blocks += [self.block_func(filters=self.filters, name=f"{self.name}/{i}")]

        super().build(input_shape)

    def call(self, inputs, training=None):

        x = inputs

        for hr_block in self.hr_blocks:
            x = hr_block(x, training=training)

        return x


class HighResolutionFuseStack(tf.keras.Model):
    def __init__(self, dest_branch_index=0, src_branch_index=0, channnels_list=[], num_branches=1, name=None):
        super().__init__(name=name)

        self.dest_branch_index = dest_branch_index
        self.src_branch_index = src_branch_index
        self.channels_list = channnels_list
        self.num_branches = num_branches

    def build(self, input_shape):

        self.fuse_layers = []

        branch_index_diff = self.dest_branch_index - self.src_branch_index

        for k in range(branch_index_diff):

            if k == branch_index_diff - 1:
                filters = self.channels_list[self.dest_branch_index]
                self.fuse_layers.append(ConvBlock(filters, (3, 3), use_relu=False, strides=2, name=f"{self.name}/{k}"))
            else:
                filters = self.channels_list[self.src_branch_index]
                self.fuse_layers.append(ConvBlock(filters, (3, 3), use_relu=True, strides=2, name=f"{self.name}/{k}"))

        super().build(input_shape)

    def call(self, inputs, training=None):

        x = inputs

        for fuse_layer in self.fuse_layers:
            x = fuse_layer(x, training=training)

        return x


class HighResolutionFuseModule(tf.keras.Model):
    def __init__(self, multi_scale_output=True, name=None) -> None:
        super().__init__(name=name)

        self.multi_scale_output = multi_scale_output

    def build(self, input_shape):

        input_shape_list = input_shape

        self.num_branches = len(input_shape_list)

        self.channels_list = [input_shape_list[i][-1] for i in range(self.num_branches)]

        self.fuse_branches = []

        for i in range(self.num_branches if self.multi_scale_output else 1):

            fuse_layers = []

            for j in range(self.num_branches):
                if j > i:
                    fuse_layers.append(ConvBlock(self.channels_list[i], use_relu=False, name=f"{self.name}/{i}/{j}"))
                elif j == i:
                    fuse_layers.append(None)
                else:
                    fuse_layers.append(
                        HighResolutionFuseStack(
                            i, j, self.channels_list, self.num_branches, name=f"{self.name}/{i}/{j}"
                        )
                    )

            self.fuse_branches.append(fuse_layers)

        super().build(input_shape)

    def call(self, inputs, training=None):

        x_list = inputs

        for i in range(len(self.fuse_branches)):
            y = x_list[0] if i == 0 else self.fuse_branches[i][0](x_list[0], training=training)

            for j in range(1, self.num_branches):

                x = x_list[j]

                if i != j:
                    x = self.fuse_branches[i][j](x, training=training)

                    if j > i:
                        target_size = tf.shape(x_list[i])[1:3]
                        x = tf.compat.v1.image.resize(x, size=target_size, method="bilinear", align_corners=True)

                y += tf.cast(x, y.dtype)

            x_list[i] = tf.nn.relu(y)

        return x_list


class HighResolutionModule(tf.keras.Model):
    def __init__(
        self, block_func, num_block_list, filters_list, multi_scale_output=True, name=None,
    ):

        super().__init__(name=name)

        self.block_func = block_func
        self.num_block_list = num_block_list
        self.filters_list = filters_list

        self.multi_scale_output = multi_scale_output

    def build(self, input_shape):

        input_shape_list = input_shape

        self.num_branhces = len(input_shape_list)

        self.branches = []

        for i in range(self.num_branhces):
            self.branches += [
                HighResolutionLayer(
                    self.block_func, self.filters_list[i], self.num_block_list[i], name=f"{self.name}/branches/{i}"
                )
            ]

        self.fuse_module = HighResolutionFuseModule(self.multi_scale_output, name=f"{self.name}/fuse_layers")

        super().build(input_shape)

    def call(self, inputs, training=None):

        x_list = inputs

        for i in range(self.num_branhces):
            x_list[i] = self.branches[i](x_list[i], training=training)

        if self.num_branhces == 1:
            return x_list

        y_list = self.fuse_module(x_list, training=training)

        return y_list


class HighResolutionTransitionLayer(tf.keras.Model):
    def __init__(self, filters_list=[], name=None) -> None:
        super().__init__(name=name)

        self.filters_list = filters_list

    def build(self, input_shape):

        input_shape_list = input_shape

        num_in_branches = len(input_shape_list)
        num_out_braches = len(self.filters_list)

        channels_list = [input_shape_list[i][-1] for i in range(num_in_branches)]

        self.transition_layers = []

        for i in range(num_out_braches):
            if i < num_in_branches:
                if self.filters_list[i] != channels_list[i]:
                    self.transition_layers += [
                        ConvBlock(
                            filters=self.filters_list[i], kernel_size=(3, 3), use_relu=True, name=f"{self.name}/{i}"
                        )
                    ]
                else:
                    self.transition_layers += [None]
            else:

                sub_filters_list = []

                for j in range(i + 1 - num_in_branches):
                    sub_filters_list += [self.filters_list[i] if j == i - num_in_branches else channels_list[-1]]

                self.transition_layers += [TransitionBlockStack(sub_filters_list, name=f"{self.name}/{i}")]

        super().build(input_shape)

    def call(self, inputs, training=None):

        x_list = inputs

        num_in_branches = len(x_list)

        y_list = []

        for i in range(len(self.transition_layers)):
            if self.transition_layers[i] is not None:
                x = x_list[-1] if num_in_branches <= i else x_list[i]
                y_list += [self.transition_layers[i](x, training=training)]
            else:
                y_list += [x_list[i]]

        return y_list


class HighResolutionStage(tf.keras.Model):
    def __init__(
        self, num_modules, num_block_list, filters_list, block_func=BasicBlock, multi_scale_output=True, name=None
    ):

        super().__init__(name=name)

        filters_list = [filters_list[i] * block_func.expansion for i in range(len(filters_list))]

        self.num_modules = num_modules
        self.block_func = block_func
        self.num_block_list = num_block_list
        self.filters_list = filters_list
        self.multi_scale_output = multi_scale_output
        self.modules = []

        self.transition = HighResolutionTransitionLayer(filters_list, name=f"{self.name}/transition")

    def build(self, input_shape):

        for i in range(self.num_modules):
            # multi_scale_output is only used last module

            reset_multi_scale_output = self.multi_scale_output or i < self.num_modules - 1

            module = HighResolutionModule(
                block_func=self.block_func,
                num_block_list=self.num_block_list,
                filters_list=self.filters_list,
                multi_scale_output=reset_multi_scale_output,
                name=f"{self.name}/{i}",
            )

            self.modules += [module]

        super().build(input_shape)

    def call(self, inputs, training=None):

        x = inputs
        x = self.transition(x, training=training)

        for module in self.modules:
            x = module(x, training=training)

        return x


class HighResolutionNet(tf.keras.Model):
    def __init__(
        self, 
        stage1_filters=64, 
        stage1_block_func=Bottleneck, 
        stage1_num_blokcs=4, 
        return_endpoints=False, 
        name=None):

        super().__init__(name=name)

        self.return_endpoints = return_endpoints

        self.conv1 = tf.keras.layers.Conv2D(64, (3, 3), strides=2, padding="same", use_bias=False, name="conv1")
        self.bn1 = normalization(name="bn1")

        self.conv2 = tf.keras.layers.Conv2D(64, (3, 3), strides=2, padding="same", use_bias=False, name="conv2")
        self.bn2 = normalization(name="bn2")

        # Stage 1

        self.layer1 = HighResolutionLayer(
            block_func=stage1_block_func, filters=stage1_filters, num_blocks=stage1_num_blokcs, name="layer1"
        )

        self.stages = []

    def add_stage(
        self, num_modules=1, filters_list=[48, 96], block_func=BasicBlock, num_blocks_list=[4, 4],
    ):

        stage_index = 2 + len(self.stages)
        self.stages.append(
            HighResolutionStage(
                num_modules=num_modules,
                num_block_list=num_blocks_list,
                filters_list=filters_list,
                block_func=block_func,
                multi_scale_output=True,
                name=f"stage{stage_index}",
            )
        )

    def call(self, inputs, training=None):

        x = inputs

        x = self.conv1(x)
        x = self.bn1(x, training=training)
        x = tf.nn.relu(x)

        x = self.conv2(x)
        x = self.bn2(x, training=training)
        x = tf.nn.relu(x)

        x = self.layer1(x, training=training)

        x_list = [x]

        for stage in self.stages:
            x_list = stage(x_list, training=training)

        high_res_size = tf.shape(x_list[0])[1:3]

        y_list = [x_list[0]]

        for i in range(1, len(x_list)):
            y_list += [tf.compat.v1.image.resize(x_list[i], size=high_res_size, method="bilinear", align_corners=True)]
            y_list[-1] = tf.cast(y_list[-1], y_list[0].dtype)

        y = tf.concat(y_list, axis=-1)

        if self.return_endpoints:
            return x_list + [y]

        return y


def HRNetW48(return_endpoints=False):

    net = HighResolutionNet(64, Bottleneck, 4, return_endpoints=return_endpoints)
    net.add_stage(1, [48, 96], BasicBlock, [4, 4])
    net.add_stage(4, [48, 96, 192], BasicBlock, [4, 4, 4])
    net.add_stage(3, [48, 96, 192, 384], BasicBlock, [4, 4, 4, 4])

    return net


def HRNetW32(return_endpoints=False):

    net = HighResolutionNet(64, Bottleneck, 4, return_endpoints=return_endpoints)
    net.add_stage(1, [32, 64], BasicBlock, [4, 4])
    net.add_stage(4, [32, 64, 128], BasicBlock, [4, 4, 4])
    net.add_stage(3, [32, 64, 128, 256], BasicBlock, [4, 4, 4, 4])

    return net
