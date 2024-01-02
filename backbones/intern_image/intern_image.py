# ===================================================================
# MIT License
# Copyright (c) 2024 edwardyehuang (https://github.com/edwardyehuang)
# ===================================================================

# This code implments InternImage
# Original pytorch repo: https://github.com/OpenGVLab/InternImage

import tensorflow as tf
import numpy as np

from iseg.backbones.intern_image.intern_image_block import InternImageBlock
from iseg.backbones.intern_image.stem_layer import StemLayer

class InternImage(tf.keras.Model):

    def __init__(
        self,
        stem_filters=64,
        depths=[3, 4, 18, 5],
        groups=[3, 6, 12, 24],
        mlp_ratio=4,
        dropout_rate=0.,
        drop_path_rate=0.2,
        drop_path_type="linear",
        activation=tf.nn.gelu,
        layer_scale=None,
        offset_scale=1.0,
        use_post_norm=False,
        depthwise_kernel_size=None,
        use_level2_post_norm=False,
        level2_post_norm_block_ids=None,
        use_res_post_norm=False,
        use_center_feature_scale=False,
        return_endpoints=False,
        name=None
    ):
        super().__init__(name=name)

        self.stem_filters = stem_filters
        self.depths = depths
        self.groups = groups
        self.mlp_ratio = mlp_ratio

        self.dropout_rate = dropout_rate
        self.drop_path_rate = drop_path_rate
        self.drop_path_type = drop_path_type.lower()

        self.activation = activation
        self.layer_scale = layer_scale
        self.offset_scale = offset_scale
        self.use_post_norm = use_post_norm

        self.depthwise_kernel_size = depthwise_kernel_size

        self.use_level2_post_norm = use_level2_post_norm
        self.level2_post_norm_block_ids = level2_post_norm_block_ids
        self.use_res_post_norm = use_res_post_norm
        self.use_center_feature_scale = use_center_feature_scale

        self.return_endpoints = return_endpoints


    def build(self, input_shape):

        num_blocks = len(self.depths)
        num_layers = sum(self.depths)

        self.patch_embed = StemLayer(
            filters=self.stem_filters,
            activation=self.activation,
            name="patch_embed"
        )

        self.pos_drop = tf.keras.layers.Dropout(
            rate=self.dropout_rate, name="pos_drop"
        )

        self.blocks = []

        if self.drop_path_rate == "uniform":
            dpr = [self.drop_path_rate] * num_layers
        elif self.drop_path_type == "linear":
            dpr = [x for x in np.linspace(0.0, self.drop_path_rate, num_layers)]
        else:
            raise ValueError(f"drop_path_type: {self.drop_path_type} not supported")
        

        for i in range(num_blocks):
            post_norm_block_ids = None

            if self.use_level2_post_norm and i == 2:
                post_norm_block_ids = self.level2_post_norm_block_ids

            block = InternImageBlock(
                depth=self.depths[i],
                groups=self.groups[i],
                use_downsample=(i < num_blocks - 1),
                mlp_ratio=self.mlp_ratio,
                dropout_rate=self.dropout_rate,
                drop_path_rate=dpr[sum(self.depths[:i]):sum(self.depths[:i+1])],
                activation=self.activation,
                use_post_norm=self.use_post_norm,
                layer_scale=self.layer_scale,
                offset_scale=self.offset_scale,
                depthwise_kernel_size=self.depthwise_kernel_size,
                post_norm_block_ids=post_norm_block_ids,
                use_res_post_norm=self.use_res_post_norm,
                center_feature_scale=self.use_center_feature_scale,
                name=f"block/{i}"
            )

            self.blocks.append(block)

        super().build(input_shape)


    def call(self, inputs, training=None):

        x = inputs

        x, before_2nd_stride_x = self.patch_embed(x, training=training)
        x = self.pos_drop(x, training=training)

        endpoints = [before_2nd_stride_x]

        for i in range(len(self.blocks)):
            x, x_before_downsample = self.blocks[i](x, training=training)
            endpoints.append(x_before_downsample)

        if self.return_endpoints:
            return endpoints
        
        return x
    
def intern_image_tiny (return_endpoints=False):

    return InternImage(
        stem_filters=64,
        depths=[4, 4, 18, 4],
        groups=[4, 8, 16, 32],
        mlp_ratio=4.,
        drop_path_rate=0.2,
        layer_scale=1.0,
        offset_scale=1.0,
        use_post_norm=False,
        return_endpoints=return_endpoints,
        name="intern_image_tiny"
    )


def intern_image_small (return_endpoints=False):

    return InternImage(
        stem_filters=80,
        depths=[4, 4, 21, 4],
        groups=[5, 10, 20, 40],
        mlp_ratio=4.,
        drop_path_rate=0.3,
        layer_scale=1.0,
        offset_scale=1.0,
        use_post_norm=True,
        return_endpoints=return_endpoints,
        name="intern_image_small"
    )


def intern_image_huge (return_endpoints=False):

    return InternImage(
        stem_filters=320,
        depths=[6, 6, 32, 6],
        groups=[10, 20, 40, 80],
        mlp_ratio=4.,
        drop_path_rate=0.5,
        layer_scale=None,
        offset_scale=1.0,
        use_post_norm=False,
        depthwise_kernel_size=5,
        use_res_post_norm=True,
        use_level2_post_norm=True,
        level2_post_norm_block_ids=[5, 11, 17, 23, 29],
        use_center_feature_scale=True,
        return_endpoints=return_endpoints,
        name="intern_image_small"
    )

        

        

