# ===================================================================
# MIT License
# Copyright (c) 2024 edwardyehuang (https://github.com/edwardyehuang)
# ===================================================================

import tensorflow as tf

from iseg.backbones.intern_image.intern_image_layer import InternImageLayer
from iseg.backbones.intern_image.dowmsample_layer import DownsampleLayer

from iseg.utils.keras3_utils import Keras3_Model_Wrapper

LAYER_NORM_EPSILON = 1e-6

class InternImageBlock(Keras3_Model_Wrapper):

    def __init__(
        self,
        depth,
        groups,
        use_downsample=True,
        mlp_ratio=4,
        dropout_rate=0.,
        drop_path_rate=0.,
        activation=tf.nn.gelu,
        use_post_norm=False,
        offset_scale=1.0,
        layer_scale=None,
        depthwise_kernel_size=None,
        post_norm_block_ids=None,
        use_res_post_norm=False,
        center_feature_scale=False,
        trainable=True,
        name=None
    ):
        super().__init__(trainable=trainable, name=name)

        self.depth = depth
        self.groups = groups
        self.use_downsample = use_downsample
        self.mlp_ratio = mlp_ratio
        self.dropout_rate = dropout_rate
        self.drop_path_rate = drop_path_rate
        
        self.activation = activation
        self.use_post_norm = use_post_norm

        self.offset_scale = offset_scale
        self.layer_scale = layer_scale

        self.depthwise_kernel_size = depthwise_kernel_size
        self.post_norm_block_ids = post_norm_block_ids
        self.use_res_post_norm = use_res_post_norm
        self.center_feature_scale = center_feature_scale


    def build(self, input_shape):
        
        self.blocks = []

        for i in range(self.depth):
            block = InternImageLayer(
                groups=self.groups,
                mlp_ratio=self.mlp_ratio,
                dropout_rate=self.dropout_rate,
                drop_path_rate=self.drop_path_rate[i] if isinstance(
                    self.drop_path_rate, list) else self.drop_path_rate,
                activation=self.activation,
                use_post_norm=self.use_post_norm,
                layer_scale=self.layer_scale,
                offset_scale=self.offset_scale,
                depthwise_kernel_size=self.depthwise_kernel_size,
                use_res_post_norm=self.use_res_post_norm,
                center_feature_scale=self.center_feature_scale,
                name=f"{self.name}/layer/{i}"
            )

            self.blocks.append(block)


        if not self.use_post_norm or self.center_feature_scale:
            self.norm = tf.keras.layers.LayerNormalization(
                epsilon=LAYER_NORM_EPSILON, name=f"{self.name}/norm"
            )

        if self.post_norm_block_ids is not None:
            
            self.post_norms = []

            for i in range(len(self.post_norm_block_ids)):
                post_norm = tf.keras.layers.LayerNormalization(
                    epsilon=LAYER_NORM_EPSILON, name=f"{self.name}/post_norms/{i}"
                )

                self.post_norms.append(post_norm)

        
        if self.use_downsample:
            self.downsample = DownsampleLayer(name=f"{self.name}/downsample")
        

        super().build(input_shape)


    def call(self, inputs, training=None):

        x = inputs

        for i, block in enumerate(self.blocks):
            x = block(x, training=training)

            if self.post_norm_block_ids is not None and (i in self.post_norm_block_ids):
                x = self.post_norms[self.post_norm_block_ids.index(i)](x)

        if not self.use_post_norm or self.center_feature_scale:
            x = self.norm(x)

        x_before_downsample = tf.identity(x, name="before_downsample")

        if self.use_downsample:
            x = self.downsample(x, training=training)

        return x, x_before_downsample