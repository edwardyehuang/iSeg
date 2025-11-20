import tensorflow as tf
import keras

from iseg.layers.se import SqueezeAndExcitationModule
from iseg.layers.dcn_v2 import DCNv2

from iseg.utils.common import resize_image
from iseg.utils.keras3_utils import Keras3_Model_Wrapper


class FeatureSelectionModule(SqueezeAndExcitationModule):

    def __init__(self, filters=128, activation=tf.nn.relu, name=None):
        super().__init__(
            ratio=1, 
            activation=activation, 
            use_bias=False, 
            name=name,
        )

        self.filters = filters


    def build(self, input_shape):

        self.conv = tf.keras.layers.Conv2D(self.filters, (1, 1), use_bias=False, name="conv")
        
        super().build(input_shape)


    def call(self, inputs, training=None):

        x = inputs
        x = super().call(x, training=training)

        x += tf.cast(inputs, x.dtype)

        x = self.conv(x)

        return x
    


class FeatureAlignment (Keras3_Model_Wrapper):

    def __init__(self, filters=128, name=None):
        super().__init__(name=name)

        self.filters = filters

    
    def build(self, input_shape):
        
        self.lateral_conv = FeatureSelectionModule(filters=self.filters, name="lateral_conv")
        self.offset_conv = tf.keras.layers.Conv2D(self.filters, (1, 1),  use_bias=False, name="offset_conv")

        self.depack_l2 = DCNv2(self.filters, (3, 3), use_custom_offset=True, use_jit_compile=True, name="depack_l2")
        
        super().build(input_shape)


    def call (self, inputs, training=None):

        feats_large, feats_small = inputs

        feats_up = resize_image(feats_small, tf.shape(feats_large)[1:3])

        feats_arm = self.lateral_conv(feats_large, training=training)
        feats_up = tf.cast(feats_up, feats_arm.dtype)

        offset = tf.concat([feats_arm, feats_up * 2], axis=-1)
        offset = self.offset_conv(offset)

        feat_align = self.depack_l2([feats_up, offset], training=training)
        feat_align = tf.nn.relu(feat_align)

        return feat_align + feats_arm


class FeatureAlignedPyramidNet (Keras3_Model_Wrapper):
    def __init__(
        self, 
        skip_conv_filters=256,
        trainable=True,
        warp_coarse_feature=False,
        name=None
    ):
        super().__init__(name=name, trainable=trainable)

        self.skip_conv_filters = skip_conv_filters
        self.warp_coarse_feature = warp_coarse_feature

    def build(self, input_shape):

        feature_map_shapes = input_shape

        self.align_modules = []

        for i in range(len(feature_map_shapes) - 1):
            align_module = FeatureAlignment(self.skip_conv_filters, name=f"skip_conv_filters{i}")
            self.align_modules += [align_module]

        if self.warp_coarse_feature:
            self.warp_coarse_feature = keras.layers.Dense(self.skip_conv_filters, name="coarse_warp_conv")

        super().build(input_shape)

    def call(self, inputs, training=None):

        feature_map_list = list(inputs)

        x = feature_map_list[-1]

        if self.warp_coarse_feature:
            x = self.warp_coarse_feature(x)

        result_endpoints = [x]

        for i in range(len(self.align_modules) - 1, -1, -1):

            skip_feature = feature_map_list[i]
            x = self.align_modules[i]([skip_feature, x], training=training)

            result_endpoints += [x]

        result_endpoints.reverse()

        return result_endpoints