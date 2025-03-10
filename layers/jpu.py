# ================================================================
# MIT License
# Copyright (c) 2022 edwardyehuang (https://github.com/edwardyehuang)
# ================================================================

# Code implemented 
# "FastFCN: Rethinking Dilated Convolution in the Backbone for Semantic Segmentation"
# https://arxiv.org/pdf/1903.11816.pdf

import tensorflow as tf

from iseg.layers.normalizations import normalization

from iseg.layers.model_builder import ConvNormAct
from iseg.utils.common import resize_image
from iseg.utils.keras3_utils import Keras3_Model_Wrapper


class JointPyramidUpsampling(Keras3_Model_Wrapper):
    def __init__(self, width=512, trainable=True, name=None):
        super().__init__(trainable=trainable, name=name)

        self.width = width

    def build(self, input_shape):

        endpoints_shape_list = input_shape

        assert len(endpoints_shape_list) >= 3

        base_filters = self.width

        self.endpoints_convs = [
            ConvNormAct(base_filters, (3, 3), trainable=self.trainable, name=f"endpoint_conv_{i}") 
            for i in range(3)
        ]

        dilation_rates = [1, 2, 4, 8]

        self.end_depthwise_convs = [
            tf.keras.layers.DepthwiseConv2D(
                (3, 3), 
                padding="same", 
                dilation_rate=r, 
                trainable=self.trainable, 
                name=f"end_depthwise_conv_{r}"
                )
            for r in dilation_rates
        ]

        self.end_depthwise_bns = [
            normalization(trainable=self.trainable, name=f"end_depthwise_bn_{r}") 
            for r in dilation_rates
        ]

        self.end_pointwise_convs = [
            ConvNormAct(base_filters, trainable=self.trainable, name=f"end_pointwise_convs_{r}") 
            for r in dilation_rates
        ]

    def call(self, inputs, training=None):

        endpoints = inputs

        num_endpoints = len(self.endpoints_convs)

        assert num_endpoints <= len(endpoints)

        endpoints = endpoints[-num_endpoints:]

        for i in range(num_endpoints):
            endpoint = endpoints[i]
            endpoint = self.endpoints_convs[i](endpoint, training=training)
            endpoint = resize_image(endpoint, size=tf.shape(endpoints[0])[1:3])
            endpoints[i] = endpoint

        merged_features = tf.concat(endpoints, axis=-1)

        results = []

        for i in range(len(self.end_depthwise_convs)):
            x = self.end_depthwise_convs[i](merged_features)
            x = self.end_depthwise_bns[i](x, training=training)
            x = self.end_pointwise_convs[i](x, training=training)

            results += [x]

        y = tf.concat(results, axis=-1)

        return y