# ================================================================
# MIT License
# Copyright (c) 2021 edwardyehuang (https://github.com/edwardyehuang)
# ================================================================

# The copyright of FPN belongs to "Feature Pyramid Networks for Object Detection", CVPR 2017

import tensorflow as tf
from iseg.layers.model_builder import ConvBnRelu
from iseg.utils.common import resize_image


class FeaturePyramidNetwork(tf.keras.Model):
    def __init__(self, skip_conv_filters=256, name=None):
        super().__init__(name=name)

        self.skip_conv_filters = skip_conv_filters

    def build(self, input_shape):

        feature_map_shapes = input_shape

        self.skip_convs = []

        for i in range(len(feature_map_shapes) - 1):
            skip_conv = ConvBnRelu(self.skip_conv_filters, name=f"skip_conv_filters{i}")
            self.skip_convs += [skip_conv]

    def call(self, inputs, training=None):

        feature_map_list = list(inputs)

        x = feature_map_list[-1]

        result_endpoints = [x]

        for i in range(len(self.skip_convs) - 1, -1, -1):

            skip_feature = feature_map_list[i]
            skip_feature = self.skip_convs[i](skip_feature, training=training)

            x = resize_image(x, size=tf.shape(skip_feature)[1:3], name=f"upsample_{i}")
            x += skip_feature

            result_endpoints += [x]

        result_endpoints.reverse()

        return result_endpoints


class SemanticPyramidNetworkBlock_V1(tf.keras.Model):
    def __init__(self, filters=128, name=None):
        super().__init__(name=name)

        self.filters = filters

    def build(self, input_shape):

        feature_map_shapes = input_shape

        self.cells = []

        for i in range(len(feature_map_shapes)):
            cell = SemanticPyramidNetworkCell_v1(self.filters, name=f"cell_{i}")
            self.cells.append(cell)

        self.merge_conv = ConvBnRelu(len(feature_map_shapes) * self.filters, (3, 3), name="merge_conv")

    def call(self, inputs, training=None):

        feature_map_list = list(inputs)

        y = [self.cells[i](feature_map_list[i], training=training) for i in range(len(self.cells))]
        y = [y[0]] + [resize_image(feats, size=tf.shape(y[0])[1:3]) for feats in y[1:]]
        y = tf.concat(y, axis=-1)

        y = self.merge_conv(y, training=training)

        return y


class SemanticPyramidNetworkCell_v1(tf.keras.Model):
    def __init__(self, filters=128, name=None):
        super().__init__(name=name)

        self.filters = filters

    def build(self, input_shape):

        self.conv0 = ConvBnRelu(self.filters, (3, 3), name="linear_conv0")
        self.conv1 = ConvBnRelu(self.filters, (3, 3), name="linear_conv1")

    def call(self, inputs, training=None):

        x = inputs
        x = self.conv0(x, training=training)
        x = self.conv1(x, training=training)

        return x


class SemanticPyramidNetworkBlock_V2(tf.keras.Model):
    def __init__(self, filters=128, name=None):
        super().__init__(name=name)

        self.filters = filters

    def build(self, input_shape):

        feature_map_shapes = input_shape

        self.convs = []

        self.num_feature_map = len(feature_map_shapes)

        for i in range(self.num_feature_map):
            num_convs = 1 if i == 0 else i

            for j in range(num_convs):
                conv = ConvBnRelu(self.filters, (3, 3), name=f"s_{i}_conv_{j}")
                self.convs.append(conv)

        self.end_conv = ConvBnRelu(self.num_feature_map * self.filters, name="end_conv")

    def call(self, inputs, training=None):

        feature_map_list = list(inputs)

        assert len(feature_map_list) == self.num_feature_map

        feature_map_sizes = [tf.shape(m)[1:3] for m in feature_map_list]

        results = []

        remained_convs = self.convs

        for i in range(self.num_feature_map):
            num_convs = 1 if i == 0 else i
            convs = remained_convs[:num_convs]
            remained_convs = remained_convs[num_convs:]

            x = feature_map_list[i]

            for j in range(len(convs)):
                x = convs[j](x, training=training)

                if i > 0:
                    target_size = feature_map_sizes[i - j - 1]
                    x = resize_image(x, target_size, name=f"feature_{i}_resize_stage_{j}")

            x = tf.identity(x, name=f"final_feature_{i}")

            results.append(x)

        y = tf.add_n(results)
        y = self.end_conv(y, training=training)

        return y
