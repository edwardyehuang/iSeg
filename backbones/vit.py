# ================================================================
# MIT License
# Copyright (c) 2021 edwardyehuang (https://github.com/edwardyehuang)
# ================================================================

# WIP DO NOT USE

import tensorflow as tf

from iseg.utils.attention_utils import flatten_hw
from iseg.layers.model_builder import get_tensor_shape


class VisionTransformer(tf.keras.Model):
    def __init__(self, patch_size, num_layer, num_head, filters=768, mlp_filters=4096, dropout_rate=0.1, name=None):

        super().__init__(name=name)

        self.patch_size = patch_size
        self.num_layer = num_layer
        self.num_head = num_head
        self.filters = filters
        self.mlp_filters = mlp_filters
        self.dropout_rate = dropout_rate

    def build(self, input_shape):

        self.patch_encoder = tf.keras.layers.Conv2D(
            filters=self.filters,
            kernel_size=(self.patch_size, self.patch_size),
            strides=(self.patch_size, self.patch_size),
            padding="valid",
            name="embedding",
        )

        self.positional_encoder = PositionalEncoder(name="posembed_input")

        self.blocks = []

        for i in range(self.num_layer):
            self.blocks += [
                TransformerBlock(
                    self.mlp_filters, num_heads=self.num_head, dropout_rate=self.dropout_rate, name=f"encoderblock_{i}"
                )
            ]

    def call(self, inputs, training=None):

        x = inputs

        x = self.patch_encoder(x)
        x = flatten_hw(x)

        with tf.name_scope("Transformer"):
            x = self.positional_encoder(x)

            for i in range(self.num_layer):
                x = self.blocks[i](x, training=training)

            return x


class PositionalEncoder(tf.keras.Model):
    def __init__(self, trainable=True, name=None):
        super().__init__(name=name, trainable=trainable)

    def build(self, input_shape):

        # input_shape = batch_size, num_patches, channels

        self.num_patches = input_shape[1]

        filters = input_shape[-1]

        self.position_embedding = tf.keras.layers.Embedding(
            input_dim=self.num_patches, output_dim=filters, name="pos_embedding"
        )

        self.built = True

    def call(self, inputs):

        positions = tf.range(start=0, limit=self.num_patches, delta=1)
        encoded = inputs + self.position_embedding(positions)

        return encoded


class TransformerBlock(tf.keras.Model):
    def __init__(self, mlp_filters=4096, num_heads=16, dropout_rate=0.1, name=None):
        super().__init__(name=name)

        self.num_head = num_heads
        self.mlp_filters = mlp_filters
        self.dropout_rate = dropout_rate

    def build(self, input_shape):

        channels = input_shape[-1]

        self.attention_norm = tf.keras.layers.LayerNormalization(epsilon=1e-6, name="LayerNorm_0")
        self.attention = tf.keras.layers.MultiHeadAttention(
            num_heads=self.num_head, key_dim=channels, dropout=self.dropout_rate, name="MultiHeadDotProductAttention_1"
        )

        self.mlp_norm = tf.keras.layers.LayerNormalization(epsilon=1e-6, name="LayerNorm_2")
        self.mlp = MLPBlock(self.mlp_filters, self.dropout_rate, name="MlpBlock_3")

    def call(self, inputs, training=None):

        x = self.attention_norm(inputs)
        x = self.attention(x, x, training=training)

        x = idenity = tf.add(x, inputs)

        x = self.mlp_norm(x)
        x = self.mlp(x, training=training)

        x = tf.add(x, idenity)

        return x


class MLPBlock(tf.keras.Model):
    def __init__(self, filters, dropout_rate=0.1, name=None):
        super().__init__(name=name)

        self.filters = filters
        self.dropout_rate = dropout_rate

    def build(self, input_shape):

        self.dense0 = tf.keras.layers.Dense(self.filters, activation=tf.nn.gelu, name="Dense_0")
        self.dense0_dropout = tf.keras.layers.Dropout(self.dropout_rate)

        self.dense1 = tf.keras.layers.Dense(input_shape[-1], name="Dense_1")
        self.dense1_dropout = tf.keras.layers.Dropout(self.dropout_rate)

    def call(self, inputs, training=None):

        x = self.dense0(inputs)
        x = self.dense0_dropout(x, training=training)

        x = self.dense1(x)
        x = self.dense1_dropout(x, training=training)

        return x


def ViT16L():

    return VisionTransformer(patch_size=16, num_layer=24, num_head=16, filters=1024, mlp_filters=4096, name="ViT-L_16")


def ViT16B():

    return VisionTransformer(patch_size=16, num_layer=12, num_head=12, filters=768, mlp_filters=3072, name="ViT-B_16")


def ViT16S():

    return VisionTransformer(patch_size=16, num_layer=12, num_head=6, filters=384, mlp_filters=1536, name="ViT-S_16")
