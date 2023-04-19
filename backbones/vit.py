# ================================================================
# MIT License
# Copyright (c) 2021 edwardyehuang (https://github.com/edwardyehuang)
# ================================================================


import tensorflow as tf

from iseg.utils.attention_utils import flatten_hw
from iseg.layers.model_builder import get_tensor_shape
from iseg.layers.common_layers import PatchEmbed


def resize_pos_embed(
    pos_embed, # [1, hw, C]
    target_size,
    num_extra_tokens=1
):
    
    assert len(pos_embed.shape) == 3

    extra_tokens = pos_embed[:, :num_extra_tokens]
    pos_embed = pos_embed[:, num_extra_tokens:]

    pos_embed_shape = tf.shape(pos_embed)

    batch_size = pos_embed_shape[0]
    pos_embed_length = pos_embed_shape[1]

    pos_embed_length = tf.cast(pos_embed_length, tf.float32)

    pos_embed_axis_length = tf.cast(
        tf.sqrt(pos_embed_length), dtype=tf.int32
    )

    pos_embed_axis_length = tf.cast(pos_embed_axis_length, tf.int32)

    pos_embed = tf.reshape(pos_embed, [
        batch_size, 
        pos_embed_axis_length,
        pos_embed_axis_length,
        pos_embed.shape[-1]]
    )

    pos_embed = tf.image.resize(
        pos_embed, 
        size=target_size, 
        method=tf.image.ResizeMethod.BICUBIC,
        name="pos_embed_resize"
    )

    pos_embed = tf.reshape(pos_embed, [batch_size, -1, pos_embed.shape[-1]])

    return tf.concat([extra_tokens, pos_embed], axis=1)


class MLPBlock(tf.keras.Model):
    def __init__(
        self, 
        filters, 
        dropout_rate=0.0,
        activation=tf.nn.gelu,
        name=None
    ):
        super().__init__(name=name)

        self.filters = filters
        self.dropout_rate = dropout_rate
        self.activation = activation


    def build(self, input_shape):

        self.dense0 = tf.keras.layers.Dense(
            self.filters, 
            activation=self.activation,
            name=f"{self.name}/dense0"
        )
        
        self.dense0_dropout = tf.keras.layers.Dropout(self.dropout_rate)

        self.dense1 = tf.keras.layers.Dense(
            input_shape[-1], 
            name=f"{self.name}/dense1"
        )
        
        self.dense1_dropout = tf.keras.layers.Dropout(
            self.dropout_rate,
        )

        super().build(input_shape)


    def call(self, inputs, training=None):

        x = self.dense0(inputs)
        x = self.dense0_dropout(x, training=training)

        x = self.dense1(x)
        x = self.dense1_dropout(x, training=training)

        return x
    

class TransformerBlock(tf.keras.Model):
    def __init__(self, mlp_filters=4096, num_heads=16, dropout_rate=0.1, name=None):
        super().__init__(name=name)

        self.num_head = num_heads
        self.mlp_filters = mlp_filters
        self.dropout_rate = dropout_rate

    def build(self, input_shape):

        channels = input_shape[-1]

        self.attention_norm = tf.keras.layers.LayerNormalization(
            epsilon=1e-6, 
            name=f"{self.name}/ln1"
        )

        self.attention = tf.keras.layers.MultiHeadAttention(
            num_heads=self.num_head, 
            key_dim=channels // self.num_head, 
            dropout=self.dropout_rate, 
            name=f"{self.name}/attn"
        )

        self.mlp_norm = tf.keras.layers.LayerNormalization(
            epsilon=1e-6, 
            name=f"{self.name}/ln2"
        )

        self.mlp = MLPBlock(
            self.mlp_filters, 
            self.dropout_rate, 
            name=f"{self.name}/ffn"
        )

        super().build(input_shape)


    def call(self, inputs, training=None):

        x = self.attention_norm(inputs)
        x = self.attention(x, x, training=training)

        x = idenity = tf.add(x, inputs)

        x = self.mlp_norm(x)
        x = self.mlp(x, training=training)

        x = tf.add(x, idenity)

        return x


class VisionTransformer(tf.keras.Model):
    def __init__(
        self, 
        patch_size, 
        num_layer, 
        num_head, 
        filters=768, 
        mlp_filters=4096, 
        dropout_rate=0.1,
        use_class_token=True,
        pretrain_size=224,
        return_endpoints=False,
        name=None
    ):

        super().__init__(name=name)

        self.patch_size = patch_size
        self.num_layer = num_layer
        self.num_head = num_head
        self.filters = filters
        self.mlp_filters = mlp_filters
        self.dropout_rate = dropout_rate

        self.use_class_token = use_class_token

        self.pretrain_size = pretrain_size

        self.return_endpoints = return_endpoints

        

    def build(self, input_shape):

        self.patch_encoder = PatchEmbed(
            patch_size=(self.patch_size, self.patch_size),
            embed_filters=self.filters,
            name=f"{self.name}/patch_embed",
        )

        num_patches_axis = self.pretrain_size // self.patch_size

        self.num_patches = num_patches_axis ** 2
        self.extra_patches = 0

        if self.use_class_token:
            self.class_token = self.add_weight(
                f"{self.name}/class_token",
                shape=[1, 1, self.filters],
                dtype=tf.float32,
                initializer=tf.keras.initializers.Zeros(),
                trainable=True,
            )

            self.extra_patches = 1

        self.position_embedding = self.add_weight(
            f"{self.name}/pos_embed",
            shape=[1, self.num_patches + self.extra_patches, self.filters],
            dtype=tf.float32,
            initializer=tf.keras.initializers.TruncatedNormal(stddev=0.02),
            trainable=True,
        )

        self.blocks = []

        for i in range(self.num_layer):
            self.blocks += [
                TransformerBlock(
                    self.mlp_filters, 
                    num_heads=self.num_head, 
                    dropout_rate=self.dropout_rate, 
                    name=f"{self.name}/layers/{i}"
                )
            ]

        super().build(input_shape)
            

    def call(self, inputs, training=None):

        x = inputs

        x = self.patch_encoder(x)

        batch_size, height, width, channels = get_tensor_shape(x)

        x = flatten_hw(x)

        if self.use_class_token:
            class_token = self.class_token
            class_token = tf.broadcast_to(
                class_token,
                shape=[tf.shape(x)[0], 1, class_token.shape[-1]],
                name="class_token_batch_broadcast"
            )

            x = tf.concat([class_token, x], axis=1)

        position_embedding = resize_pos_embed(
            pos_embed=self.position_embedding,
            target_size=(height, width),
            num_extra_tokens=self.extra_patches,
        )

        x = tf.add(x, position_embedding, name="position_embedding_add")

        for i in range(self.num_layer):
            x = self.blocks[i](x, training=training)

        if self.use_class_token:
            x = x[:, self.extra_patches: ] # remove class token

        x = tf.reshape(x, [batch_size, height, width, channels])

        if self.return_endpoints:
            x = [x]

        return x
            


def ViT16L(return_endpoints=False):

    return VisionTransformer(
        patch_size=16, 
        num_layer=24, 
        num_head=16, 
        filters=1024, 
        mlp_filters=4096, 
        return_endpoints=return_endpoints,
        name="ViT-L_16"
    )


def ViT16B(return_endpoints=False):

    return VisionTransformer(
        patch_size=16, 
        num_layer=12, 
        num_head=12, 
        filters=768, 
        mlp_filters=3072,
        pretrain_size=384,
        return_endpoints=return_endpoints,
        name="ViT-B_16"
    )


def ViT16S(return_endpoints=False):

    return VisionTransformer(
        patch_size=16, 
        num_layer=12, 
        num_head=6, 
        filters=384, 
        mlp_filters=1536,
        pretrain_size=384,
        return_endpoints=return_endpoints, 
        name="ViT-S_16"
    )
