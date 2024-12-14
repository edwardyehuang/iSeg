# ====================================================================
# MIT License
# Copyright (c) 2024 edwardyehuang (https://github.com/edwardyehuang)
# ====================================================================

import numpy as np
import tensorflow as tf

from iseg.layers.common_layers import PatchEmbed
from iseg.utils.common import resample_absolute_position_embedding, get_tensor_shape
from iseg.utils.sugars import to_2d_tuple
from iseg.backbones.utils.layerwise_decay import decay_layers_lr

from iseg.backbones.eva.rotar_embedding_cat import RotaryEmbeddingCat
from iseg.backbones.eva.block import EvaBlock
from iseg.utils.version_utils import is_keras3
from iseg.utils.keras3_utils import Keras3_Model_Wrapper

class Eva (Keras3_Model_Wrapper):

    def __init__ (
        self,
        pretrain_img_size=224,
        pretrain_patch_size=14,
        patch_size=16,
        embed_filters=768,
        depth=12,
        num_heads=12,
        qkv_bias=True,
        qkv_fused=True,
        mlp_ratio=4.0,
        swiglu_mlp=False,
        scale_mlp=False,
        scale_attention_inner=False,
        droppout_rate=0.0,
        pos_droppout_rate=0.0,
        attention_dropout_rate=0.0,
        projection_dropout_rate=0.0,
        drop_path_rate=0.0,
        init_values=None,
        use_class_token=True,
        use_abs_pos_emb=True,
        use_rot_pos_emb=True,
        use_post_norm=False,
        dynamic_img_size=True,
        ref_feat_shape=None,
        patch_padding="valid",
        return_endpoints=False,
        trainable=True,
        name=None,
    ):
        
        super().__init__(trainable=trainable, name=name)

        self.pretrain_img_size = pretrain_img_size
        self.pretrain_patch_size = pretrain_patch_size

        self.patch_size = patch_size
        self.embed_filters = embed_filters
        self.depth = depth

        self.num_heads = num_heads
        self.qkv_bias = qkv_bias
        self.qkv_fused = qkv_fused

        self.mlp_ratio = mlp_ratio
        self.swiglu_mlp = swiglu_mlp
        self.scale_mlp = scale_mlp

        self.scale_attention_inner = scale_attention_inner

        self.droppout_rate = droppout_rate
        self.pos_droppout_rate = pos_droppout_rate
        self.attention_dropout_rate = attention_dropout_rate
        self.projection_dropout_rate = projection_dropout_rate
        self.drop_path_rate = drop_path_rate

        self.init_values = init_values

        self.use_class_token = use_class_token
        self.use_abs_pos_emb = use_abs_pos_emb
        self.use_rot_pos_emb = use_rot_pos_emb

        self.use_post_norm = use_post_norm

        self.dynamic_img_size = dynamic_img_size
        self.ref_feat_shape = ref_feat_shape

        self.patch_padding = patch_padding

        self.return_endpoints = return_endpoints

    
    def build(self, input_shape):

        input_height, input_width = input_shape[1:3]

        self.patch_embed : PatchEmbed = PatchEmbed(
            patch_size=self.patch_size,
            weights_patch_size=self.pretrain_patch_size,
            embed_filters=self.embed_filters,
            padding=self.patch_padding,
        )

        grid_size_h = input_height // self.pretrain_patch_size
        grid_size_w = input_width // self.pretrain_patch_size
        grid_size = [grid_size_h, grid_size_w]

        self.grid_size = grid_size

        num_patches =  grid_size_h * grid_size_w

        p_grid_size_h = self.pretrain_img_size // self.pretrain_patch_size
        p_grid_size_w = self.pretrain_img_size // self.pretrain_patch_size

        num_prefix_tokens = 1 if self.use_class_token else 0

        if self.use_class_token:
            self.class_token = self.add_weight(
                name="class_token",
                shape=[1, 1, self.embed_filters],
                initializer=tf.keras.initializers.Zeros(),
                trainable=True,
            )

        if self.use_abs_pos_emb:
            self.position_embedding = self.add_weight(
                name="pos_embed",
                shape=[1, num_patches + num_prefix_tokens, self.embed_filters],
                initializer=tf.keras.initializers.Zeros(),
                trainable=True,
            )
            
            assign_op = self.position_embedding.assign

            def assign_op_wrapper_fn(value, use_locking=False, name=None, read_value=True):

                value_shape = get_tensor_shape(value, return_list=True)

                print(f"assign_op_wrapper_fn: value_shape={value_shape}")

                resized_value = resample_absolute_position_embedding(
                    position_embedding=value,
                    target_size=(grid_size_h, grid_size_w),
                    source_size=(p_grid_size_h, p_grid_size_w),
                    num_prefix_tokens=num_prefix_tokens,
                    method="bicubic",
                )

                if not is_keras3():
                    return assign_op(resized_value, use_locking=use_locking, name=name, read_value=read_value)

                return assign_op(resized_value)
            
            self.position_embedding.assign = assign_op_wrapper_fn


        self.pos_droppout = tf.keras.layers.Dropout(
            rate=self.pos_droppout_rate,
            name="pos_droppout",
        )

        if self.use_rot_pos_emb:
            ref_feat_shape = to_2d_tuple(self.ref_feat_shape) if self.ref_feat_shape is not None else None

            self.rope = RotaryEmbeddingCat(
                filters=self.embed_filters // self.num_heads,
                in_pixels=False,
                feat_shape=None if self.dynamic_img_size else list(to_2d_tuple(grid_size)),
                ref_feat_shape=ref_feat_shape,
            )
        else:
            self.rope = None

        dpr = [x for x in np.linspace(0.0, self.drop_path_rate, self.depth)]

        self.blocks = []

        for i in range(self.depth):
            self.blocks += [
                EvaBlock(
                    num_heads=self.num_heads,
                    qkv_bias=self.qkv_bias,
                    qkv_fused=self.qkv_fused,
                    mlp_ratio=self.mlp_ratio,
                    swiglu_mlp=self.swiglu_mlp,
                    scale_mlp=self.scale_mlp,
                    scale_attention_inner=self.scale_attention_inner,
                    attention_dropout_rate=self.attention_dropout_rate,
                    projection_dropout_rate=self.projection_dropout_rate,
                    drop_path_rate=dpr[i],
                    init_values=self.init_values,
                    use_post_norm=self.use_post_norm,
                    name=f"blocks/{i}",
                )
            ]

        super().build(input_shape)


    def _pos_embed(self, x, training=None):

        batch_size, height, width, channels = get_tensor_shape(x)

        x = tf.reshape(x, [batch_size, height * width, channels])

        pos_embed = self.position_embedding

        if self.use_class_token:
            class_token = tf.cast(self.class_token, x.dtype)
            class_token = tf.broadcast_to(
                class_token, 
                [batch_size, 1, channels]
            )
            x = tf.concat([class_token, x], axis=1) # [N, 1 + HW, C]

        if pos_embed is not None:

            pos_embed = tf.cast(pos_embed, x.dtype)

            if self.dynamic_img_size:
                pos_embed = resample_absolute_position_embedding(
                    position_embedding=pos_embed,
                    target_size=(height, width),
                    source_size=self.grid_size,
                    num_prefix_tokens=1 if self.use_class_token else 0,
                    method="bilinear",
                )

            x += pos_embed

        x = self.pos_droppout(x, training=training)

        return x
    

    def decay_lr(
        self, 
        rate=0.99
    ):

        # layers =  [self.patch_embed] + list(self.blocks)
        layers = list(self.blocks)
        # layers.reverse()

        dacay_weights_names = [
        #    self.position_embedding.name,
            self.class_token.name,
        ]

        all_weights = self.trainable_weights

        matched_weights = []

        for weight in all_weights:
            name = weight.name

            for decay_name in dacay_weights_names:
                if decay_name in name:
                    matched_weights.append(weight)
                    print(f"matched weights : {name} vs {decay_name}")
                    break

        decay_layers_lr(layers, weights=matched_weights, rate=rate)


    def call (self, inputs, training=None):

        x = inputs

        x = self.patch_embed(x)

        patch_embedding = tf.identity(x, name="patch_embedding")

        batch_size, height, width, channels = get_tensor_shape(x)

        rope = self.rope([height, width])

        # x = tf.reshape(x, [batch_size, height * width, channels])
        x = self._pos_embed(x, training=training)

        num_blocks = len(self.blocks)

        endpoints = []

        for i in range(num_blocks):
            x = self.blocks[i]([x, rope], training=training)
            _x = x[:, 1:, :]
            _x = tf.reshape(_x, [batch_size, height, width, channels])
            endpoints.append(_x)

        class_token = x[:, :1, :]

        if self.return_endpoints:
            endpoints = [class_token, patch_embedding] + endpoints
            return endpoints

        return x
    


def EVA02_large_patch14_448(return_endpoints=False):

    return Eva(
        pretrain_img_size=448,
        pretrain_patch_size=14,
        patch_size=14,
        embed_filters=1024,
        depth=24,
        num_heads=16,
        qkv_fused=False,
        mlp_ratio=4 * 2 / 3,
        swiglu_mlp=True,
        scale_mlp=True,
        scale_attention_inner=False,
        drop_path_rate=0.3,
        init_values=None,
        use_class_token=True,
        use_abs_pos_emb=True,
        use_rot_pos_emb=True,
        use_post_norm=False,
        return_endpoints=return_endpoints,
        name="eva02_large_patch14_448",
    )


def EVA02_large_patch14_224(return_endpoints=False):

    return Eva(
        pretrain_img_size=224,
        pretrain_patch_size=14,
        patch_size=14,
        embed_filters=1024,
        depth=24,
        num_heads=16,
        qkv_fused=False,
        mlp_ratio=4 * 2 / 3,
        swiglu_mlp=True,
        scale_mlp=True,
        scale_attention_inner=False,
        drop_path_rate=0.3,
        init_values=None,
        use_class_token=True,
        use_abs_pos_emb=True,
        use_rot_pos_emb=True,
        use_post_norm=False,
        return_endpoints=return_endpoints,
        name="eva02_large_patch14_224",
    )


def EVA02_large_patch16_224(return_endpoints=False):

    return Eva(
        pretrain_img_size=224,
        pretrain_patch_size=16,
        patch_size=16,
        embed_filters=1024,
        depth=24,
        num_heads=16,
        qkv_fused=False,
        mlp_ratio=4 * 2 / 3,
        swiglu_mlp=True,
        scale_mlp=True,
        scale_attention_inner=False,
        drop_path_rate=0.3,
        init_values=None,
        use_class_token=True,
        use_abs_pos_emb=True,
        use_rot_pos_emb=True,
        use_post_norm=False,
        return_endpoints=return_endpoints,
        name="eva02_large_patch16_224",
    )


def EVA02_large_patch16_512_COCO(return_endpoints=False):

    return Eva(
        pretrain_img_size=512,
        pretrain_patch_size=16,
        patch_size=16,
        embed_filters=1024,
        depth=24,
        num_heads=16,
        qkv_fused=False,
        mlp_ratio=4 * 2 / 3,
        swiglu_mlp=True,
        scale_mlp=True,
        scale_attention_inner=False,
        drop_path_rate=0.3,
        init_values=None,
        use_class_token=True,
        use_abs_pos_emb=True,
        use_rot_pos_emb=True,
        use_post_norm=False,
        return_endpoints=return_endpoints,
        name="eva02_large_patch16_512_coco",
    )



def EVA02_tiny_patch_14_336(return_endpoints=False):

    return Eva(
        pretrain_img_size=336,
        pretrain_patch_size=14,
        patch_size=14,
        embed_filters=192,
        depth=12,
        num_heads=3,
        qkv_fused=True,
        mlp_ratio=4 * 2 / 3,
        swiglu_mlp=True,
        scale_mlp=False,
        scale_attention_inner=False,
        drop_path_rate=0.0,
        init_values=None,
        use_class_token=True,
        use_abs_pos_emb=True,
        use_rot_pos_emb=True,
        use_post_norm=False,
        return_endpoints=return_endpoints,
        name="eva02_tiny_patch_14_336",
    )