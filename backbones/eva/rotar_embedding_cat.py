# ====================================================================
# MIT License
# Copyright (c) 2024 edwardyehuang (https://github.com/edwardyehuang)
# ====================================================================

import tensorflow as tf
import math

from iseg.utils.version_utils import is_keras3
from iseg.utils.keras3_utils import Keras3_Layer_Wrapper

def pixel_freq_bands(
    num_bands,
    max_freq=224,
    linear_bands=True,
    dtype=tf.float32,
):
    if linear_bands:
        bands = tf.linspace(
            tf.constant(1.0, dtype=dtype), 
            max_freq / 2, num_bands
        )
    else:
        bands = tf.linspace(
            tf.constant(0.0, dtype=dtype), 
            tf.math.log(max_freq) / tf.math.log(2) - 1, 
            num_bands
        )
        bands = tf.math.pow(2, bands)

    bands *= tf.constant(math.pi, dtype=dtype)

    return bands # [num_bands]


def freq_bands(
    num_bands,
    temperature=10000.,
    step=2,
    dtype=tf.float32,
):
    rg = tf.range(num_bands, delta=step, dtype=dtype)
    bands = rg / num_bands
    bands = temperature ** bands
    bands = 1. / bands

    return bands # [num_bands]


def build_fourier_pos_embed(
    feat_shape,
    bands=None,
    num_bands=64,
    max_res=224,
    temperature=10000.,
    linear_bands=False,
    include_grid=False,
    in_pixels=True,
    ref_feat_shape=None,
    dtype=tf.float32,
):
    
    if bands is None:
        if in_pixels:
            bands = pixel_freq_bands(
                num_bands=num_bands,
                max_freq=float(max_res),
                linear_bands=linear_bands,
                dtype=dtype,
            )
        else:
            bands = freq_bands(
                num_bands=num_bands,
                temperature=temperature,
                step=1,
                dtype=dtype,
            )

    one = tf.constant(1.0, dtype=dtype)

    if in_pixels:
        t = [tf.linspace(-one, one, num=s) for s in feat_shape] # 
    else:
        t = [tf.cast(tf.range(s), dtype=dtype) for s in feat_shape]

    if ref_feat_shape is not None:
        t = [x / f * r for x, f, r in zip(t, feat_shape, ref_feat_shape)]

    mesh = tf.meshgrid(*t, indexing="ij")  # [H, W],[H, W]
    grid = tf.stack(mesh, axis=-1)  # [H, W, 2]
    grid = tf.expand_dims(grid, axis=-1)  # [H, W, 2, 1] 

    pos = grid * bands # [H, W, 2, num_bands]

    pos_sin = tf.sin(pos)
    pos_cos = tf.cos(pos)

    if include_grid:
        return grid, pos_sin, pos_cos

    return pos_sin, pos_cos


def rot(inputs):

    x = inputs

    neg_x = -x[..., 1::2]
    pos_x = x[..., ::2]

    x = tf.stack([neg_x, pos_x], axis=-1)
    x = tf.reshape(x, tf.shape(inputs))

    return x


def apply_rot_embed_cat (x, emb):
    emb = tf.cast(emb, x.dtype) # [HW, 8 * num_bands]
    sin_emb, cos_emb = tf.split(emb, 2, axis=-1) # [HW, 8 * num_bands] -> [HW, 4 * num_bands], [HW, 4 * num_bands]

    return x * cos_emb + rot(x) * sin_emb


def build_rotary_pos_embed(
    feat_shape,
    bands=None,
    filters=64,
    max_res=224,
    temperature=10000,
    linear_bands=False,
    in_pixels=True,
    ref_feat_shape=None,
    dtype=tf.float32,
):
    
    sin_emb, cos_emb = build_fourier_pos_embed(
        feat_shape=feat_shape,
        bands=bands,
        num_bands=filters // 4,
        max_res=max_res,
        temperature=temperature,
        linear_bands=linear_bands,
        in_pixels=in_pixels,
        ref_feat_shape=ref_feat_shape,
        dtype=dtype,
    ) # [H, W, 2, num_bands], [H, W, 2, num_bands]

    num_spatial_filters = 1

    for x in feat_shape:
        num_spatial_filters *= x

    sin_emb = tf.reshape(sin_emb, [num_spatial_filters, -1]) # [HW, 2 * num_bands]
    sin_emb = tf.repeat(sin_emb, repeats=[2], axis=-1) # [HW, 4 * num_bands]

    cos_emb = tf.reshape(cos_emb, [num_spatial_filters, -1]) # [HW, 2 * num_bands]
    cos_emb = tf.repeat(cos_emb, repeats=[2], axis=-1) # [HW, 4 * num_bands]

    return sin_emb, cos_emb


class RotaryEmbeddingCat (Keras3_Layer_Wrapper):

    def __init__(
        self,
        filters,
        max_res=224,
        temperature=10000.,
        in_pixels=True,
        linear_bands=False,
        feat_shape=None,
        ref_feat_shape=None,
    ):
        
        super().__init__()

        self.filters = filters
        self.max_res = max_res
        self.temperature = temperature
        self.in_pixels = in_pixels
        self.linear_bands = linear_bands
        self.feat_shape = feat_shape
        self.ref_feat_shape = ref_feat_shape

        if is_keras3():
            self._allow_non_tensor_positional_args = True



    def build(self, input_shape):
        
        if self.feat_shape is None:
            if self.in_pixels:
                bands = pixel_freq_bands(
                    num_bands=self.filters // 4,
                    max_freq=float(self.max_res),
                    linear_bands=self.linear_bands,
                )
            else:
                bands = freq_bands(
                    num_bands=self.filters // 4,
                    temperature=self.temperature,
                    step=1,
                )
            
            self.bands = tf.identity(bands, name="bands")
            self.pos_embed = None
        else:
            # cache full sin/cos embeddings if shape provided up front
            embeds = build_rotary_pos_embed(
                feat_shape=self.feat_shape,
                filters=self.filters,
                max_res=self.max_res,
                linear_bands=self.linear_bands,
                in_pixels=self.in_pixels,
                ref_feat_shape=self.ref_feat_shape,
            )

            self.bands = None
            self.pos_embed = tf.concat(embeds, axis=-1, name="pos_embed") # [HW, 8 * num_bands]


        super().build(input_shape)

    
    def get_embed (self, shape=None):
        if self.bands is not None and shape is not None:
            embeds = build_rotary_pos_embed(
                shape,
                self.bands,
                in_pixels=self.in_pixels,
                ref_feat_shape=self.ref_feat_shape,
            )

            return tf.concat(embeds, axis=-1, name="pos_embed") # [HW, 8 * num_bands]
        elif self.pos_embed is not None:
            return self.pos_embed
        else:
            raise ValueError("get_embed() requires pre-computed pos_embed or valid shape w/ pre-computed bands")
        

    def call (self, inputs):

        spatial_size = inputs

        pos_embed = self.get_embed(spatial_size)

        return pos_embed


        
    
        
