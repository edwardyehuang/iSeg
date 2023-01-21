# ================================================================
# MIT License
# Copyright (c) 2021 edwardyehuang (https://github.com/edwardyehuang)
# ================================================================

import tensorflow as tf


def extract_spatial_patches(x, size=4, use_mean_padding_value=False, padding_direction=0, inverse_slice=False):

    inputs_shape = tf.shape(x)
    batch_size = inputs_shape[0]
    height = inputs_shape[1]
    width = inputs_shape[2]
    channels = x.shape[-1]

    if isinstance(size, tuple):
        size = list(size)

    if not isinstance(size, list):
        size = [size, size]

    patch_size_h = size[0]
    patch_size_w = size[1]

    num_row = height // patch_size_h
    num_col = width // patch_size_w

    r_h = height % patch_size_h
    r_w = width % patch_size_w

    pad_h = tf.where(r_h == 0, 0, patch_size_h - r_h)
    pad_w = tf.where(r_w == 0, 0, patch_size_w - r_w)

    num_row = tf.where(pad_h > 0, num_row + 1, num_row)
    num_col = tf.where(pad_w > 0, num_col + 1, num_col)

    padding_value = 0

    if use_mean_padding_value:
        raise NotImplementedError()

    possible_paddings_arr = [
        [[0, 0], [0, pad_h], [0, pad_w], [0, 0]],
        [[0, 0], [pad_h, 0], [0, pad_w], [0, 0]],
        [[0, 0], [pad_h, 0], [pad_w, 0], [0, 0]],
        [[0, 0], [0, pad_h], [pad_w, 0], [0, 0]],
    ]

    x = tf.pad(x, paddings=possible_paddings_arr[padding_direction], constant_values=padding_value)

    if not inverse_slice:
        x = tf.reshape(x, [batch_size, num_row, patch_size_h, num_col, patch_size_w, channels])
    else:
        x = tf.reshape(x, [batch_size, patch_size_h, num_row, patch_size_w, num_col, channels])

    return x, pad_h, pad_w



class PatchEmbed(tf.keras.Model):
    def __init__(self, patch_size=(4, 4), embed_filters=96, norm_layer=None, name=None):

        super().__init__(name=name)

        self.patch_size = patch_size
        self.embed_dim = embed_filters

        self.proj = tf.keras.layers.Conv2D(
            embed_filters, kernel_size=patch_size, strides=patch_size, name=f"{name}/proj"
        )
        if norm_layer is not None:
            self.norm = norm_layer(epsilon=1e-5, name=f"{name}/norm")
        else:
            self.norm = None

    def build(self, input_shape):
        super().build(input_shape)

    def call(self, x):

        height = tf.shape(x)[1]
        width = tf.shape(x)[2]

        pad_h = tf.where(height % self.patch_size[0] == 0, 0, self.patch_size[0] - height % self.patch_size[0])
        pad_w = tf.where(width % self.patch_size[1] == 0, 0, self.patch_size[1] - width % self.patch_size[1])

        x = tf.pad(x, [[0, 0], [0, pad_h], [0, pad_w], [0, 0]])

        padded_height = tf.shape(x)[1]
        padded_width = tf.shape(x)[2]

        x = self.proj(x)

        x = tf.reshape(
            x, shape=[-1, (padded_height // self.patch_size[0]), (padded_width // self.patch_size[0]), self.embed_dim]
        )

        if self.norm is not None:
            x = self.norm(x)

        return x