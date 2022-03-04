# ================================================================
# MIT License
# Copyright (c) 2021 edwardyehuang (https://github.com/edwardyehuang)
# ================================================================

# Size free for inputs to support tasks like Image Segmentation

import numpy as np
import tensorflow as tf

from iseg.utils.drops import drop_path


class Mlp(tf.keras.Model):
    def __init__(self, in_features, hidden_features=None, out_features=None, dropout_rate=0.0, name=None):
        super().__init__(name=name)

        out_features = out_features or in_features
        hidden_features = hidden_features or in_features

        self.fc1 = tf.keras.layers.Dense(hidden_features, name=f"{name}/fc1")
        self.fc2 = tf.keras.layers.Dense(out_features, name=f"{name}/fc2")

        self.dropout = tf.keras.layers.Dropout(dropout_rate, name=f"{name}/dropout")

    def build(self, input_shape):
        super().build(input_shape)

    def call(self, inputs, training=None):

        x = inputs

        x = self.fc1(x)
        x = tf.nn.gelu(x)
        x = self.dropout(x, training=training)

        x = self.fc2(x)
        x = self.dropout(x, training=training)

        return x


def window_partition(x, window_size):

    height = tf.shape(x)[1]
    width = tf.shape(x)[2]
    channels = x.shape[-1]

    x = tf.reshape(x, shape=[-1, height // window_size, window_size, width // window_size, window_size, channels])
    x = tf.transpose(x, [0, 1, 3, 2, 4, 5])

    windows = tf.reshape(x, shape=[-1, window_size, window_size, channels])

    return windows


def window_reverse(windows, window_size, height, width, channels):

    x = tf.reshape(windows, shape=[-1, height // window_size, width // window_size, window_size, window_size, channels])
    x = tf.transpose(x, [0, 1, 3, 2, 4, 5])
    x = tf.reshape(x, shape=[-1, height, width, channels])

    return x


class WindowAttention(tf.keras.Model):
    def __init__(
        self, filters, window_size, num_heads, use_qkv_bias=True, qk_scale=None, attn_drop=0.0, proj_drop=0.0, name=None
    ):

        super().__init__(name=name)

        self.filters = filters
        self.window_size = window_size
        self.num_heads = num_heads
        head_filters = filters // num_heads
        self.scale = qk_scale or head_filters ** -0.5
        self.use_qkv_bias = use_qkv_bias
        self.attn_drop = attn_drop

        self.proj_drop = proj_drop

    def build(self, input_shape):

        self.relative_position_bias_table = self.add_weight(
            name="relative_position_bias_table",
            shape=((2 * self.window_size[0] - 1) * (2 * self.window_size[1] - 1), self.num_heads),
            initializer=tf.initializers.Zeros(),
            trainable=True,
        )

        coords_h = np.arange(self.window_size[0])
        coords_w = np.arange(self.window_size[1])
        coords = np.stack(np.meshgrid(coords_h, coords_w, indexing="ij"))
        coords_flatten = coords.reshape(2, -1)

        relative_coords = coords_flatten[:, :, None] - coords_flatten[:, None, :]
        relative_coords = relative_coords.transpose([1, 2, 0])
        relative_coords[:, :, 0] += self.window_size[0] - 1
        relative_coords[:, :, 1] += self.window_size[1] - 1
        relative_coords[:, :, 0] *= 2 * self.window_size[1] - 1

        relative_position_index = relative_coords.sum(-1).astype(np.int64)

        self.relative_position_index = tf.Variable(
            initial_value=tf.convert_to_tensor(relative_position_index), trainable=False, name="relative_position_index"
        )

        self.qkv = tf.keras.layers.Dense(self.filters * 3, use_bias=self.use_qkv_bias, name=f"{self.name}/qkv")
        self.attention_dropout = tf.keras.layers.Dropout(self.attn_drop, name=f"{self.name}/attention_dropout")
        self.project = tf.keras.layers.Dense(self.filters, name=f"{self.name}/proj")
        self.project_dropout = tf.keras.layers.Dropout(self.proj_drop, name=f"{self.name}/project_dropout")

        super().build(input_shape)

    def call(self, x, attention_mask=None, training=None):

        _, N, C = x.get_shape().as_list()

        qvk = self.qkv(x)
        qvk = tf.reshape(qvk, [-1, N, 3, self.num_heads, C // self.num_heads])
        qkv = tf.transpose(qvk, [2, 0, 3, 1, 4])

        q, k, v = qkv[0], qkv[1], qkv[2]

        q *= self.scale

        attn = q @ tf.transpose(k, [0, 1, 3, 2])

        relative_position_bias = tf.gather(
            self.relative_position_bias_table, tf.reshape(self.relative_position_index, shape=[-1])
        )
        relative_position_bias = tf.reshape(
            relative_position_bias,
            [self.window_size[0] * self.window_size[1], self.window_size[0] * self.window_size[1], -1],
        )
        relative_position_bias = tf.transpose(relative_position_bias, [2, 0, 1])

        relative_position_bias = tf.cast(relative_position_bias, dtype=attn.dtype)

        attn += tf.expand_dims(relative_position_bias, axis=0)

        # attn_dtype = attn.dtype

        # attn = tf.cast(attn, dtype = tf.float32)

        if attention_mask is not None:
            nW = tf.shape(attention_mask)[0]
            attn = tf.reshape(attn, shape=[-1, nW, self.num_heads, N, N]) + tf.cast(
                tf.expand_dims(tf.expand_dims(attention_mask, axis=1), axis=0), attn.dtype
            )
            attn = tf.reshape(attn, shape=[-1, self.num_heads, N, N])
            attn = tf.nn.softmax(attn, axis=-1)
        else:
            attn = tf.nn.softmax(attn, axis=-1)

        # attn = tf.cast(attn, attn_dtype)

        attn = self.attention_dropout(attn, training=training)

        x = tf.transpose((attn @ v), [0, 2, 1, 3])
        x = tf.reshape(x, shape=[-1, N, C])
        x = self.project(x)
        x = self.project_dropout(x, training=training)

        return x


class DropPath(tf.keras.layers.Layer):
    def __init__(self, drop_prob=None):
        super().__init__()
        self.drop_prob = drop_prob

    def call(self, x, training=None):
        return drop_path(x, self.drop_prob, training=training)


class SwinTransformerBlock(tf.keras.Model):
    def __init__(
        self,
        filters,
        num_heads,
        window_size=7,
        shift_size=0,
        mlp_ratio=4.0,
        use_qkv_bias=True,
        qk_scale=None,
        dropout_rate=0.0,
        attention_dropout_rate=0.0,
        drop_path_prob=0.0,
        norm_layer=tf.keras.layers.LayerNormalization,
        name=None,
    ):

        super().__init__(name=name)

        self.dim = filters
        self.num_heads = num_heads
        self.window_size = window_size
        self.shift_size = shift_size
        self.mlp_ratio = mlp_ratio

        assert 0 <= self.shift_size < self.window_size, "shift_size must in 0-window_size"

        self.norm1 = norm_layer(epsilon=1e-5, name=f"{name}/norm1")

        self.attention = WindowAttention(
            filters,
            window_size=(self.window_size, self.window_size),
            num_heads=num_heads,
            use_qkv_bias=use_qkv_bias,
            qk_scale=qk_scale,
            attn_drop=attention_dropout_rate,
            proj_drop=dropout_rate,
            name=f"{name}/attn",
        )

        self.drop_path = DropPath(drop_path_prob if drop_path_prob > 0.0 else 0.0)

        self.norm2 = norm_layer(epsilon=1e-5, name=f"{name}/norm2")
        mlp_hidden_dim = int(filters * mlp_ratio)
        self.mlp = Mlp(
            in_features=filters, hidden_features=mlp_hidden_dim, dropout_rate=dropout_rate, name=f"{name}/mlp"
        )

    def get_pad_values(self, h, w):
        pad_h = (self.window_size - h % self.window_size) % self.window_size
        pad_w = (self.window_size - w % self.window_size) % self.window_size

        return pad_h, pad_w

    def build(self, input_shape):

        super().build(input_shape)

    def call(self, inputs, attention_mask=None, training=None):

        x = inputs

        input_height = tf.shape(x)[1]
        input_width = tf.shape(x)[2]
        channels = x.shape[-1]

        shortcut = tf.reshape(x, [-1, input_height * input_width, channels])

        x = self.norm1(x)

        pad_h, pad_w = self.get_pad_values(input_height, input_width)

        x = tf.pad(x, [[0, 0], [0, pad_h], [0, pad_w], [0, 0]])

        input_height_padded = tf.shape(x)[1]
        input_width_padded = tf.shape(x)[2]

        # cyclic shift
        if self.shift_size > 0:
            shifted_x = tf.roll(x, [-self.shift_size, -self.shift_size], axis=[1, 2])
            attn_mask = attention_mask
        else:
            shifted_x = x
            attn_mask = None

        # partition windows
        x_windows = window_partition(shifted_x, self.window_size)
        x_windows = tf.reshape(x_windows, shape=[-1, self.window_size * self.window_size, channels])

        # W-MSA/SW-MSA
        attn_windows = self.attention(x_windows, attention_mask=attn_mask, training=training)

        # merge windows
        attn_windows = tf.reshape(attn_windows, [-1, self.window_size, self.window_size, channels])
        shifted_x = window_reverse(attn_windows, self.window_size, input_height_padded, input_width_padded, channels)

        # reverse cyclic shift
        if self.shift_size > 0:
            x = tf.roll(shifted_x, [self.shift_size, self.shift_size], axis=[1, 2])
        else:
            x = shifted_x

        # if pad_h > 0 or pad_w > 0:
        x = x[:, :input_height, :input_width, :]

        x = tf.reshape(x, shape=[-1, input_height * input_width, channels])

        # FFN
        shortcut = tf.cast(shortcut, dtype=x.dtype)

        x = shortcut + self.drop_path(x, training=training)
        x += self.drop_path(self.mlp(self.norm2(x), training=training))

        x = tf.reshape(x, shape=[-1, input_height, input_width, channels])

        return x


class PatchMerging(tf.keras.Model):
    def __init__(self, filters, norm_layer=tf.keras.layers.LayerNormalization, name=None):

        super().__init__(name=name)

        self.filters = filters
        self.reduction = tf.keras.layers.Dense(2 * filters, use_bias=False, name=f"{name}/reduction")
        self.norm = norm_layer(epsilon=1e-5, name=f"{name}/norm")

    def build(self, input_shape):
        super().build(input_shape)

    def call(self, inputs):

        x = inputs

        input_height = tf.shape(x)[1]
        input_width = tf.shape(x)[2]
        channels = x.shape[-1]

        # if input_height % 2 != 0 or input_width % 2 != 0:
        x = tf.pad(x, [[0, 0], [0, input_height % 2], [0, input_width % 2], [0, 0]], name="pad_to_even")
        input_height = tf.shape(x)[1]
        input_width = tf.shape(x)[2]

        x0 = x[:, 0::2, 0::2, :]  # B H/2 W/2 C
        x1 = x[:, 1::2, 0::2, :]  # B H/2 W/2 C
        x2 = x[:, 0::2, 1::2, :]  # B H/2 W/2 C
        x3 = x[:, 1::2, 1::2, :]  # B H/2 W/2 C

        x = tf.concat([x0, x1, x2, x3], axis=-1)
        x = tf.reshape(
            x, shape=[-1, (input_height // 2) * (input_width // 2), 4 * channels], name="reshape_half_size_to_channel"
        )

        x = self.norm(x)
        x = self.reduction(x)

        x = tf.reshape(x, shape=[-1, input_height // 2, input_width // 2, x.shape[-1]], name="end_reshape")

        return x


class BasicLayer(tf.keras.Model):
    def __init__(
        self,
        filters,
        depth,
        num_heads,
        window_size,
        mlp_ratio=4.0,
        qkv_bias=True,
        qk_scale=None,
        drop=0.0,
        attn_drop=0.0,
        drop_path_prob=0.0,
        norm_layer=tf.keras.layers.LayerNormalization,
        downsample=None,
        name=None,
    ):

        super().__init__(name=name)

        self.filters = filters
        self.depth = depth

        self.shift_size = window_size // 2
        self.window_size = window_size

        self.blocks = []

        for i in range(depth):
            block = SwinTransformerBlock(
                filters=filters,
                num_heads=num_heads,
                window_size=window_size,
                shift_size=0 if (i % 2 == 0) else self.shift_size,
                mlp_ratio=mlp_ratio,
                use_qkv_bias=qkv_bias,
                qk_scale=qk_scale,
                dropout_rate=drop,
                attention_dropout_rate=attn_drop,
                drop_path_prob=drop_path_prob[i] if isinstance(drop_path_prob, list) else drop_path_prob,
                norm_layer=norm_layer,
                name=f"{name}/blocks/{i}",
            )

            self.blocks.append(block)

        if downsample is not None:
            self.downsample = downsample(filters=filters, norm_layer=norm_layer, name=f"{name}/downsample")
        else:
            self.downsample = None

    def generate_attention_mask(self, input_height, input_width, window_size, shift_size):

        Hp = input_height / window_size
        Wp = input_width / window_size

        Hp = tf.cast(tf.math.ceil(Hp), tf.int32) * window_size
        Wp = tf.cast(tf.math.ceil(Wp), tf.int32) * window_size

        h_lens = [Hp - window_size, window_size - shift_size, shift_size]
        w_lens = [Wp - window_size, window_size - shift_size, shift_size]

        cnt = 0

        image_mask = []

        for h in h_lens:
            w_masks = []

            for w in w_lens:
                one = tf.ones((), dtype=tf.int32)
                p = tf.ones(
                    shape=tf.stack([one, tf.cast(h, dtype=tf.int32), tf.cast(w, dtype=tf.int32), one], axis=0),
                    dtype=tf.int16,
                )
                p *= cnt
                w_masks.append(p)

                cnt += 1

            w_masks = tf.concat(w_masks, axis=2)
            image_mask.append(w_masks)

        image_mask = tf.concat(image_mask, axis=1, name="image_mask")
        image_mask = tf.cast(image_mask, dtype=tf.float32)

        mask_windows = window_partition(image_mask, window_size)
        mask_windows = tf.reshape(mask_windows, shape=[-1, window_size * window_size])
        attn_mask = tf.expand_dims(mask_windows, axis=1) - tf.expand_dims(mask_windows, axis=2)

        attn_mask = tf.where(attn_mask != 0, -100.0, attn_mask)
        attn_mask = tf.where(attn_mask == 0, 0.0, attn_mask)

        return attn_mask

    def build(self, input_shape):
        super().build(input_shape)

    def call(self, inputs, training=None):

        x = inputs

        input_height = tf.shape(inputs)[1]
        input_width = tf.shape(inputs)[2]

        attn_mask = self.generate_attention_mask(input_height, input_width, self.window_size, self.shift_size)

        for block in self.blocks:
            x = block(x, attention_mask=attn_mask, training=training)

        before_downsample = x

        if self.downsample is not None:
            x = self.downsample(x, training=training)

        return x, before_downsample


class PatchEmbed(tf.keras.Model):
    def __init__(self, patch_size=(4, 4), in_channels=3, embed_filters=96, norm_layer=None, name=None):

        super().__init__(name=name)

        self.patch_size = patch_size

        self.in_chans = in_channels
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


class SwinTransformerModel(tf.keras.Model):
    def __init__(
        self,
        patch_size=(4, 4),
        embed_dim=96,
        depths=[2, 2, 6, 2],
        num_heads=[3, 6, 12, 24],
        window_size=7,
        mlp_ratio=4.0,
        qkv_bias=True,
        qk_scale=None,
        dropout_rate=0.0,
        attention_dropout_rate=0.0,
        drop_path_rate=0.1,
        norm_layer=tf.keras.layers.LayerNormalization,
        use_absolute_pos_embed=False,
        patch_norm=True,
        return_endpoints=False,
        name="swin_tiny_patch4_window7_224",
        **kwargs,
    ):

        super().__init__(name=name)

        self.patch_size = patch_size
        self.depths = depths
        self.num_layers = len(depths)
        self.embed_dim = embed_dim
        self.use_absolute_pos_embed = use_absolute_pos_embed
        self.patch_norm = patch_norm
        self.num_features = int(embed_dim * 2 ** (self.num_layers - 1))
        self.num_heads = num_heads
        self.window_size = window_size
        self.mlp_ratio = mlp_ratio

        self.qkv_bias = qkv_bias
        self.qk_scale = qk_scale

        self.dropout_rate = dropout_rate
        self.attention_dropout_rate = attention_dropout_rate
        self.drop_path_rate = drop_path_rate

        self.norm_layer = norm_layer

        self.return_endpoints = return_endpoints

    def build(self, input_shape):

        channels = input_shape[-1]

        # split image into non-overlapping patches
        self.patch_embed = PatchEmbed(
            patch_size=self.patch_size,
            in_channels=channels,
            embed_filters=self.embed_dim,
            norm_layer=self.norm_layer if self.patch_norm else None,
            name="patch_embed",
        )

        # absolute postion embedding
        if self.use_absolute_pos_embed:
            self.absolute_pos_embed = self.add_weight(
                name="absolute_pos_embed",
                shape=(1, self.patch_embed.num_patches, self.embed_dim),
                initializer=tf.initializers.Zeros(),
            )

        self.pos_drop = tf.keras.layers.Dropout(self.dropout_rate, name="postional_dropout")

        # stochastic depth
        dpr = [x for x in np.linspace(0.0, self.drop_path_rate, sum(self.depths))]

        # build layers

        self.basic_layers = []

        for i_layer in range(self.num_layers):
            basic_layer = BasicLayer(
                filters=int(self.embed_dim * 2 ** i_layer),
                depth=self.depths[i_layer],
                num_heads=self.num_heads[i_layer],
                window_size=self.window_size,
                mlp_ratio=self.mlp_ratio,
                qkv_bias=self.qkv_bias,
                qk_scale=self.qk_scale,
                drop=self.dropout_rate,
                attn_drop=self.attention_dropout_rate,
                drop_path_prob=dpr[sum(self.depths[:i_layer]) : sum(self.depths[: i_layer + 1])],
                norm_layer=self.norm_layer,
                downsample=PatchMerging if (i_layer < self.num_layers - 1) else None,
                name=f"layers/{i_layer}",
            )

            self.basic_layers.append(basic_layer)

        super().build(input_shape)

    def call(self, inputs, training=None):
        x = inputs

        x = self.patch_embed(x)

        if self.use_absolute_pos_embed:
            x = x + tf.reshape(self.absolute_pos_embed, shape=tf.shape(x))

        x = self.pos_drop(x, training=training)

        endpoints = [x]

        for layer in self.basic_layers:
            x, before_downsample = layer(x, training=training)
            endpoints += [before_downsample]

        assert len(endpoints) == len(self.basic_layers) + 1

        if self.return_endpoints:
            return endpoints
        else:
            return x


def swin_tiny_224(return_endpoints=False):

    return SwinTransformerModel(
        name="swin_tiny_224",
        window_size=7,
        embed_dim=96,
        depths=[2, 2, 6, 2],
        num_heads=[3, 6, 12, 24],
        return_endpoints=return_endpoints,
    )


def swin_base_384(return_endpoints=False):

    return SwinTransformerModel(
        name="swin_base_384",
        window_size=12,
        embed_dim=128,
        depths=[2, 2, 18, 2],
        num_heads=[4, 8, 16, 32],
        return_endpoints=return_endpoints,
    )


def swin_large_384(return_endpoints=False):

    return SwinTransformerModel(
        name="swin_large_384",
        window_size=12,
        embed_dim=192,
        depths=[2, 2, 18, 2],
        num_heads=[6, 12, 24, 48],
        return_endpoints=return_endpoints,
    )
