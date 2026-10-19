# ===================================================================
# MIT License
# Copyright (c) 2024 edwardyehuang (https://github.com/edwardyehuang)
# ===================================================================

# TPU-optimized DCNv2 (drop-in replacement: same weight names/shapes as
# iseg.layers.dcnv2.DCNv2 / DCNv2_Flash).
#
# Same dense window machinery as dcnv4_tpu.dcnv4_core_tpu (see that module for
# the XLA:TPU motivation and the measured dead ends), adapted to DCNv2
# semantics:
#   - offset/mask from a 3x3 conv (dilation-aware), zero-init
#   - offsets are (y, x) pairs; mask = sigmoid
#   - sampling grid ignores dilation (same quirk as dcn_v2.py)
#   - sampling position is clamped into the padded image [0, Hp-1]^2 (NOT
#     zeroed like DCNv4); corners at the extreme edge (pos exactly Hp-1)
#     contribute 0, matching dcn_v2's corner clipping
#   - patch points ordered height-outer/width-inner, matching the flattened
#     [kh, kw, C, F] kernel layout
#   - the per-point aggregated map [B,H,W,K,C] feeds the final GEMM
#     [K*C, F] (K can NOT collapse into the weights here, unlike DCNv4)
#
# window_radius tiers match DCNv4_TPU ("exact"/int R with soft clamp).

import math

import tensorflow as tf
import keras

from iseg.utils import get_tensor_shape
from iseg.utils.version_utils import is_keras3
from iseg.utils.value_utils import values_to_tuple_2d
from iseg.utils.keras3_utils import Keras3_Layer_Wrapper

from iseg.layers.dcnv4.dcnv4_tpu import _soft_clip


def dcnv2_core_gather(value, offset, kernel_size=(3, 3), dilation=(1, 1)):
    """Reference gather-based DCNv2 aggregation core (mirrors DCNv2_Flash's
    math after the offset conv). Used for the window_radius="exact" tier.

    value:  [B, H, W, C]
    offset: [B, H, W, 3*K] offset conv output
    returns: map [B, H, W, K, C]
    """

    kh, kw = kernel_size
    ph, pw = (kh - 1) // 2, (kw - 1) // 2
    num_points = kh * kw

    bs, ih, iw, ic = get_tensor_shape(value)

    compute_dtype = tf.float64 if offset.dtype == tf.float64 else tf.float32

    patch_y = [p // kw - ph for p in range(num_points)]
    patch_x = [p % kw - pw for p in range(num_points)]

    oyox = tf.cast(offset[..., : 2 * num_points], compute_dtype)
    oyox = tf.reshape(oyox, [bs, ih, iw, num_points, 2])
    mask = tf.nn.sigmoid(tf.cast(offset[..., 2 * num_points :], compute_dtype))

    grid_y = tf.range(ih, dtype=compute_dtype)[:, None] + tf.constant(patch_y, compute_dtype)[None, :] + ph
    grid_x = tf.range(iw, dtype=compute_dtype)[:, None] + tf.constant(patch_x, compute_dtype)[None, :] + pw

    pos_y = grid_y[None, :, None, :] + oyox[..., 0]
    pos_x = grid_x[None, None, :, :] + oyox[..., 1]

    ih_pad = ih + 2 * ph
    iw_pad = iw + 2 * pw

    y_max = tf.cast(ih_pad - 1, compute_dtype)
    x_max = tf.cast(iw_pad - 1, compute_dtype)

    iy0 = tf.floor(pos_y)
    ix0 = tf.floor(pos_x)
    iy1 = tf.clip_by_value(iy0 + 1.0, 0.0, y_max)
    ix1 = tf.clip_by_value(ix0 + 1.0, 0.0, x_max)
    iy0 = tf.clip_by_value(iy0, 0.0, y_max)
    ix0 = tf.clip_by_value(ix0, 0.0, x_max)
    pos_y = tf.clip_by_value(pos_y, 0.0, y_max)
    pos_x = tf.clip_by_value(pos_x, 0.0, x_max)

    dy0 = pos_y - iy0
    dy1 = iy1 - pos_y
    dx0 = pos_x - ix0
    dx1 = ix1 - pos_x

    weights = [
        mask * (dy0 * dx0),
        mask * (dy0 * dx1),
        mask * (dy1 * dx0),
        mask * (dy1 * dx1),
    ]
    corner_iy = [iy1, iy1, iy0, iy0]
    corner_ix = [ix1, ix0, ix1, ix0]

    x = tf.pad(value, [[0, 0], [ph, ph], [pw, pw], [0, 0]])
    x_flat = tf.reshape(x, [bs * ih_pad * iw_pad, ic])

    iw_pad_i32 = tf.cast(iw_pad, tf.int32)
    batch_index = (
        tf.range(bs, dtype=tf.int32)[:, None, None, None]
        * tf.cast(ih_pad * iw_pad, tf.int32)
    )

    map_all = tf.zeros([bs, ih, iw, num_points, ic], dtype=value.dtype)

    for c in range(4):
        iy = tf.cast(corner_iy[c], tf.int32)
        ix = tf.cast(corner_ix[c], tf.int32)
        idx = iy * iw_pad_i32 + ix + batch_index
        gathered_c = tf.gather(x_flat, idx)
        w_c = tf.cast(weights[c], gathered_c.dtype)
        map_all = map_all + gathered_c * w_c[..., None]

    return map_all


def dcnv2_core_tpu(
    value,
    offset,
    kernel_size=(3, 3),
    dilation=(1, 1),
    offset_scale=1.0,
    window_radius=2,
):
    """Dense window-based deformable aggregation matching DCNv2's sampling
    semantics (see dcnv2_flash.py header), up to the window radius.

    value:  [B, H, W, C]
    offset: [B, H, W, 3*K] offset conv output
    returns: map [B, H, W, K, C] (per-point aggregated map, before the final
             GEMM)
    """

    kh, kw = kernel_size
    ph, pw = (kh - 1) // 2, (kw - 1) // 2
    num_points = kh * kw

    bs, ih, iw, ic = get_tensor_shape(value)

    compute_dtype = tf.float64 if offset.dtype == tf.float64 else tf.float32
    weight_dtype = value.dtype if value.dtype == tf.bfloat16 else compute_dtype

    # patch points, height-outer/width-inner (p = py * kw + px); NOTE: the
    # sampling grid ignores dilation (same quirk as dcn_v2.py)
    points = [(py - ph, px - pw) for py in range(kh) for px in range(kw)]

    oyox = tf.cast(offset[..., : 2 * num_points], compute_dtype)
    oyox = tf.reshape(oyox, [bs, ih, iw, num_points, 2])
    mask = tf.nn.sigmoid(tf.cast(offset[..., 2 * num_points :], compute_dtype))  # [B,H,W,K]

    off_y = oyox[..., 0] * offset_scale  # [B,H,W,K]
    off_x = oyox[..., 1] * offset_scale

    window_radius_f = float(window_radius)
    off_y = _soft_clip(off_y, window_radius_f)
    off_x = _soft_clip(off_x, window_radius_f)

    # sampling happens in the padded-image coordinates: pos = h + ph + py + off
    # -> relative displacement from the output position is (py + ph + off);
    # rel range per axis: [p_min - R, p_max + R]
    p_y_list = [float(py_) for (py_, _) in points]
    p_x_list = [float(px_) for (_, px_) in points]

    rel_y = off_y + tf.constant(p_y_list, compute_dtype)[None, None, None, :]  # [B,H,W,K]
    rel_x = off_x + tf.constant(p_x_list, compute_dtype)[None, None, None, :]

    # padded canvas [B, Hp, Wp, C]; DCNv2 clamps pos into [0, Hp-1] (both axes)
    hp = ih + 2 * ph
    wp = iw + 2 * pw

    # taps relative to the output position in VALUE coords:
    # [p_min - R, p_max + R + 1] covers all reachable corners (floor, floor+1)
    # of pos = base + rel (the pad offset cancels out of the hat argument)
    tap_y0 = int(math.floor(min(p_y_list) - window_radius_f))
    tap_y1 = int(math.floor(max(p_y_list) + window_radius_f)) + 1
    tap_x0 = int(math.floor(min(p_x_list) - window_radius_f))
    tap_x1 = int(math.floor(max(p_x_list) + window_radius_f)) + 1

    t_h = tap_y1 - tap_y0 + 1
    t_w = tap_x1 - tap_x0 + 1
    num_taps = t_h * t_w

    # pad value so that output (h, w) reads canvas2[h : h+t_h, w : w+t_w]
    pad_top = -tap_y0
    pad_left = -tap_x0
    pad_bottom = (ih + t_h - 1) - ih - pad_top
    pad_right = (iw + t_w - 1) - iw - pad_left

    # ---- clamped sampling position (dcnv2 semantics) -------------------------
    rows = tf.range(ih, dtype=compute_dtype)  # [H]
    cols = tf.range(iw, dtype=compute_dtype)  # [W]

    # absolute base of output position (h, w) in canvas coords: (h + ph, w + pw)
    base_y = rows + ph  # [H]
    base_x = cols + pw  # [W]

    hp_f = tf.cast(hp - 1, compute_dtype)
    wp_f = tf.cast(wp - 1, compute_dtype)

    pos_y = base_y[None, :, None, None] + rel_y  # [B,H,W,K], canvas coords
    pos_x = base_x[None, None, :, None] + rel_x
    pos_y = tf.clip_by_value(pos_y, 0.0, hp_f)
    pos_x = tf.clip_by_value(pos_x, 0.0, wp_f)

    # dcnv2's corner clipping makes a point contribute 0 when pos lands exactly
    # on the far edge (pos == Hp-1 or Wp-1): replicate with an edge mask
    not_edge = tf.cast((pos_y < hp_f) & (pos_x < wp_f), compute_dtype)  # [B,H,W,K]
    tap_pos_y = base_y[:, None] + tf.constant(
        [tap_y0 + t for t in range(t_h)], compute_dtype
    )[None, :]  # [H, T_h]
    tap_pos_x = base_x[:, None] + tf.constant(
        [tap_x0 + t for t in range(t_w)], compute_dtype
    )[None, :]  # [W, T_w]

    hat_y = pos_y[..., None] - tap_pos_y[None, :, None, None, :]  # [B,H,W,K,T_h]
    hat_y = tf.maximum(1.0 - tf.abs(hat_y), 0.0)
    hat_x = pos_x[..., None] - tap_pos_x[None, None, :, None, :]  # [B,H,W,K,T_w]
    hat_x = tf.maximum(1.0 - tf.abs(hat_x), 0.0)

    hat_y = tf.cast(hat_y, weight_dtype)
    hat_x = tf.cast(hat_x, weight_dtype)

    # ---- weights: [B, H, W, K, T_h, T_w] (K kept; it feeds the final GEMM) ---
    weights = (
        hat_y[..., :, None]
        * hat_x[..., None, :]
        * tf.cast(mask * not_edge, weight_dtype)[..., None, None]
    )

    # ---- window extraction ---------------------------------------------------
    value_padded = tf.pad(
        value, [[0, 0], [pad_top, pad_bottom], [pad_left, pad_right], [0, 0]]
    )

    map_all = _window_aggregate_per_k(
        value_padded, weights, t_h, t_w, ih, iw
    )  # [B, H, W, K, C]

    return map_all


def _window_aggregate_per_k(value_padded, weights, t_h, t_w, hout, wout):
    """out[b,h,w,k,c] = sum_{ty,tx} weights[b,h,w,k,ty,tx] * vp[b,h+ty,w+tx,c]

    value_padded: [B, Hp, Wp, C]
    weights:      [B, Hout, Wout, K, T_h, T_w]

    The backward is handwritten (value adjoint + weights path), so no large
    buffer lives across the fwd/bwd boundary (XLA offloads those to host).
    """

    bs, channels = get_tensor_shape(value_padded)[0], get_tensor_shape(value_padded)[-1]
    num_points = get_tensor_shape(weights)[3]
    num_taps = t_h * t_w

    def build_patches(v):
        slices = []
        for ty in range(t_h):
            for tx in range(t_w):
                slices.append(v[:, ty : ty + hout, tx : tx + wout, :])
        return tf.stack(slices, axis=-1)  # [B, Hout, Wout, C, T2]

    @tf.custom_gradient
    def _agg(vp, w):
        # w: [B,H,W,K,Th,Tw]; vp: [B,Hp,Wp,C]
        w2d = tf.reshape(w, [bs, hout, wout, num_points, num_taps])  # [B,H,W,K,T2]
        patches = build_patches(vp)  # [B,H,W,C,T2]

        # per-point fused mul+reduce (the shape class that vectorizes in the
        # DCNv4 core): map_k = sum_t patches[...,c,t] * w[k,t]
        outs = []
        for k in range(num_points):
            w_k = w2d[:, :, :, k]  # [B,H,W,T2]
            outs.append(tf.reduce_sum(patches * w_k[:, :, :, None, :], axis=-1))
        out = tf.stack(outs, axis=3)  # [B,H,W,K,C]

        def grad(dout):
            # dout: [B,H,W,K,C]
            patches_bwd = build_patches(vp)

            # dW[b,h,w,k,t] = sum_c patches[...,c,t] * dout[...,k,c]
            dW = []
            for k in range(num_points):
                dout_k = dout[:, :, :, k]  # [B,H,W,C]
                dW.append(
                    tf.reduce_sum(patches_bwd * dout_k[..., None], axis=3)
                )
            dW = tf.stack(dW, axis=3)  # [B,H,W,K,T2]
            dW = tf.reshape(dW, [bs, hout, wout, num_points, t_h, t_w])

            # value adjoint: dVpad = sum_{k,t} pad(W[k,t] * dout[k]) — done
            # per (ty,tx) with K folded into the term (one pad per tap)
            terms = []
            for ty in range(t_h):
                for tx in range(t_w):
                    t = ty * t_w + tx
                    # dU_t[...,c] = sum_k w2d[...,k,t] * dout[...,k,c]
                    dU_t = tf.einsum("bhwk,bhwkc->bhwc", w2d[..., t], dout)
                    term = tf.pad(
                        dU_t,
                        [[0, 0], [ty, t_h - 1 - ty], [tx, t_w - 1 - tx], [0, 0]],
                    )
                    terms.append(term)
            d_vp = tf.add_n(terms)

            return d_vp, dW

        return out, grad

    return _agg(value_padded, weights)


class DCNv2_TPU(Keras3_Layer_Wrapper):
    """DCNv2 backed by the dense window-based TPU core.

    Same weights/semantics as iseg.layers.dcnv2.DCNv2 within the window
    radius (see module header). window_radius="exact" falls back to the
    reference gather-based computation via DCNv2_Flash's forward.
    """

    def __init__(
        self,
        filters,
        kernel_size,
        dilation_rate=1,
        use_bias=True,
        kernel_initializer="glorot_uniform",
        bias_initializer="zeros",
        kernel_regularizer=None,
        bias_regularizer=None,
        use_custom_offset=False,
        activation=None,
        window_radius=2,
        **kwargs,
    ):

        super().__init__(**kwargs)

        self.filters = filters
        self.kernel_size = values_to_tuple_2d(kernel_size)
        self.dilation = values_to_tuple_2d(dilation_rate)
        self.use_bias = use_bias
        self.kernel_initializer = kernel_initializer
        self.bias_initializer = bias_initializer
        self.kernel_regularizer = kernel_regularizer
        self.bias_regularizer = bias_regularizer
        self.use_custom_offset = use_custom_offset
        self.activation = keras.activations.get(activation)
        self.window_radius = window_radius

        self.deformable_groups = 1
        self.stride = (1, 1, 1, 1)

    def build(self, input_shape):

        if self.use_custom_offset:
            input_channels = input_shape[0][-1]
        else:
            input_channels = input_shape[-1]

        self.kernel = self.add_weight(
            name="kernel",
            shape=self.kernel_size + (input_channels, self.filters),
            initializer=self.kernel_initializer,
            regularizer=self.kernel_regularizer,
            trainable=True,
            dtype="float32",
        )

        if self.use_bias:
            self.bias = self.add_weight(
                name="bias",
                shape=(self.filters,),
                initializer=self.bias_initializer,
                regularizer=self.bias_regularizer,
                trainable=True,
                dtype="float32",
            )

        self.offset_kernel = self.add_weight(
            name="offset_kernel",
            shape=self.kernel_size
            + (input_channels, 3 * self.deformable_groups * self.kernel_size[0] * self.kernel_size[1]),
            initializer="zeros",
            trainable=True,
            dtype="float32",
        )

        self.offset_bias = self.add_weight(
            name="offset_bias",
            shape=(3 * self.kernel_size[0] * self.kernel_size[1] * self.deformable_groups,),
            initializer="zeros",
            trainable=True,
            dtype="float32",
        )

        super().build(input_shape)

    def _forward(self, x, offset):

        offset_input = x if offset is None else offset
        offset_out = tf.nn.conv2d(
            offset_input,
            tf.cast(self.offset_kernel, offset_input.dtype),
            strides=(1, 1, 1, 1),
            padding="SAME",
            dilations=self.dilation,
        )
        offset_out = tf.add(offset_out, tf.cast(self.offset_bias, offset_input.dtype))

        if self.window_radius == "exact":
            map_all = dcnv2_core_gather(
                x, offset_out, kernel_size=self.kernel_size, dilation=self.dilation
            )
        else:
            map_all = dcnv2_core_tpu(
                x,
                offset_out,
                kernel_size=self.kernel_size,
                dilation=self.dilation,
                window_radius=self.window_radius,
            )  # [B, H, W, K, C]

        bs, ih, iw, ic = get_tensor_shape(x)

        # final projection as a plain 2D GEMM (same as DCNv2_Flash)
        num_points = self.kernel_size[0] * self.kernel_size[1]
        map_all = tf.reshape(map_all, [bs * ih * iw, num_points * ic])
        final_kernel = tf.reshape(tf.cast(self.kernel, map_all.dtype), [num_points * ic, self.filters])
        output = tf.matmul(map_all, final_kernel)
        output = tf.reshape(output, [bs, ih, iw, self.filters])

        if self.use_bias:
            output = tf.add(output, tf.cast(self.bias, output.dtype), name="output.add.bias")

        output = self.activation(output)

        return output

    def call(self, inputs, training=None):

        if self.use_custom_offset:
            x, offset = tuple(inputs)
        else:
            x = offset = inputs

        return self._forward(x, offset)

    def compute_output_shape(self, input_shape):

        if self.use_custom_offset:
            input_shape = input_shape[0]

        return tuple(input_shape[:-1]) + (self.filters,)

    def get_config(self):

        config = {
            "filters": self.filters,
            "kernel_size": self.kernel_size,
            "dilation_rate": self.dilation,
            "use_bias": self.use_bias,
            "kernel_initializer": self.kernel_initializer,
            "bias_initializer": self.bias_initializer,
            "kernel_regularizer": self.kernel_regularizer,
            "bias_regularizer": self.bias_regularizer,
            "use_custom_offset": self.use_custom_offset,
            "activation": keras.activations.serialize(self.activation),
            "window_radius": self.window_radius,
        }

        base_config = super().get_config()

        return dict(list(base_config.items()) + list(config.items()))
