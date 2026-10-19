# ===================================================================
# MIT License
# Copyright (c) 2024 edwardyehuang (https://github.com/edwardyehuang)
# ===================================================================

# TPU-optimized DCNv4.
#
# Why: on TPU, XLA lowers the dynamic tf.gather / scatter (backward) of the
# reference dcnv4_core to per-slice loops with catastrophic throughput
# (measured: gather fwd ~9GB/s/chip effective at 256x256, scatter backward
# +50ms at 128x128). TPUs are fast at dense matmul / elementwise / static
# slicing instead.
#
# This implementation replaces the gather-based bilinear sampling with an
# exactly-equivalent-in-window dense formulation:
#
#   sample(p) = sum_j value[j] * max(0, 1 - |p - j|)   (bilinear "hat")
#
# computed over a static local window of T x T taps around each output
# position (im2col-style static slices), followed by one einsum:
#
#   out[b,h,w,g,d] = sum_{k,t} w[b,h,w,g,k,t] * U[b,h,w,t,g,d]
#
# No dynamic gather/scatter anywhere -> forward AND backward are dense.
#
# Numerics vs dcnv4_core:
#   - identical whenever the (scaled) learned offsets satisfy |off| <= R
#     (window_radius, default 2 pixels)
#   - beyond R, offsets are soft-clamped: |off| -> R + tanh(|off| - R), so
#     sampling stays at the window edge and gradients keep flowing
#     (hard clamp would kill the gradient for out-of-window offsets)
#   - boundary handling identical: zero padding == corner zeroing, and the
#     whole point is dropped outside (-1, W) x (-1, H) as in the official op

import math

import tensorflow as tf

from iseg.utils import get_tensor_shape

from iseg.layers.dcnv4.dcnv4 import DCNv4, _output_hw


def _soft_clip(x, radius):
    # identity inside [-radius, radius], tanh-compressed outside;
    # derivative is 1 inside and sech^2 outside (never exactly 0)
    over = tf.abs(x) - radius
    return tf.where(
        over <= 0.0,
        x,
        tf.sign(x) * (radius + tf.tanh(over)),
    )


def dcnv4_core_tpu(
    value,
    offset_mask,
    kernel_size=(3, 3),
    stride=(1, 1),
    pad=(1, 1),
    dilation=(1, 1),
    group=4,
    offset_scale=1.0,
    remove_center=False,
    window_radius=2,
):
    """Dense window-based deformable aggregation, numerically equal to
    dcnv4_core for |offset * offset_scale| <= window_radius.

    value:       [B, H, W, C] with C = group * group_channels
    offset_mask: [B, Hout, Wout, >= group * K * 3], per group layout
                 [x0, y0, ..., xK-1, yK-1, m0, ..., mK-1]
    returns:     [B, Hout, Wout, C]
    """

    kh, kw = kernel_size
    sh, sw = stride
    ph, pw = pad
    dh, dw = dilation

    center_h, center_w = kh // 2, kw // 2

    points = [
        (i, j)
        for i in range(kw)
        for j in range(kh)
        if not (remove_center and i == center_w and j == center_h)
    ]
    num_points = len(points)

    bs, height, width, channels = get_tensor_shape(value)
    hout = get_tensor_shape(offset_mask)[1]
    wout = get_tensor_shape(offset_mask)[2]

    group_channels = channels // group

    compute_dtype = tf.float64 if value.dtype == tf.float64 else tf.float32

    offset_mask = offset_mask[..., : group * num_points * 3]
    offset_mask = tf.cast(offset_mask, compute_dtype)
    offset_mask = tf.reshape(offset_mask, [bs, hout, wout, group, 3 * num_points])

    offsets = offset_mask[..., : 2 * num_points]
    offsets = tf.reshape(offsets, [bs, hout, wout, group, num_points, 2])
    mask = offset_mask[..., 2 * num_points :]  # [B, Hout, Wout, G, K]

    off_x = offsets[..., 0] * offset_scale  # [B, Hout, Wout, G, K]
    off_y = offsets[..., 1] * offset_scale

    # soft-clamp the (scaled) learned offsets into the window
    window_radius_f = float(window_radius)
    off_x = _soft_clip(off_x, window_radius_f)
    off_y = _soft_clip(off_y, window_radius_f)

    # grid offsets relative to the base position (same as reference impl)
    center_off_h = (dh * (kh - 1)) // 2
    center_off_w = (dw * (kw - 1)) // 2

    point_dx = tf.constant(
        [(i * dw - center_off_w) * offset_scale for (i, _) in points], dtype=compute_dtype
    )  # [K]
    point_dy = tf.constant(
        [(j * dh - center_off_h) * offset_scale for (_, j) in points], dtype=compute_dtype
    )  # [K]

    # displacement from base position = grid part + learned part
    rel_x = off_x + point_dx[None, None, None, None, :]  # [B, Hout, Wout, G, K]
    rel_y = off_y + point_dy[None, None, None, None, :]

    # ---- static tap window -------------------------------------------------
    # taps cover every bilinear corner reachable with |off| <= window_radius:
    # rel in [p_min - R, p_max + R] -> corners floor(rel), floor(rel)+1
    p_x_min = float(min((i * dw - center_off_w) * offset_scale for (i, _) in points))
    p_x_max = float(max((i * dw - center_off_w) * offset_scale for (i, _) in points))
    p_y_min = float(min((j * dh - center_off_h) * offset_scale for (_, j) in points))
    p_y_max = float(max((j * dh - center_off_h) * offset_scale for (_, j) in points))

    tap_x0 = int(math.floor(p_x_min - window_radius_f))
    tap_y0 = int(math.floor(p_y_min - window_radius_f))
    tap_x1 = int(math.floor(p_x_max + window_radius_f)) + 1
    tap_y1 = int(math.floor(p_y_max + window_radius_f)) + 1

    t_w = tap_x1 - tap_x0 + 1
    t_h = tap_y1 - tap_y0 + 1
    num_taps = t_h * t_w

    if sh != 1 or sw != 1:
        raise ValueError(
            "dcnv4_core_tpu requires stride=1 (the window extraction assumes "
            f"consecutive output positions), got stride=({sh}, {sw})"
        )

    # anchor the tap window at the base position base(h) = (center_off - p) + h
    shift_h = center_off_h - ph
    shift_w = center_off_w - pw

    pad_top = -shift_h - tap_y0
    pad_left = -shift_w - tap_x0
    pad_bottom = (hout + t_h - 1) - height - pad_top
    pad_right = (wout + t_w - 1) - width - pad_left

    # ---- hat weights over the window ----------------------------------------
    tap_x = tf.range(tap_x0, tap_x1 + 1, dtype=compute_dtype)  # [T_w]
    tap_y = tf.range(tap_y0, tap_y1 + 1, dtype=compute_dtype)  # [T_h]

    hat_x = rel_x[..., None] - tap_x[None, None, None, None, None, :]
    hat_x = tf.maximum(1.0 - tf.abs(hat_x), 0.0)  # [B, Hout, Wout, G, K, T_w]
    hat_y = rel_y[..., None] - tap_y[None, None, None, None, None, :]
    hat_y = tf.maximum(1.0 - tf.abs(hat_y), 0.0)  # [B, Hout, Wout, G, K, T_h]

    # whole point dropped outside (-1, W) x (-1, H), same as official
    qx = tf.range(wout, dtype=compute_dtype)
    qy = tf.range(hout, dtype=compute_dtype)
    base_x = (center_off_w - pw) + qx * sw  # [Wout]
    base_y = (center_off_h - ph) + qy * sh  # [Hout]

    pos_x = base_x[None, None, :, None, None] + rel_x
    pos_y = base_y[None, :, None, None, None] + rel_y

    height_f = tf.cast(height, compute_dtype)
    width_f = tf.cast(width, compute_dtype)

    point_ok = (pos_x > -1.0) & (pos_x < width_f) & (pos_y > -1.0) & (pos_y < height_f)
    point_ok = tf.cast(point_ok, compute_dtype)  # [B, Hout, Wout, G, K]

    # [B, Hout, Wout, G, K, T_h, T_w]
    weights = (
        hat_y[..., :, None]
        * hat_x[..., None, :]
        * (mask * point_ok)[..., None, None]
    )
    weights = tf.reshape(weights, [bs, hout, wout, group, num_points, num_taps])

    # ---- static window extraction (im2col, no dynamic indexing) ------------
    value_padded = tf.pad(
        value, [[0, 0], [pad_top, pad_bottom], [pad_left, pad_right], [0, 0]]
    )

    patches = tf.image.extract_patches(
        value_padded,
        sizes=[1, t_h, t_w, 1],
        strides=[1, 1, 1, 1],
        rates=[1, 1, 1, 1],
        padding="VALID",
    )  # [B, Hout, Wout, T_h * T_w * C], tap-major then channel

    patches = tf.reshape(
        patches, [bs, hout, wout, num_taps, group, group_channels]
    )

    # ---- dense aggregation ---------------------------------------------------
    # out[b,h,w,g,d] = sum_{k,t} weights[b,h,w,g,k,t] * patches[b,h,w,t,g,d]
    output = tf.einsum(
        "bhwgkt,bhwtgd->bhwgd",
        tf.cast(weights, patches.dtype),
        patches,
    )

    output = tf.reshape(output, [bs, hout, wout, channels])

    return output


class DCNv4_TPU(DCNv4):
    """DCNv4 layer backed by the dense window-based TPU core.

    Numerically identical to DCNv4 for scaled offsets within
    [-window_radius, window_radius] (default +/-2 px); larger offsets are
    soft-clamped toward the window edge with a smooth (tanh) transition so
    gradients keep flowing. See dcnv4_core_tpu for details.
    """

    def __init__(self, *args, window_radius=2, **kwargs):
        # the TPU core is always dense/XLA-friendly; the client-side
        # jit_compile flag is irrelevant (and harmful on TPU remote clients)
        kwargs["use_jit_compile"] = False
        super().__init__(*args, **kwargs)
        self.window_radius = window_radius

    def _forward(self, x, offset):

        bs, height, width, channels = get_tensor_shape(x)

        hout = _output_hw(height, self.kernel_size[0], self.stride[0], self.pad[0], self.dilation[0])
        wout = _output_hw(width, self.kernel_size[1], self.stride[1], self.pad[1], self.dilation[1])

        if not self.without_pointwise:
            value = self._linear(x, self.value_proj_kernel, self.value_proj_bias)
        else:
            value = x

        if self.dw_kernel_size is not None:
            offset_mask_input = tf.nn.depthwise_conv2d(
                offset,
                tf.cast(self.offset_mask_dw_kernel, offset.dtype),
                strides=(1, 1, 1, 1),
                padding="SAME",
            )
            offset_mask_input = tf.add(
                offset_mask_input, tf.cast(self.offset_mask_dw_bias, offset.dtype)
            )
        else:
            offset_mask_input = offset

        offset_mask = self._linear(
            offset_mask_input, self.offset_mask_kernel, self.offset_mask_bias
        )

        output = dcnv4_core_tpu(
            value,
            offset_mask,
            kernel_size=self.kernel_size,
            stride=self.stride,
            pad=self.pad,
            dilation=self.dilation,
            group=self.group,
            offset_scale=self.offset_scale,
            remove_center=self.remove_center,
            window_radius=self.window_radius,
        )

        if self.center_feature_scale:
            scale = self._linear(
                output, self.center_feature_scale_kernel, self.center_feature_scale_bias
            )
            scale = tf.nn.sigmoid(scale)
            scale = tf.repeat(scale[..., None], self.group_channels, axis=-1)
            scale = tf.reshape(scale, [bs, hout, wout, channels])
            output = output * (1.0 - scale) + value * scale

        if not self.without_pointwise:
            output = self._linear(
                output,
                self.output_proj_kernel,
                self.output_proj_bias if self.output_bias else None,
            )

        return output

    def get_config(self):

        config = {"window_radius": self.window_radius}
        base_config = super().get_config()
        return dict(list(base_config.items()) + list(config.items()))
