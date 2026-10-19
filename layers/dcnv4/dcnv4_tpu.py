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
    # build the big weight tensors in the value dtype (bf16 under mixed
    # precision): halves HBM traffic vs float32. f32/f64 inputs keep full
    # precision (parity path).
    weight_dtype = value.dtype if value.dtype == tf.bfloat16 else compute_dtype

    rel_x_w = tf.cast(rel_x, weight_dtype)
    rel_y_w = tf.cast(rel_y, weight_dtype)

    # Base-hat trick: for integer grid offsets p_k, hat_k[t] = hat_base[t - p_k]
    # exactly, so the K per-point hats are static slices of one small base
    # tensor per axis (K*T entries -> T+spread entries) instead of K full
    # tensors. Off_x/off_y are the same per k; only the tap grid shifts.
    p_x_list = [float((i * dw - center_off_w) * offset_scale) for (i, _) in points]
    p_y_list = [float((j * dh - center_off_h) * offset_scale) for (_, j) in points]
    integer_grid = all(v == int(v) for v in p_x_list + p_y_list)

    if integer_grid:
        p_x_int = [int(v) for v in p_x_list]
        p_y_int = [int(v) for v in p_y_list]

        # base hat must cover (t - p_k) for every tap t and point k:
        # u in [tap0 - p_max, tap1 - p_min]
        ub_x0 = tap_x0 - max(p_x_int)
        ub_x1 = tap_x1 - min(p_x_int)
        ub_y0 = tap_y0 - max(p_y_int)
        ub_y1 = tap_y1 - min(p_y_int)

        tap_ub_x = tf.constant(list(range(ub_x0, ub_x1 + 1)), dtype=weight_dtype)
        tap_ub_y = tf.constant(list(range(ub_y0, ub_y1 + 1)), dtype=weight_dtype)

        # off (learned, clipped) is per k; the tap grid is shared
        hat_base_x = off_x_w = tf.cast(off_x, weight_dtype)
        hat_base_x = hat_base_x[..., None] - tap_ub_x[None, None, None, None, None, :]
        hat_base_x = tf.maximum(1.0 - tf.abs(hat_base_x), 0.0)  # [B,H,W,G,K,Tu_x]
        hat_base_y = off_y_w = tf.cast(off_y, weight_dtype)
        hat_base_y = hat_base_y[..., None] - tap_ub_y[None, None, None, None, None, :]
        hat_base_y = tf.maximum(1.0 - tf.abs(hat_base_y), 0.0)  # [B,H,W,G,K,Tu_y]

        # static slice per k: hat_k[t] = hat_base[t - p_k] with
        # t in [tap0, tap1] -> u = t - p_k in [tap0 - p_k, tap1 - p_k]
        hat_x = tf.stack(
            [
                hat_base_x[..., k, (tap_x0 - p_x_int[k]) - ub_x0 : (tap_x1 - p_x_int[k]) - ub_x0 + 1]
                for k in range(num_points)
            ],
            axis=4,
        )  # [B,H,W,G,K,T_w]
        hat_y = tf.stack(
            [
                hat_base_y[..., k, (tap_y0 - p_y_int[k]) - ub_y0 : (tap_y1 - p_y_int[k]) - ub_y0 + 1]
                for k in range(num_points)
            ],
            axis=4,
        )  # [B,H,W,G,K,T_h]
    else:
        tap_x = tf.constant(list(range(tap_x0, tap_x1 + 1)), dtype=weight_dtype)  # [T_w]
        tap_y = tf.constant(list(range(tap_y0, tap_y1 + 1)), dtype=weight_dtype)  # [T_h]

        hat_x = rel_x_w[..., None] - tap_x[None, None, None, None, None, :]
        hat_x = tf.maximum(1.0 - tf.abs(hat_x), 0.0)  # [B, Hout, Wout, G, K, T_w]
        hat_y = rel_y_w[..., None] - tap_y[None, None, None, None, None, :]
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

    # k collapses into the weights first (exact: k appears only in weights):
    # out = sum_t patches[t] * sum_k weights[k, t].
    s = tf.cast(mask * point_ok, weight_dtype)  # [B, Hout, Wout, G, K]

    # ---- static window extraction + dense aggregation -----------------------
    # out[b,h,w,g,d] = sum_t weights[b,h,w,g,t] * value_padded[b,h+ty,w+tx,g,d]
    #
    # XLA:TPU lowering notes (all measured):
    # - tf.image.extract_patches: catastrophic (+59ms at 128x128x128)
    # - one-hot depthwise conv: catastrophic (75ms fwd at 128x128x128)
    # - loop-accumulating the aggregation over T^2 slices: 40ms fwd+bwd at
    #   256x256x256 (64 mul-adds do not fuse); slices+stack + mul+reduce is
    #   much faster
    # - buffers that live across the fwd/bwd boundary get offloaded to host
    #   memory by XLA (~10ms of S(3) traffic per step at 256x256x256 bf16),
    #   so the whole weights-path backward is handwritten inside the custom
    #   gradient and only small tensors cross the boundary
    value_padded = tf.pad(
        value, [[0, 0], [pad_top, pad_bottom], [pad_left, pad_right], [0, 0]]
    )

    if integer_grid:
        hat_arg_x, hat_arg_y = hat_base_x, hat_base_y
        slice_off_x = [(tap_x0 - p_x_int[k]) - ub_x0 for k in range(num_points)]
        slice_off_y = [(tap_y0 - p_y_int[k]) - ub_y0 for k in range(num_points)]
    else:
        hat_arg_x, hat_arg_y = hat_x, hat_y
        slice_off_x = slice_off_y = None

    output = _window_aggregate(
        value_padded,
        s,
        hat_arg_x,
        hat_arg_y,
        t_h,
        t_w,
        group,
        hout,
        wout,
        group_channels,
        slice_off_x,
        slice_off_y,
    )

    output = tf.reshape(output, [bs, hout, wout, channels])

    return output


def _window_aggregate(
    value_padded,
    s,
    hat_arg_x,
    hat_arg_y,
    t_h,
    t_w,
    group,
    hout,
    wout,
    group_channels,
    slice_off_x,
    slice_off_y,
):
    """out[b,h,w,g,d] = sum_{ty,tx} W[b,h,w,g,ty,tx] * vp[b,h+ty,w+tx,g,d]

    with the k-collapsed weights W[ty,tx] = sum_k s[k] * hat_y[k,ty] * hat_x[k,tx].

    value_padded: [B, Hp, Wp, C] with C = G * D
    s:            [B, Hout, Wout, G, K]  (mask * point_ok)
    hat_arg_x/y:  per-point hats [B, Hout, Wout, G, K, T_w/T_h], or base hats
                  [B, Hout, Wout, G, K, Tu] to be sliced statically with
                  slice_off_x/slice_off_y (integer grid path)
    returns:      [B, Hout, Wout, G, D]
    """

    bs = get_tensor_shape(value_padded)[0]
    channels = group * group_channels
    num_points = get_tensor_shape(s)[4]
    num_taps = t_h * t_w

    def slice_hats(hx, hy):
        if slice_off_x is None:
            return hx, hy
        hat_x = tf.stack(
            [hx[..., k, slice_off_x[k] : slice_off_x[k] + t_w] for k in range(num_points)],
            axis=4,
        )
        hat_y = tf.stack(
            [hy[..., k, slice_off_y[k] : slice_off_y[k] + t_h] for k in range(num_points)],
            axis=4,
        )
        return hat_x, hat_y

    def build_weights(s_, hx_, hy_):
        # W[b,h,w,g,ty,tx] = sum_k s[k] * hat_y[k,ty] * hat_x[k,tx]
        hat_x, hat_y = slice_hats(hx_, hy_)
        tmp = hat_y * s_[..., None]  # [B,H,W,G,K,T_h]
        return tf.reduce_sum(tmp[..., :, None] * hat_x[..., None, :], axis=4)

    @tf.custom_gradient
    def _agg(vp, s_, hx_, hy_):

        def build_patches(v):
            slices = []
            for ty in range(t_h):
                for tx in range(t_w):
                    slices.append(v[:, ty : ty + hout, tx : tx + wout, :])
            p = tf.stack(slices, axis=-1)  # [B, Hout, Wout, C, T2]
            return tf.reshape(p, [bs, hout, wout, group, group_channels, num_taps])

        patches = build_patches(vp)
        w = build_weights(s_, hx_, hy_)  # [B,H,W,G,T_h,T_w]

        out = tf.reduce_sum(
            patches * tf.cast(tf.reshape(w, [bs, hout, wout, group, 1, num_taps]), patches.dtype),
            axis=-1,
        )

        def grad(dout):
            patches_bwd = build_patches(vp)

            # upstream weights gradient: dW[b,h,w,g,ty,tx] = sum_d patches * dout
            dW = tf.reduce_sum(
                patches_bwd * tf.cast(dout[..., None], patches_bwd.dtype), axis=4
            )  # [B,H,W,G,T2]
            dW = tf.reshape(dW, [bs, hout, wout, group, t_h, t_w])

            # ---- value adjoint (recomputed weights; nothing big crosses) ----
            w_bwd = tf.cast(build_weights(s_, hx_, hy_), patches_bwd.dtype)
            dout_e = tf.reshape(
                tf.cast(dout, patches_bwd.dtype), [bs, hout, wout, group, group_channels]
            )

            # At small spatial sizes the 64-term (mul -> pad -> add_n) chain is
            # launch-bound (~192 HLO ops); materializing dp = w * dout in one
            # big broadcast-mul trades a D*T2-sized tensor for ~64 fewer
            # kernel launches. At large sizes that tensor is too big, so keep
            # the per-tap broadcast multiplies there.
            hoist_dp = (hout * wout * channels) <= (1 << 22)

            if hoist_dp:
                dp = w_bwd[..., None, :, :] * tf.reshape(
                    dout_e, [bs, hout, wout, group, group_channels, 1, 1]
                )  # [B,H,W,G,D,Th,Tw]
                dp = tf.reshape(dp, [bs, hout, wout, channels, t_h, t_w])
                terms = []
                for ty in range(t_h):
                    for tx in range(t_w):
                        term = tf.pad(
                            dp[..., ty, tx],
                            [[0, 0], [ty, t_h - 1 - ty], [tx, t_w - 1 - tx], [0, 0]],
                        )
                        terms.append(term)
            else:
                terms = []
                for ty in range(t_h):
                    for tx in range(t_w):
                        term = w_bwd[..., ty, tx][..., None] * dout_e  # [B,H,W,G,D]
                        term = tf.reshape(term, [bs, hout, wout, channels])
                        term = tf.pad(
                            term,
                            [[0, 0], [ty, t_h - 1 - ty], [tx, t_w - 1 - tx], [0, 0]],
                        )
                        terms.append(term)

            d_vp = tf.add_n(terms)  # [B, Hp, Wp, C]

            # ---- weights-path backward (handwritten, exact) -----------------
            # W[ty,tx] = sum_k s[k] * hat_y[k,ty] * hat_x[k,tx]
            hat_x_b, hat_y_b = slice_hats(hx_, hy_)

            # a1[b,h,w,g,ty,k] = sum_tx dW[ty,tx] * hat_x[k,tx]
            a1 = tf.reduce_sum(
                dW[..., :, None, :] * hat_x_b[..., None, :, :], axis=-1
            )  # [B,H,W,G,T_h,K]
            # a2[b,h,w,g,tx,k] = sum_ty dW[ty,tx] * hat_y[k,ty]
            dW_t = tf.transpose(dW, [0, 1, 2, 3, 5, 4])  # [B,H,W,G,T_w,T_h]
            a2 = tf.reduce_sum(
                dW_t[..., :, None, :] * hat_y_b[..., None, :, :], axis=-1
            )  # [B,H,W,G,T_w,K]

            # ds[k] = sum_ty a1[ty,k] * hat_y[k,ty]
            d_s = tf.reduce_sum(a1 * tf.transpose(hat_y_b, [0, 1, 2, 3, 5, 4]), axis=-2)
            d_s = tf.cast(d_s, s_.dtype)  # [B,H,W,G,K]

            # dhat_y[k,ty] = s[k] * a1[ty,k]; dhat_x[k,tx] = s[k] * a2[tx,k]
            dhat_y = a1 * s_[..., None, :]  # [B,H,W,G,T_h,K]
            dhat_x = a2 * s_[..., None, :]  # [B,H,W,G,T_w,K]
            dhat_y = tf.transpose(dhat_y, [0, 1, 2, 3, 5, 4])  # [B,H,W,G,K,T_h]
            dhat_x = tf.transpose(dhat_x, [0, 1, 2, 3, 5, 4])  # [B,H,W,G,K,T_w]

            if slice_off_x is None:
                d_hx, d_hy = dhat_x, dhat_y
            else:
                # scatter the per-point hat grads back into the base hats
                tu_x = get_tensor_shape(hx_)[-1]
                tu_y = get_tensor_shape(hy_)[-1]
                d_hx = tf.stack(
                    [
                        tf.pad(
                            dhat_x[..., k, :],
                            [
                                [0, 0],
                                [0, 0],
                                [0, 0],
                                [0, 0],
                                [slice_off_x[k], tu_x - t_w - slice_off_x[k]],
                            ],
                        )
                        for k in range(num_points)
                    ],
                    axis=4,
                )
                d_hy = tf.stack(
                    [
                        tf.pad(
                            dhat_y[..., k, :],
                            [
                                [0, 0],
                                [0, 0],
                                [0, 0],
                                [0, 0],
                                [slice_off_y[k], tu_y - t_h - slice_off_y[k]],
                            ],
                        )
                        for k in range(num_points)
                    ],
                    axis=4,
                )

            d_hx = tf.cast(d_hx, hx_.dtype)
            d_hy = tf.cast(d_hy, hy_.dtype)

            return d_vp, d_s, d_hx, d_hy

        return out, grad

    return _agg(value_padded, s, hat_arg_x, hat_arg_y)


class DCNv4_TPU(DCNv4):
    """DCNv4 layer backed by the dense window-based TPU core.

    Semantics vs the reference DCNv4, controlled by window_radius:
      - "exact"  : bit-for-bit the reference gather-based core (no clamping,
                   slow on TPU; use for parity checks / fallback)
      - an int R : numerically identical to DCNv4 for scaled offsets within
                   [-R, R] px; larger offsets are soft-clamped toward the
                   window edge with a smooth (tanh) transition so gradients
                   keep flowing. R=2 ("balanced") and R=1 ("ultra") are the
                   recommended speed tiers. See dcnv4_core_tpu for details.
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

        if self.window_radius == "exact":
            from iseg.layers.dcnv4.dcnv4 import dcnv4_core

            output = dcnv4_core(
                value,
                offset_mask,
                kernel_size=self.kernel_size,
                stride=self.stride,
                pad=self.pad,
                dilation=self.dilation,
                group=self.group,
                offset_scale=self.offset_scale,
                remove_center=self.remove_center,
            )
        else:
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
