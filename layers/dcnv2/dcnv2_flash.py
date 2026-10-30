# ===================================================================
# MIT License
# Copyright (c) 2024 edwardyehuang (https://github.com/edwardyehuang)
# ===================================================================

# High-performance rewrite of iseg.layers.dcnv2.DCNv2_Base (same numerics,
# same weight names/shapes/layouts: drop-in replacement).
#
# XLA optimizations applied (validated in iseg/layers/dcnv4):
#   - explicit per-corner bilinear weights and per-corner gathers, so the
#     backward stays purely elementwise instead of introducing broadcast
#     reduces, and each gather+multiply-add fuses into one kernel
#   - plain 2D GEMM for the final projection
#
# Semantics replicated from the fallback implementation (dcnv2.py):
#   - offset/mask from a conv (3x3 by default) with dilation, zero-init
#   - offsets are (y, x) pairs; sampling positions are clamped into the padded
#     image (NOT zeroed like DCNv4), mask = sigmoid
#   - patch points ordered height-outer/width-inner, matching the flattened
#     [kh, kw, C, F] kernel layout
#   - the sampling grid ignores dilation (same quirk as dcnv2.py)
#   - with use_custom_offset, offset_kernel's input channels follow the
#     channels of x (same quirk as dcnv2.py)

import tensorflow as tf

from iseg.utils import get_tensor_shape
from iseg.layers.dcnv2.dcnv2 import DCNv2_Base


def _corner_aggregate(value_flat, weights, indices):
    """Aggregate the 4 bilinear corners per point with a memory-lean backward.

    value_flat: [N, C] (flattened padded image)
    weights:    [B, 4, H, W, ks] (mask already folded in)
    indices:    [B, 4, H, W, ks] int32 flat row indices
    returns:    [B, H, W, ks, C]

    The backward recomputes the corner gathers instead of saving them, so no
    4 x [B,H,W,ks,C] tensor is kept alive for the backward pass.
    """
    bs, _, ih, iw, ks = get_tensor_shape(indices)
    ic = get_tensor_shape(value_flat)[-1]

    @tf.custom_gradient
    def _agg(value_flat, weights, indices):
        map_all = tf.zeros([bs, ih, iw, ks, ic], dtype=value_flat.dtype)
        for c in range(4):
            g = tf.gather(value_flat, indices[:, c])
            map_all = map_all + g * tf.cast(weights[:, c], g.dtype)[..., None]

        def grad(dy):
            grad_value = tf.zeros_like(value_flat)
            grad_weights = []
            for c in range(4):
                g = tf.gather(value_flat, indices[:, c])
                grad_weight = tf.reduce_sum(tf.cast(dy, g.dtype) * g, axis=-1)
                grad_weights.append(tf.cast(grad_weight, weights.dtype))
                upd = tf.cast(dy, value_flat.dtype) * tf.cast(weights[:, c], value_flat.dtype)[..., None]
                grad_value = tf.tensor_scatter_nd_add(grad_value, indices[:, c][..., None], upd)
            return grad_value, tf.stack(grad_weights, axis=1), None

        return map_all, grad

    return _agg(value_flat, weights, indices)


class DCNv2_Flash(DCNv2_Base):
    """XLA-optimized DCNv2. Weights and config are identical to DCNv2_Base;
    only the forward computation is restructured."""

    def build(self, input_shape):

        super().build(input_shape)

        # patch point (py, px), height-outer/width-inner order
        self._patch_y = [p // self.kernel_size[1] - self.ph for p in range(self.ks)]
        self._patch_x = [p % self.kernel_size[1] - self.pw for p in range(self.ks)]

    def _forward(self, x, offset):

        offset_kernel = tf.cast(self.offset_kernel, offset.dtype)
        offset_bias = tf.cast(self.offset_bias, offset.dtype)
        offset = tf.nn.conv2d(
            offset,
            offset_kernel,
            strides=self.stride,
            padding="SAME",
            dilations=self.dilation,
        )
        offset = tf.add(offset, offset_bias, name="offset.add.bias")

        bs, ih, iw, ic = get_tensor_shape(x)

        compute_dtype = tf.float64 if offset.dtype == tf.float64 else tf.float32

        # [B, H, W, ks] each
        oyox = tf.cast(offset[..., : 2 * self.ks], compute_dtype)
        oyox = tf.reshape(oyox, [bs, ih, iw, self.ks, 2])
        mask = tf.nn.sigmoid(tf.cast(offset[..., 2 * self.ks :], compute_dtype))

        # base grid in padded-image coordinates: (y + ph + py, x + pw + px)
        grid_y = tf.range(ih, dtype=compute_dtype)[:, None] + tf.constant(self._patch_y, compute_dtype)[None, :] + self.ph
        grid_x = tf.range(iw, dtype=compute_dtype)[:, None] + tf.constant(self._patch_x, compute_dtype)[None, :] + self.pw

        pos_y = grid_y[None, :, None, :] + oyox[..., 0]  # [B, H, W, ks]
        pos_x = grid_x[None, None, :, :] + oyox[..., 1]  # [B, H, W, ks]

        # padded image size and valid index range [0, H+1] / [0, W+1]
        ih_pad = ih + 2 * self.ph
        iw_pad = iw + 2 * self.pw

        y_max = tf.cast(ih_pad - 1, compute_dtype)
        x_max = tf.cast(iw_pad - 1, compute_dtype)

        # same clamping order as dcn_v2.py: corners first, then the position
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

        # corner weights, corner order as dcn_v2.py: (y1,x1), (y1,x0), (y0,x1), (y0,x0);
        # mask is folded in here (cheap [B,H,W,ks] ops) instead of multiplying the
        # big [B,H,W,ks,C] aggregated tensor afterwards
        weights = [
            mask * (dy0 * dx0),
            mask * (dy0 * dx1),
            mask * (dy1 * dx0),
            mask * (dy1 * dx1),
        ]
        corner_iy = [iy1, iy1, iy0, iy0]
        corner_ix = [ix1, ix0, ix1, ix0]

        x = tf.pad(x, [[0, 0], [self.ph, self.ph], [self.pw, self.pw], [0, 0]])
        x_flat = tf.reshape(x, [bs * ih_pad * iw_pad, ic])

        iw_pad_i32 = tf.cast(iw_pad, tf.int32)
        batch_index = (
            tf.range(bs, dtype=tf.int32)[:, None, None, None]
            * tf.cast(ih_pad * iw_pad, tf.int32)
        )

        # one gather per corner; accumulate is elementwise (the point reduction
        # happens later in the GEMM, not here). The custom backward recomputes
        # the corner gathers instead of keeping them alive (memory-lean).
        weights_st = tf.stack(weights, axis=1)  # [B, 4, H, W, ks]
        indices_st = tf.stack(
            [
                tf.cast(corner_iy[c], tf.int32) * iw_pad_i32
                + tf.cast(corner_ix[c], tf.int32)
                + batch_index
                for c in range(4)
            ],
            axis=1,
        )  # [B, 4, H, W, ks]
        map_all = _corner_aggregate(x_flat, weights_st, indices_st)

        # final projection as a plain 2D GEMM
        map_all = tf.reshape(map_all, [bs * ih * iw, self.ks * ic])
        final_kernel = tf.reshape(tf.cast(self.kernel, map_all.dtype), [self.ks * ic, self.filters])
        output = tf.matmul(map_all, final_kernel)
        output = tf.reshape(output, [bs, ih, iw, self.filters])

        if self.use_bias:
            output = tf.add(output, tf.cast(self.bias, output.dtype), name="output.add.bias")

        output = self.activation(output)

        return output

