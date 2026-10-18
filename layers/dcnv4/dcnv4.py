# ===================================================================
# MIT License
# Copyright (c) 2024 edwardyehuang (https://github.com/edwardyehuang)
# ===================================================================

# TensorFlow port of DCNv4 (https://github.com/OpenGVLab/DCNv4)
# Pure TF ops (no custom op), XLA JIT compatible.
# Numerics follow the official CUDA kernel (dcnv4_im2col_cuda.cuh):
#   - offsets stored per group as [x0, y0, ..., xK-1, yK-1, m0, ..., mK-1]
#   - sampling position: p0 + (grid + offset) * offset_scale
#   - aggregation mask: raw logits (the official op passes softmax=False)
#   - bilinear sampling, out-of-bound corners contribute 0, sampling points
#     outside (-1, H) x (-1, W) contribute 0
#   - point ordering: width index (i) outer loop, height index (j) inner loop,
#     center point skipped when remove_center=True

import tensorflow as tf

from iseg.utils import get_tensor_shape
from iseg.utils.version_utils import is_keras3
from iseg.utils.value_utils import values_to_tuple_2d
from iseg.utils.keras3_utils import Keras3_Layer_Wrapper


def _pad_2d_from(kernel_size, dilation):
    return (
        (dilation[0] * (kernel_size[0] - 1)) // 2,
        (dilation[1] * (kernel_size[1] - 1)) // 2,
    )


def _output_hw(length, kernel, stride, pad, dilation):
    return (length + 2 * pad - dilation * (kernel - 1) - 1) // stride + 1


def dcnv4_core(
    value,
    offset_mask,
    kernel_size=(3, 3),
    stride=(1, 1),
    pad=(1, 1),
    dilation=(1, 1),
    group=4,
    offset_scale=1.0,
    remove_center=False,
):
    """Core deformable aggregation of DCNv4.

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

    # point order: width index outer, height index inner (same as CUDA kernel)
    points = [
        (i, j)
        for i in range(kw)
        for j in range(kh)
        if not (remove_center and i == center_w and j == center_h)
    ]
    num_points = len(points)  # K per group

    bs, height, width, channels = get_tensor_shape(value)
    hout = get_tensor_shape(offset_mask)[1]
    wout = get_tensor_shape(offset_mask)[2]

    group_channels = channels // group

    # grid/weight math in float32 (like the official CUDA opmath_t),
    # or float64 when running in float64
    compute_dtype = tf.float64 if value.dtype == tf.float64 else tf.float32

    offset_mask = offset_mask[..., : group * num_points * 3]
    offset_mask = tf.cast(offset_mask, compute_dtype)
    offset_mask = tf.reshape(offset_mask, [bs, hout, wout, group, 3 * num_points])

    offsets = offset_mask[..., : 2 * num_points]
    offsets = tf.reshape(offsets, [bs, hout, wout, group, num_points, 2])
    # DCNv4 uses the raw mask logits as aggregation weights (no softmax,
    # softmax=False is passed in the official dcnv4_func.py)
    mask = offset_mask[..., 2 * num_points :]  # [B, Hout, Wout, G, K]

    off_x = offsets[..., 0] * offset_scale  # [B, Hout, Wout, G, K]
    off_y = offsets[..., 1] * offset_scale

    # base grid, constants folded by XLA
    center_off_h = (dh * (kh - 1)) // 2
    center_off_w = (dw * (kw - 1)) // 2

    point_dx = tf.constant(
        [(i * dw - center_off_w) * offset_scale for (i, _) in points], dtype=compute_dtype
    )  # [K]
    point_dy = tf.constant(
        [(j * dh - center_off_h) * offset_scale for (_, j) in points], dtype=compute_dtype
    )  # [K]

    qx = tf.range(wout, dtype=compute_dtype)  # [Wout]
    qy = tf.range(hout, dtype=compute_dtype)  # [Hout]

    base_x = (center_off_w - pw) + qx * sw  # [Wout]
    base_y = (center_off_h - ph) + qy * sh  # [Hout]

    base_x = base_x[:, None] + point_dx[None, :]  # [Wout, K]
    base_y = base_y[:, None] + point_dy[None, :]  # [Hout, K]

    x_pos = base_x[None, None, :, None, :] + off_x  # [B, Hout, Wout, G, K]
    y_pos = base_y[None, :, None, None, :] + off_y  # [B, Hout, Wout, G, K]

    # bilinear corners; per-corner tensors are explicit (no broadcasting) so
    # that the backward pass stays purely elementwise instead of introducing
    # broadcast-reduce ops
    ix0 = tf.floor(x_pos)
    iy0 = tf.floor(y_pos)
    ix1 = ix0 + 1.0
    iy1 = iy0 + 1.0

    lx = x_pos - ix0
    ly = y_pos - iy0
    hx = 1.0 - lx
    hy = 1.0 - ly

    height_f = tf.cast(height, compute_dtype)
    width_f = tf.cast(width, compute_dtype)

    # whole sampling point is dropped outside (-1, H) x (-1, W)
    point_ok = (x_pos > -1.0) & (x_pos < width_f) & (y_pos > -1.0) & (y_pos < height_f)
    point_ok = tf.cast(point_ok, compute_dtype)

    # out-of-bound corners contribute 0
    c1_ok = tf.cast((ix0 >= 0.0) & (iy0 >= 0.0), compute_dtype)
    c2_ok = tf.cast((ix1 < width_f) & (iy0 >= 0.0), compute_dtype)
    c3_ok = tf.cast((ix0 >= 0.0) & (iy1 < height_f), compute_dtype)
    c4_ok = tf.cast((ix1 < width_f) & (iy1 < height_f), compute_dtype)

    base_w = mask * point_ok
    weights = [
        base_w * (hy * hx * c1_ok),
        base_w * (hy * lx * c2_ok),
        base_w * (ly * hx * c3_ok),
        base_w * (ly * lx * c4_ok),
    ]

    # gather row indices into value reshaped as [B * H * W * G, D]
    width_i32 = tf.cast(width, tf.int32)

    def _flat_index(ix, iy):
        ix = tf.cast(tf.clip_by_value(ix, 0.0, width_f - 1.0), tf.int32)
        iy = tf.cast(tf.clip_by_value(iy, 0.0, height_f - 1.0), tf.int32)
        return iy * width_i32 + ix

    indices = [
        _flat_index(ix0, iy0),
        _flat_index(ix1, iy0),
        _flat_index(ix0, iy1),
        _flat_index(ix1, iy1),
    ]

    # flat row index (in units of D-vectors) of [B, H, W, G, D] flattened
    group_index = tf.range(group, dtype=tf.int32)[None, None, None, :, None]
    batch_index = (
        tf.range(bs, dtype=tf.int32)[:, None, None, None, None]
        * tf.cast(height * width * group, tf.int32)
    )

    value_flat = tf.reshape(value, [bs * height * width * group, group_channels])

    # one gather per corner + accumulate: XLA fuses each gather with the
    # multiply-reduce much better than a single fused [.., K*4, D] gather
    output = tf.zeros([bs, hout, wout, group, group_channels], dtype=value.dtype)

    for c in range(4):
        idx_c = indices[c] * group + group_index + batch_index  # [B, Hout, Wout, G, K]
        gathered_c = tf.gather(value_flat, idx_c)  # [B, Hout, Wout, G, K, D]
        w_c = tf.cast(weights[c], gathered_c.dtype)
        output = output + tf.reduce_sum(gathered_c * w_c[..., None], axis=-2)

    output = tf.reshape(output, [bs, hout, wout, channels])

    return output


class DCNv4(Keras3_Layer_Wrapper):
    def __init__(
        self,
        kernel_size=3,
        group=4,
        stride=1,
        pad=None,
        dilation=1,
        offset_scale=1.0,
        dw_kernel_size=None,
        center_feature_scale=False,
        remove_center=False,
        output_bias=True,
        without_pointwise=False,
        use_jit_compile=False,
        **kwargs,
    ):
        super().__init__(**kwargs)

        self.kernel_size = values_to_tuple_2d(kernel_size)
        self.group = group
        self.stride = values_to_tuple_2d(stride)
        self.dilation = values_to_tuple_2d(dilation)
        self.pad = values_to_tuple_2d(pad) if pad is not None else _pad_2d_from(
            self.kernel_size, self.dilation
        )
        self.offset_scale = offset_scale
        self.dw_kernel_size = dw_kernel_size
        self.center_feature_scale = center_feature_scale
        self.remove_center = remove_center
        self.output_bias = output_bias
        self.without_pointwise = without_pointwise

        if is_keras3():
            use_jit_compile = False

        self.use_jit_compile = use_jit_compile

        kh, kw = self.kernel_size
        self.num_points = kh * kw - (1 if remove_center else 0)
        # padded to a multiple of 8, same as the official implementation
        self.offset_mask_dim = ((self.group * self.num_points * 3 + 7) // 8) * 8

        # the offset/mask predictor shares the input grid, so the layer only
        # supports resolution-preserving configs (same as the official module)
        if self.stride != (1, 1) or self.pad != _pad_2d_from(self.kernel_size, self.dilation):
            raise ValueError(
                "DCNv4 predicts offsets on the input grid, so only stride=1 with "
                f"pad=(dilation*(kernel-1))//2 is supported, got stride={self.stride}, "
                f"pad={self.pad}, kernel_size={self.kernel_size}, dilation={self.dilation}"
            )

    def build(self, input_shape):

        channels = input_shape[-1]

        if channels % self.group != 0:
            raise ValueError(
                f"channels must be divisible by group, but got {channels} and {self.group}"
            )

        self.channels = channels
        self.group_channels = channels // self.group

        if self.dw_kernel_size is not None:
            self.offset_mask_dw_kernel = self.add_weight(
                name="offset_mask_dw_kernel",
                shape=(self.dw_kernel_size, self.dw_kernel_size, channels, 1),
                initializer="glorot_uniform",
                trainable=True,
                dtype="float32",
            )
            self.offset_mask_dw_bias = self.add_weight(
                name="offset_mask_dw_bias",
                shape=(channels,),
                initializer="zeros",
                trainable=True,
                dtype="float32",
            )

        self.offset_mask_kernel = self.add_weight(
            name="offset_mask_kernel",
            shape=(channels, self.offset_mask_dim),
            initializer="zeros",
            trainable=True,
            dtype="float32",
        )
        self.offset_mask_bias = self.add_weight(
            name="offset_mask_bias",
            shape=(self.offset_mask_dim,),
            initializer="zeros",
            trainable=True,
            dtype="float32",
        )

        if not self.without_pointwise:
            self.value_proj_kernel = self.add_weight(
                name="value_proj_kernel",
                shape=(channels, channels),
                initializer="glorot_uniform",
                trainable=True,
                dtype="float32",
            )
            self.value_proj_bias = self.add_weight(
                name="value_proj_bias",
                shape=(channels,),
                initializer="zeros",
                trainable=True,
                dtype="float32",
            )
            self.output_proj_kernel = self.add_weight(
                name="output_proj_kernel",
                shape=(channels, channels),
                initializer="glorot_uniform",
                trainable=True,
                dtype="float32",
            )
            if self.output_bias:
                self.output_proj_bias = self.add_weight(
                    name="output_proj_bias",
                    shape=(channels,),
                    initializer="zeros",
                    trainable=True,
                    dtype="float32",
                )

        if self.center_feature_scale:
            self.center_feature_scale_kernel = self.add_weight(
                name="center_feature_scale_kernel",
                shape=(channels, self.group),
                initializer="zeros",
                trainable=True,
                dtype="float32",
            )
            self.center_feature_scale_bias = self.add_weight(
                name="center_feature_scale_bias",
                shape=(self.group,),
                initializer="zeros",
                trainable=True,
                dtype="float32",
            )

        super().build(input_shape)

    @tf.function(jit_compile=True, autograph=False)
    def _xla_forward(self, x):
        return self._forward(x)

    def _linear(self, x, kernel, bias=None):
        # flatten to 2D so that the weight gradient is a plain GEMM
        # (batched-matmul weight grads lower to slow reduce paths in XLA)
        out_shape = get_tensor_shape(x)[:-1] + (get_tensor_shape(kernel)[-1],)
        x = tf.reshape(x, [-1, get_tensor_shape(x)[-1]])
        x = tf.matmul(x, tf.cast(kernel, x.dtype))
        if bias is not None:
            x = tf.add(x, tf.cast(bias, x.dtype))
        x = tf.reshape(x, out_shape)
        return x

    def _forward(self, x):

        bs, height, width, channels = get_tensor_shape(x)

        hout = _output_hw(height, self.kernel_size[0], self.stride[0], self.pad[0], self.dilation[0])
        wout = _output_hw(width, self.kernel_size[1], self.stride[1], self.pad[1], self.dilation[1])

        if not self.without_pointwise:
            value = self._linear(x, self.value_proj_kernel, self.value_proj_bias)
        else:
            value = x

        if self.dw_kernel_size is not None:
            offset_mask_input = tf.nn.depthwise_conv2d(
                x,
                tf.cast(self.offset_mask_dw_kernel, x.dtype),
                strides=(1, 1, 1, 1),
                padding="SAME",
            )
            offset_mask_input = tf.add(
                offset_mask_input, tf.cast(self.offset_mask_dw_bias, x.dtype)
            )
        else:
            offset_mask_input = x

        offset_mask = self._linear(
            offset_mask_input, self.offset_mask_kernel, self.offset_mask_bias
        )

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

        if self.center_feature_scale:
            scale = self._linear(
                output, self.center_feature_scale_kernel, self.center_feature_scale_bias
            )
            scale = tf.nn.sigmoid(scale)  # [B, Hout, Wout, G]
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

    def call(self, inputs, training=None):

        if self.use_jit_compile:
            return self._xla_forward(inputs)

        return self._forward(inputs)

    def compute_output_shape(self, input_shape):

        # guarded by the stride/pad validation in __init__: output keeps the
        # input resolution
        return tuple(input_shape)

    def get_config(self):

        config = {
            "kernel_size": self.kernel_size,
            "group": self.group,
            "stride": self.stride,
            "pad": self.pad,
            "dilation": self.dilation,
            "offset_scale": self.offset_scale,
            "dw_kernel_size": self.dw_kernel_size,
            "center_feature_scale": self.center_feature_scale,
            "remove_center": self.remove_center,
            "output_bias": self.output_bias,
            "without_pointwise": self.without_pointwise,
            "use_jit_compile": self.use_jit_compile,
        }

        base_config = super().get_config()

        return dict(list(base_config.items()) + list(config.items()))
