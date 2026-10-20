# ===================================================================
# MIT License
# Copyright (c) 2026 edwardyehuang (https://github.com/edwardyehuang)
# ===================================================================

# Single-scale deformable attention (Deformable DETR style) for NHWC feature
# maps (https://github.com/fundamentalvision/Deformable-DETR).
# Pure TF ops (no custom op), XLA JIT compatible.
#
# Each query position predicts, per head, num_points sampling offsets and
# their attention weights; the output is the attention-weighted sum of the
# values bilinearly sampled at the offset locations. Complexity is
# O(B * Hq * Wq * heads * points * head_channels) instead of quadratic.
#
# Numerics follow the official MSDeformAttn module:
#   - offsets and attention weights are predicted from the query; the offset
#     projection kernel and both projection biases are zero-initialized,
#     except the offset bias which uses the per-head spiral pattern
#     (radius = point index + 1, in pixels), so sampling starts local
#   - offsets are in pixel units, relative to the query position
#   - attention weights: softmax over the sampling points of each head
#   - bilinear sampling with zero padding: out-of-bound corners contribute 0
#     and the weights are not renormalized (F.grid_sample semantics)

import math

import tensorflow as tf
import keras

from iseg.utils import get_tensor_shape
from iseg.utils.version_utils import is_keras3
from iseg.utils.keras3_utils import Keras3_Layer_Wrapper
from iseg.utils.op_utils import safed_softmax


def _offset_bias_value(num_heads, num_points):
    # per-head rotation spiral from MSDeformAttn._reset_parameters, stored as
    # (dy, dx) = (sin, cos) * (point index + 1) in pixel units
    values = []
    for m in range(num_heads):
        theta = 2.0 * math.pi * m / num_heads
        dy, dx = math.sin(theta), math.cos(theta)
        scale = max(abs(dy), abs(dx))
        dy, dx = dy / scale, dx / scale
        for k in range(num_points):
            values.extend([dy * (k + 1), dx * (k + 1)])
    return values


def deformable_attention_core(value, offsets, attn_weights, num_heads):
    """Core deformable aggregation of deformable attention.

    value:        [B, Hv, Wv, C] with C = num_heads * head_channels
    offsets:      [B, Hq, Wq, num_heads, num_points, 2], pixel offsets (dy, dx)
                  relative to the query position
    attn_weights: [B, Hq, Wq, num_heads, num_points], softmaxed over num_points
    returns:      [B, Hq, Wq, C]
    """

    bs, hv, wv, channels = get_tensor_shape(value)
    hq, wq = get_tensor_shape(offsets)[1:3]

    head_channels = channels // num_heads

    # grid/weight math in float32 (like the official CUDA opmath_t),
    # or float64 when running in float64
    compute_dtype = tf.float64 if value.dtype == tf.float64 else tf.float32

    offsets = tf.cast(offsets, compute_dtype)
    attn_weights = tf.cast(attn_weights, compute_dtype)

    # reference grid: query pixel centers, offsets are in pixel units
    qy = tf.range(hq, dtype=compute_dtype)  # [Hq]
    qx = tf.range(wq, dtype=compute_dtype)  # [Wq]

    y_pos = qy[None, :, None, None, None] + offsets[..., 0]  # [B, Hq, Wq, M, K]
    x_pos = qx[None, None, :, None, None] + offsets[..., 1]  # [B, Hq, Wq, M, K]

    ix0 = tf.floor(x_pos)
    iy0 = tf.floor(y_pos)
    ix1 = ix0 + 1.0
    iy1 = iy0 + 1.0

    lx = x_pos - ix0
    ly = y_pos - iy0
    hx = 1.0 - lx
    hy = 1.0 - ly

    hv_f = tf.cast(hv, compute_dtype)
    wv_f = tf.cast(wv, compute_dtype)

    # out-of-bound corners contribute 0 (zero padding, no renormalization);
    # both bounds are checked per corner: offsets are unbounded, so a
    # sampling point can fall arbitrarily far outside the value map (unlike
    # dcnv4_core, there is no whole-point validity mask to rely on)
    ix0_ok = (ix0 >= 0.0) & (ix0 < wv_f)
    ix1_ok = (ix1 >= 0.0) & (ix1 < wv_f)
    iy0_ok = (iy0 >= 0.0) & (iy0 < hv_f)
    iy1_ok = (iy1 >= 0.0) & (iy1 < hv_f)

    c1_ok = tf.cast(iy0_ok & ix0_ok, compute_dtype)
    c2_ok = tf.cast(iy0_ok & ix1_ok, compute_dtype)
    c3_ok = tf.cast(iy1_ok & ix0_ok, compute_dtype)
    c4_ok = tf.cast(iy1_ok & ix1_ok, compute_dtype)

    weights = [
        attn_weights * (hy * hx * c1_ok),
        attn_weights * (hy * lx * c2_ok),
        attn_weights * (ly * hx * c3_ok),
        attn_weights * (ly * lx * c4_ok),
    ]

    # gather row indices into value reshaped as [B * Hv * Wv * M, D]
    wv_i32 = tf.cast(wv, tf.int32)

    def _flat_index(ix, iy):
        ix = tf.cast(tf.clip_by_value(ix, 0.0, wv_f - 1.0), tf.int32)
        iy = tf.cast(tf.clip_by_value(iy, 0.0, hv_f - 1.0), tf.int32)
        return iy * wv_i32 + ix

    indices = [
        _flat_index(ix0, iy0),
        _flat_index(ix1, iy0),
        _flat_index(ix0, iy1),
        _flat_index(ix1, iy1),
    ]

    head_index = tf.range(num_heads, dtype=tf.int32)[None, None, None, :, None]
    batch_index = tf.range(bs, dtype=tf.int32)[:, None, None, None, None] * tf.cast(
        hv * wv * num_heads, tf.int32
    )

    value_flat = tf.reshape(value, [bs * hv * wv * num_heads, head_channels])

    # one gather per corner + accumulate: XLA fuses each gather with the
    # multiply-reduce much better than a single fused [.., K*4, D] gather
    output = tf.zeros([bs, hq, wq, num_heads, head_channels], dtype=value.dtype)

    for c in range(4):
        idx_c = indices[c] * num_heads + head_index + batch_index  # [B, Hq, Wq, M, K]
        gathered_c = tf.gather(value_flat, idx_c)  # [B, Hq, Wq, M, K, D]
        w_c = tf.cast(weights[c], gathered_c.dtype)
        output = output + tf.reduce_sum(gathered_c * w_c[..., None], axis=-2)

    output = tf.reshape(output, [bs, hq, wq, channels])

    return output


class DeformableAttention(Keras3_Layer_Wrapper):
    """Single-scale deformable attention (Deformable DETR style).

    call(inputs)         -> self-attention on the [B, H, W, C] map
    call([query, value]) -> the query map attends to a value map, possibly
                            with a different resolution and channel count

    filters: value/output channels; -1 keeps the query channel count.
    """

    def __init__(
        self,
        filters=-1,
        num_heads=8,
        num_points=4,
        apply_linear=True,
        apply_output_projection=True,
        dropout_rate=0.0,
        use_jit_compile=False,
        **kwargs,
    ):
        super().__init__(**kwargs)

        self.filters = filters
        self.num_heads = num_heads
        self.num_points = num_points
        self.apply_linear = apply_linear
        self.apply_output_projection = apply_output_projection
        self.dropout_rate = dropout_rate

        if is_keras3():
            use_jit_compile = False

        self.use_jit_compile = use_jit_compile

    def build(self, input_shape):

        query_shape, value_shape = self._unpack_input_shapes(input_shape)

        query_channels = query_shape[-1]
        value_channels = value_shape[-1]

        value_filters = query_channels if self.filters == -1 else self.filters

        if value_filters % self.num_heads != 0:
            raise ValueError(
                f"filters ({value_filters}) must be divisible by num_heads ({self.num_heads})"
            )

        if not self.apply_linear and value_channels != value_filters:
            raise ValueError(
                f"apply_linear=False requires value channels ({value_channels}) "
                f"to equal filters ({value_filters})"
            )

        self.value_filters = value_filters

        if self.apply_linear:
            self.value_proj_kernel = self.add_weight(
                name="value_proj_kernel",
                shape=(value_channels, value_filters),
                initializer="glorot_uniform",
                trainable=True,
                dtype="float32",
            )
            self.value_proj_bias = self.add_weight(
                name="value_proj_bias",
                shape=(value_filters,),
                initializer="zeros",
                trainable=True,
                dtype="float32",
            )

        self.offset_kernel = self.add_weight(
            name="offset_kernel",
            shape=(query_channels, self.num_heads * self.num_points * 2),
            initializer="zeros",
            trainable=True,
            dtype="float32",
        )
        self.offset_bias = self.add_weight(
            name="offset_bias",
            shape=(self.num_heads * self.num_points * 2,),
            initializer=keras.initializers.Constant(
                _offset_bias_value(self.num_heads, self.num_points)
            ),
            trainable=True,
            dtype="float32",
        )

        self.attn_kernel = self.add_weight(
            name="attn_kernel",
            shape=(query_channels, self.num_heads * self.num_points),
            initializer="zeros",
            trainable=True,
            dtype="float32",
        )
        self.attn_bias = self.add_weight(
            name="attn_bias",
            shape=(self.num_heads * self.num_points,),
            initializer="zeros",
            trainable=True,
            dtype="float32",
        )

        if self.apply_output_projection:
            self.output_proj_kernel = self.add_weight(
                name="output_proj_kernel",
                shape=(value_filters, value_filters),
                initializer="glorot_uniform",
                trainable=True,
                dtype="float32",
            )
            self.output_proj_bias = self.add_weight(
                name="output_proj_bias",
                shape=(value_filters,),
                initializer="zeros",
                trainable=True,
                dtype="float32",
            )

        self.dropout_layer = keras.layers.Dropout(self.dropout_rate)

        super().build(input_shape)

    @staticmethod
    def _unpack_input_shapes(input_shape):
        if (
            isinstance(input_shape, (list, tuple))
            and len(input_shape) == 2
            and isinstance(input_shape[0], (list, tuple, tf.TensorShape))
        ):
            return input_shape[0], input_shape[1]
        return input_shape, input_shape

    @tf.function(jit_compile=True, autograph=False)
    def _xla_forward(self, query, value, training=None):
        return self._forward(query, value, training=training)

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

    def _forward(self, query, value, training=None):

        bs, hq, wq, _ = get_tensor_shape(query)

        if self.apply_linear:
            value = self._linear(value, self.value_proj_kernel, self.value_proj_bias)

        offsets = self._linear(query, self.offset_kernel, self.offset_bias)
        offsets = tf.reshape(
            offsets, [bs, hq, wq, self.num_heads, self.num_points, 2]
        )  # [B, Hq, Wq, M, K, 2], (dy, dx) in pixels

        attn_logits = self._linear(query, self.attn_kernel, self.attn_bias)
        attn_logits = tf.reshape(
            attn_logits, [bs, hq, wq, self.num_heads, self.num_points]
        )
        attn_weights = safed_softmax(attn_logits)  # [B, Hq, Wq, M, K]
        attn_weights = self.dropout_layer(attn_weights, training=training)

        output = deformable_attention_core(value, offsets, attn_weights, self.num_heads)

        if self.apply_output_projection:
            output = self._linear(
                output, self.output_proj_kernel, self.output_proj_bias
            )

        return output

    def call(self, inputs, training=None):

        if isinstance(inputs, (list, tuple)):
            query, value = tuple(inputs)
        else:
            query = value = inputs

        if self.use_jit_compile:
            return self._xla_forward(query, value, training=training)

        return self._forward(query, value, training=training)

    def compute_output_shape(self, input_shape):

        query_shape, _ = self._unpack_input_shapes(input_shape)

        out_channels = query_shape[-1] if self.filters == -1 else self.filters

        return tuple(query_shape[:-1]) + (out_channels,)

    def get_config(self):

        config = {
            "filters": self.filters,
            "num_heads": self.num_heads,
            "num_points": self.num_points,
            "apply_linear": self.apply_linear,
            "apply_output_projection": self.apply_output_projection,
            "dropout_rate": self.dropout_rate,
            "use_jit_compile": self.use_jit_compile,
        }

        base_config = super().get_config()

        return dict(list(base_config.items()) + list(config.items()))
