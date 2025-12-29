import tensorflow as tf
import keras
import math

from iseg.utils import get_tensor_shape
from iseg import check_numerics
from iseg.utils.op_utils import replace_nan_or_inf
from iseg.utils.version_utils import is_keras3
from iseg.utils.keras3_utils import Keras3_Model_Wrapper
from iseg.utils.op_utils import safed_softmax


class DeformableMultiHeadSelfAttentionLayer(Keras3_Model_Wrapper):
    """
    Deformable multi-head self-attention for 2D feature maps.
    - Learns per-head, per-point 2D offsets and attention weights from the query.
    - Samples value features via bilinear interpolation at offset locations around each query position.
    - Aggregates per-point samples using softmax-normalized attention weights.
    Complexity: O(N * H * W * heads * num_points), typically much less than quadratic.
    """

    def __init__(
        self,
        filters=-1,
        num_heads=4,
        num_points=4,
        apply_linear=True,
        shared_qk=False,  # kept for API symmetry; not used here
        trainable=True,
        use_dense_for_linear=False,
        offset_range_factor=8.0,  # offsets are bounded to +/- (H/offset_range_factor, W/offset_range_factor)
        use_jit_compile=False,
        name=None,
    ):
        super().__init__(trainable=trainable, name=name)

        self.filters = filters
        self.num_heads = num_heads
        self.num_points = num_points
        self.apply_linear = apply_linear
        self.shared_qk = shared_qk
        self.use_dense_for_linear = use_dense_for_linear
        self.offset_range_factor = float(offset_range_factor)

        if is_keras3():
            self.use_jit_compile = False
        else:
            self.use_jit_compile = use_jit_compile

        if tf.distribute.has_strategy():
            strategy = tf.distribute.get_strategy()
            if isinstance(strategy, tf.distribute.TPUStrategy):
                self.use_jit_compile = False

        # Layers will be created in build
        self.value_proj = None
        self.offset_proj = None
        self.attn_proj = None

    def _make_linear_1x1(self, out_channels, name):
        if self.use_dense_for_linear:
            return keras.layers.Dense(out_channels, name=name, trainable=self.trainable)
        else:
            return keras.layers.Conv2D(out_channels, (1, 1), name=name, trainable=self.trainable)

    def build(self, input_shape):
        channels = int(input_shape[-1])
        value_filters = channels if self.filters == -1 else int(self.filters)

        if value_filters % self.num_heads != 0:
            raise ValueError(
                f"value filters ({value_filters}) must be divisible by num_heads ({self.num_heads})."
            )

        if self.apply_linear:
            self.value_proj = self._make_linear_1x1(value_filters, name="value_proj")

        # Predict 2D offsets per head and per point
        self.offset_proj = self._make_linear_1x1(
            self.num_heads * self.num_points * 2, name="offset_proj"
        )
        # Predict attention logits per head and per point
        self.attn_proj = self._make_linear_1x1(
            self.num_heads * self.num_points, name="attn_proj"
        )

        super().build(input_shape)

    def _compute_sampling_grid(self, height, width, batch_size, heads, num_points, dtype):
        # Base coordinates grid [1, H, W, 1, 1]
        y_base = tf.range(height, dtype=dtype)
        x_base = tf.range(width, dtype=dtype)
        y_base = tf.reshape(y_base, [1, height, 1, 1, 1])
        x_base = tf.reshape(x_base, [1, 1, width, 1, 1])
        y_base = tf.broadcast_to(y_base, [1, height, width, 1, 1])
        x_base = tf.broadcast_to(x_base, [1, height, width, 1, 1])
        # Broadcast to [N, H, W, heads, points]
        y_base = tf.broadcast_to(y_base, [batch_size, height, width, heads, num_points])
        x_base = tf.broadcast_to(x_base, [batch_size, height, width, heads, num_points])
        return y_base, x_base

    def _bilinear_sample(self, value, y, x):
        """
        Bilinear sample value at fractional coords (y, x).
        value: [N, H, W, heads, C]
        y,x:   [N, H, W, heads, P]
        returns: [N, H, W, heads, P, C]
        """
        N, H, W, heads, C = get_tensor_shape(value)
        P = int(x.shape[-1])

        # Prepare [B, H, W, C]
        value = tf.transpose(value, [0, 3, 1, 2, 4])         # [N, heads, H, W, C]
        B = N * heads
        value = tf.reshape(value, [B, H, W, C])               # [B, H, W, C]

        # Reshape coords to [B, H, W, P]
        y = tf.transpose(y, [0, 3, 1, 2, 4])                  # [N, heads, H, W, P]
        x = tf.transpose(x, [0, 3, 1, 2, 4])
        y = tf.reshape(y, [B, H, W, P])
        x = tf.reshape(x, [B, H, W, P])

        # Clip to bounds for interpolation neighbors
        y0 = tf.floor(y)
        x0 = tf.floor(x)
        y1 = y0 + 1.0
        x1 = x0 + 1.0

        y0c = tf.clip_by_value(tf.cast(y0, tf.int32), 0, H - 1)
        x0c = tf.clip_by_value(tf.cast(x0, tf.int32), 0, W - 1)
        y1c = tf.clip_by_value(tf.cast(y1, tf.int32), 0, H - 1)
        x1c = tf.clip_by_value(tf.cast(x1, tf.int32), 0, W - 1)

        wy1 = y - y0
        wx1 = x - x0
        wy0 = 1.0 - wy1
        wx0 = 1.0 - wx1

        def gather_at(y_idx, x_idx):
            # y_idx, x_idx: [B, H, W, P] int32
            L = H * W * P
            # Flatten
            y_flat = tf.reshape(y_idx, [B, L])
            x_flat = tf.reshape(x_idx, [B, L])
            b_flat = tf.reshape(
                tf.broadcast_to(tf.reshape(tf.range(B, dtype=tf.int32), [B, 1]), [B, L]),
                [B, L],
            )
            idx = tf.stack([b_flat, y_flat, x_flat], axis=-1)  # [B, L, 3]
            idx = tf.reshape(idx, [B * L, 3])                  # [B*L, 3]
            gathered = tf.gather_nd(value, idx)                # [B*L, C]
            return tf.reshape(gathered, [B, H, W, P, C])       # [B, H, W, P, C]

        v00 = gather_at(y0c, x0c)
        v01 = gather_at(y0c, x1c)
        v10 = gather_at(y1c, x0c)
        v11 = gather_at(y1c, x1c)

        wy0 = tf.expand_dims(wy0, axis=-1)  # [B, H, W, P, 1]
        wy1 = tf.expand_dims(wy1, axis=-1)
        wx0 = tf.expand_dims(wx0, axis=-1)
        wx1 = tf.expand_dims(wx1, axis=-1)

        w00 = wy0 * wx0
        w01 = wy0 * wx1
        w10 = wy1 * wx0
        w11 = wy1 * wx1

        out = w00 * v00 + w01 * v01 + w10 * v10 + w11 * v11   # [B, H, W, P, C]

        # Restore to [N, H, W, heads, P, C]
        out = tf.reshape(out, [N, heads, H, W, P, C])
        out = tf.transpose(out, [0, 2, 3, 1, 4, 5])
        return out

    def compute_attention_internal(self, query, value):
        # Shapes
        batch_size, height, width, _ = get_tensor_shape(query)
        dtype = query.dtype

        # Replace NaN/Inf
        query = replace_nan_or_inf(query, keras.backend.epsilon())
        value = replace_nan_or_inf(value, keras.backend.epsilon())

        query = check_numerics(query, "query contains NaN/Inf", level=1)
        value = check_numerics(value, "value contains NaN/Inf", level=1)

        # Project value if needed
        if self.apply_linear:
            value = self.value_proj(value)  # [N, H, W, Cv]

        Cv = int(value.shape[-1])
        C_head = Cv // self.num_heads

        # Predict offsets: [N, H, W, heads*points*2] -> [N, H, W, heads, points, 2]
        offsets = self.offset_proj(query)
        offsets = tf.reshape(
            offsets, [batch_size, height, width, self.num_heads, self.num_points, 2]
        )

        # Bound offsets (tanh) within +- (H/alpha, W/alpha)
        # Scale separately for y (h) and x (w)
        offset_scale_y = tf.cast(height / self.offset_range_factor, dtype)
        offset_scale_x = tf.cast(width / self.offset_range_factor, dtype)
        offsets = tf.tanh(offsets)
        dy = offsets[..., 0] * offset_scale_y
        dx = offsets[..., 1] * offset_scale_x

        # Attention logits -> weights across points
        attn_logits = self.attn_proj(query)  # [N, H, W, heads*points]
        attn_logits = tf.reshape(
            attn_logits, [batch_size, height, width, self.num_heads, self.num_points]
        )
        attn_weights = safed_softmax(attn_logits, axis=-1)  # [N, H, W, heads, points]
        attn_weights = replace_nan_or_inf(attn_weights, keras.backend.epsilon())
        attn_weights = check_numerics(attn_weights, "attn_weights contains NaN/Inf", level=1)

        # Split value into heads
        value = tf.reshape(value, [batch_size, height, width, self.num_heads, C_head])  # [N,H,W,Hd,C]

        # Build sampling positions: base + offsets
        y_base, x_base = self._compute_sampling_grid(
            height, width, batch_size, self.num_heads, self.num_points, dtype
        )
        y = y_base + dy  # [N,H,W,heads,points]
        x = x_base + dx  # [N,H,W,heads,points]

        # Clip to valid range
        y = tf.clip_by_value(y, 0.0, tf.cast(height - 1, dtype))
        x = tf.clip_by_value(x, 0.0, tf.cast(width - 1, dtype))

        # Bilinear sampling
        sampled = self._bilinear_sample(value, y, x)  # [N, H, W, heads, points, C_head]

        # Aggregate with attention weights over points
        attn = tf.expand_dims(attn_weights, axis=-1)          # [N, H, W, heads, points, 1]
        out = tf.reduce_sum(sampled * attn, axis=-2)          # [N, H, W, heads, C_head]

        # Concatenate heads
        out = tf.reshape(out, [batch_size, height, width, self.num_heads * C_head])  # [N,H,W,Cv]

        out = replace_nan_or_inf(out, keras.backend.epsilon())
        out = check_numerics(out, "output contains NaN/Inf", level=1)
        return out

    @tf.function(jit_compile=True, autograph=False)
    def compute_attention_xla(self, query, value):
        return self.compute_attention_internal(query, value)

    def compute_attention(self, query, value):
        if self.use_jit_compile:
            return self.compute_attention_xla(query, value)
        else:
            return self.compute_attention_internal(query, value)

    def call(self, inputs, key=None, value=None, training=None):
        query = inputs
        if value is None:
            value = query
        return self.compute_attention(query, value)
