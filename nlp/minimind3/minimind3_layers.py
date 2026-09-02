import math

import tensorflow as tf
import keras

from iseg.nlp.minimind3.minimind3_config import MiniMind3Config


class MiniMind3RMSNorm(keras.layers.Layer):
    """RMSNorm with torch semantics: `weight * x / sqrt(mean(x^2) + eps)`.

    Unlike the Gemma variant in `iseg.nlp.gemma`, the scale weight here is
    used directly (initialized to ones), not as `1 + scale`.
    """

    def __init__(self, epsilon=1e-6, **kwargs):
        super().__init__(**kwargs)
        self.epsilon = epsilon

    def build(self, input_shape):
        self.scale_weight = self.add_weight(
            name="weight",
            shape=(input_shape[-1],),
            initializer="ones",
            trainable=True,
        )
        self.built = True

    def call(self, x):
        # Always compute the normalization in float32, like the torch model.
        x_float = tf.cast(x, tf.float32)
        var = tf.reduce_mean(tf.square(x_float), axis=-1, keepdims=True)
        normed = x_float * tf.math.rsqrt(var + self.epsilon)
        normed = normed * tf.cast(self.scale_weight, tf.float32)
        return tf.cast(normed, x.dtype)


def apply_rotary_pos_emb(x, positions, rope_theta):
    """Applies rotary position embedding (rotate-half / GPT-NeoX style).

    Args:
        x: float tensor of shape [batch, seq_len, num_heads, head_dim].
        positions: float tensor of shape [seq_len].
        rope_theta: float. RoPE base frequency.

    Returns:
        Rotated tensor with the same shape as `x`.
    """
    x_shape = tf.shape(x)
    head_dim = x.shape[-1]

    freq_exponents = tf.range(0, head_dim, 2, dtype=tf.float32) / float(head_dim)
    inv_freq = 1.0 / (float(rope_theta) ** freq_exponents)  # [head_dim // 2]
    freqs = positions[:, None] * inv_freq[None, :]  # [seq_len, head_dim // 2]
    cos = tf.cos(freqs)[None, :, None, :]
    sin = tf.sin(freqs)[None, :, None, :]

    x1, x2 = tf.split(x, 2, axis=-1)
    output = tf.concat([x1 * cos - x2 * sin, x2 * cos + x1 * sin], axis=-1)
    output = tf.cast(output, x.dtype)
    return tf.reshape(output, x_shape)


class MiniMind3Attention(keras.layers.Layer):
    """Grouped query attention with per-head q/k RMSNorm, no biases."""

    def __init__(self, config: MiniMind3Config, **kwargs):
        super().__init__(**kwargs)
        self.num_heads = config.num_attention_heads
        self.num_kv_heads = config.num_key_value_heads
        self.head_dim = config.head_dim
        self.n_rep = self.num_heads // self.num_kv_heads
        self.rope_theta = config.rope_theta
        self.hidden_dim = config.hidden_size

        self.q_proj = keras.layers.Dense(
            self.num_heads * self.head_dim, use_bias=False, name="q_proj"
        )
        self.k_proj = keras.layers.Dense(
            self.num_kv_heads * self.head_dim, use_bias=False, name="k_proj"
        )
        self.v_proj = keras.layers.Dense(
            self.num_kv_heads * self.head_dim, use_bias=False, name="v_proj"
        )
        self.o_proj = keras.layers.Dense(
            self.hidden_dim, use_bias=False, name="o_proj"
        )
        self.q_norm = MiniMind3RMSNorm(epsilon=config.rms_norm_eps, name="q_norm")
        self.k_norm = MiniMind3RMSNorm(epsilon=config.rms_norm_eps, name="k_norm")

    def call(self, x):
        batch_size = tf.shape(x)[0]
        seq_len = tf.shape(x)[1]

        query = self.q_proj(x)
        key = self.k_proj(x)
        value = self.v_proj(x)

        query = tf.reshape(query, [batch_size, seq_len, self.num_heads, self.head_dim])
        key = tf.reshape(key, [batch_size, seq_len, self.num_kv_heads, self.head_dim])
        value = tf.reshape(value, [batch_size, seq_len, self.num_kv_heads, self.head_dim])

        # q/k norm is applied before RoPE, as in Qwen3 / minimind.
        query = self.q_norm(query)
        key = self.k_norm(key)

        positions = tf.range(seq_len, dtype=tf.float32)
        query = apply_rotary_pos_emb(query, positions, self.rope_theta)
        key = apply_rotary_pos_emb(key, positions, self.rope_theta)

        # GQA: repeat each kv head `n_rep` times consecutively (torch repeat_kv).
        key = tf.repeat(key, self.n_rep, axis=2)
        value = tf.repeat(value, self.n_rep, axis=2)

        # [batch, num_heads, seq_len, head_dim]
        query = tf.transpose(query, [0, 2, 1, 3])
        key = tf.transpose(key, [0, 2, 1, 3])
        value = tf.transpose(value, [0, 2, 1, 3])

        scores = tf.matmul(query, key, transpose_b=True)
        scores = scores / math.sqrt(float(self.head_dim))

        # Causal mask.
        causal_mask = tf.linalg.band_part(
            tf.ones([seq_len, seq_len], dtype=tf.bool), -1, 0
        )
        scores = tf.where(
            causal_mask[None, None, :, :],
            scores,
            tf.cast(-1e30, scores.dtype),
        )

        probs = tf.nn.softmax(tf.cast(scores, tf.float32), axis=-1)
        probs = tf.cast(probs, value.dtype)

        output = tf.matmul(probs, value)
        output = tf.transpose(output, [0, 2, 1, 3])
        output = tf.reshape(
            output, [batch_size, seq_len, self.num_heads * self.head_dim]
        )
        return self.o_proj(output)


class MiniMind3MLP(keras.layers.Layer):
    """SwiGLU feed-forward: `down_proj(silu(gate_proj(x)) * up_proj(x))`."""

    def __init__(self, config: MiniMind3Config, **kwargs):
        super().__init__(**kwargs)
        self.gate_proj = keras.layers.Dense(
            config.intermediate_size, use_bias=False, name="gate_proj"
        )
        self.up_proj = keras.layers.Dense(
            config.intermediate_size, use_bias=False, name="up_proj"
        )
        self.down_proj = keras.layers.Dense(
            config.hidden_size, use_bias=False, name="down_proj"
        )

    def call(self, x):
        return self.down_proj(tf.nn.silu(self.gate_proj(x)) * self.up_proj(x))


class MiniMind3DecoderBlock(keras.layers.Layer):
    """Pre-norm transformer block: attention + MLP with residual connections."""

    def __init__(self, config: MiniMind3Config, **kwargs):
        super().__init__(**kwargs)
        self.input_layernorm = MiniMind3RMSNorm(
            epsilon=config.rms_norm_eps, name="input_layernorm"
        )
        self.self_attn = MiniMind3Attention(config, name="self_attn")
        self.post_attention_layernorm = MiniMind3RMSNorm(
            epsilon=config.rms_norm_eps, name="post_attention_layernorm"
        )
        self.mlp = MiniMind3MLP(config, name="mlp")

    def call(self, x):
        x = x + self.self_attn(self.input_layernorm(x))
        x = x + self.mlp(self.post_attention_layernorm(x))
        return x
