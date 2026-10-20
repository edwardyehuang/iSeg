# ===================================================================
# MIT License
# Copyright (c) 2026 edwardyehuang (https://github.com/edwardyehuang)
# ===================================================================

# Self-contained checks for iseg.layers.deformable_attention:
#   - core vs an independent naive gather_nd reference (float64)
#   - XLA vs eager consistency
#   - zeroed offsets degenerate to an identity aggregation
#   - cross-attention with a different value resolution/channels
#   - gradients are finite
#   - training smoke test
#   - get_config round-trip
#
#   python iseg/layers/deformable_attention/test_deformable_attention.py

import os
import sys

import numpy as np

rootpath = os.path.abspath(
    os.path.join(os.path.dirname(__file__), os.pardir, os.pardir, os.pardir)
)
sys.path.insert(1, rootpath)

import tensorflow as tf

from iseg.layers.deformable_attention import (
    DeformableAttention,
    deformable_attention_core,
)


def naive_reference(value, offsets, attn_weights, num_heads):
    """Independent gather_nd + per-corner loop reference (float64)."""

    B, Hv, Wv, C = [int(d) for d in value.shape]
    Hq, Wq = int(offsets.shape[1]), int(offsets.shape[2])
    K = int(offsets.shape[4])
    D = C // num_heads

    value = tf.cast(tf.reshape(value, [B, Hv, Wv, num_heads, D]), tf.float64)
    offsets = tf.cast(offsets, tf.float64)
    attn_weights = tf.cast(attn_weights, tf.float64)

    qy = tf.range(Hq, dtype=tf.float64)[None, :, None, None]
    qx = tf.range(Wq, dtype=tf.float64)[None, None, :, None]

    bb = tf.broadcast_to(
        tf.range(B, dtype=tf.int64)[:, None, None, None], [B, Hq, Wq, num_heads]
    )
    mm = tf.broadcast_to(
        tf.range(num_heads, dtype=tf.int64)[None, None, None, :], [B, Hq, Wq, num_heads]
    )

    out = tf.zeros([B, Hq, Wq, num_heads, D], dtype=tf.float64)

    for k in range(K):
        y = qy + offsets[..., k, 0]  # [B, Hq, Wq, M]
        x = qx + offsets[..., k, 1]
        y0 = tf.floor(y)
        x0 = tf.floor(x)
        wy = y - y0
        wx = x - x0

        corners = [
            (y0, x0, (1.0 - wy) * (1.0 - wx)),
            (y0, x0 + 1.0, (1.0 - wy) * wx),
            (y0 + 1.0, x0, wy * (1.0 - wx)),
            (y0 + 1.0, x0 + 1.0, wy * wx),
        ]

        for cy, cx, w in corners:
            valid = (cy >= 0.0) & (cy <= Hv - 1.0) & (cx >= 0.0) & (cx <= Wv - 1.0)
            iy = tf.clip_by_value(tf.cast(cy, tf.int64), 0, Hv - 1)
            ix = tf.clip_by_value(tf.cast(cx, tf.int64), 0, Wv - 1)
            idx = tf.stack([bb, iy, ix, mm], axis=-1)  # [B, Hq, Wq, M, 4]
            gathered = tf.gather_nd(value, idx)  # [B, Hq, Wq, M, D]
            w = w * tf.cast(valid, tf.float64)
            out = out + gathered * (attn_weights[..., k] * w)[..., None]

    return tf.reshape(out, [B, Hq, Wq, C])


def check_core_vs_naive():
    print("core vs naive reference (float64)")

    rng = np.random.default_rng(0)

    for (B, Hq, Wq, Hv, Wv, C, M, K) in [
        (2, 8, 6, 8, 6, 8, 4, 4),     # self-attention layout
        (2, 8, 6, 16, 12, 8, 4, 4),   # cross-resolution value
        (1, 5, 5, 5, 5, 6, 1, 3),     # single head
        (2, 4, 7, 9, 5, 16, 8, 1),    # single point
    ]:
        value = rng.normal(size=(B, Hv, Wv, C)).astype("float64")
        # large offsets on purpose: some sampling points fall out of bounds
        offsets = rng.normal(scale=3.0, size=(B, Hq, Wq, M, K, 2)).astype("float64")
        logits = rng.normal(size=(B, Hq, Wq, M, K)).astype("float64")
        attn = tf.nn.softmax(logits, axis=-1).numpy()

        ref = naive_reference(value, offsets, attn, M)
        out = deformable_attention_core(
            tf.convert_to_tensor(value),
            tf.convert_to_tensor(offsets),
            tf.convert_to_tensor(attn),
            M,
        )

        max_abs = np.max(np.abs(out.numpy() - ref.numpy()))
        status = "OK " if max_abs < 1e-8 else "FAIL"
        print(
            f"  [{status}] B{B} q{Hq}x{Wq} v{Hv}x{Wv} C{C} M{M} K{K}: max_abs={max_abs:.3e}"
        )
        assert max_abs < 1e-8


def check_xla_vs_eager():
    print("xla vs eager")

    x = tf.random.normal([2, 16, 12, 32])

    for cfg in [
        dict(),
        dict(num_heads=4, num_points=8),
        dict(apply_output_projection=False),
        dict(filters=64),
        dict(dropout_rate=0.1),
    ]:
        layer_eager = DeformableAttention(use_jit_compile=False, **cfg)
        y_eager = layer_eager(x, training=False)

        layer_xla = DeformableAttention(use_jit_compile=True, **cfg)
        layer_xla(x)
        for v_xla, v_eager in zip(
            layer_xla.trainable_variables, layer_eager.trainable_variables
        ):
            v_xla.assign(v_eager)
        y_xla = layer_xla(x, training=False)

        max_abs = np.max(np.abs(y_xla.numpy() - y_eager.numpy()))
        # TF32 matmuls make XLA and eager differ by ~1e-3 on fp32
        status = "OK " if max_abs < 2e-3 else "FAIL"
        print(f"  [{status}] {cfg}: max_abs={max_abs:.3e}")
        assert max_abs < 2e-3


def check_identity_init():
    print("zeroed offsets degenerate to identity aggregation")

    x = tf.random.normal([2, 8, 6, 16])
    layer = DeformableAttention(
        num_heads=4,
        num_points=4,
        apply_linear=False,
        apply_output_projection=False,
    )
    layer(x)  # build
    # spiral bias -> local sampling; zeroing it samples exactly at the query
    # position, and uniform softmax weights sum to 1 -> output == input
    layer.offset_bias.assign(tf.zeros_like(layer.offset_bias))
    y = layer(x)

    max_abs = np.max(np.abs(y.numpy() - x.numpy()))
    status = "OK " if max_abs < 1e-5 else "FAIL"
    print(f"  [{status}] max_abs={max_abs:.3e}")
    assert max_abs < 1e-5


def check_cross_attention():
    print("cross-attention (different value resolution/channels)")

    query = tf.random.normal([2, 8, 8, 16])
    value = tf.random.normal([2, 16, 12, 24])

    for jit in [False, True]:
        layer = DeformableAttention(num_heads=4, num_points=4, use_jit_compile=jit)
        q_var = tf.Variable(query.numpy())
        v_var = tf.Variable(value.numpy())
        with tf.GradientTape() as tape:
            y = layer([q_var, v_var])
            loss = tf.reduce_mean(y)
        grads = tape.gradient(loss, layer.trainable_variables + [q_var, v_var])
        assert y.shape == (2, 8, 8, 16), y.shape
        assert all(
            g is not None and tf.reduce_all(tf.math.is_finite(g)) for g in grads
        )

    # filters=-1 follows the query channels even with different value channels
    assert layer.value_filters == 16
    print("  [OK ] output shape and finite grads")


def check_train_smoke():
    print("train smoke")

    x = tf.random.normal([2, 32, 24, 64])
    layer = DeformableAttention(num_heads=8, num_points=4)
    target = tf.random.normal([2, 32, 24, 64])
    optimizer = tf.keras.optimizers.SGD(learning_rate=0.05)

    losses = []
    for step in range(30):
        with tf.GradientTape() as tape:
            y = layer(x, training=True)
            loss = tf.reduce_mean(tf.square(y - target))
        grads = tape.gradient(loss, layer.trainable_variables)
        assert all(
            g is not None and tf.reduce_all(tf.math.is_finite(g)) for g in grads
        )
        optimizer.apply_gradients(zip(grads, layer.trainable_variables))
        losses.append(float(loss))

    print(f"  loss {losses[0]:.4f} -> {losses[-1]:.4f}")
    assert losses[-1] < losses[0], "loss did not decrease"


def check_config_roundtrip():
    print("get_config round-trip")

    layer = DeformableAttention(
        filters=48,
        num_heads=4,
        num_points=8,
        apply_linear=True,
        apply_output_projection=False,
        dropout_rate=0.1,
        use_jit_compile=False,
        name="deformable_attention_test",
    )
    layer2 = DeformableAttention.from_config(layer.get_config())

    for attr in [
        "filters",
        "num_heads",
        "num_points",
        "apply_linear",
        "apply_output_projection",
        "dropout_rate",
        "use_jit_compile",
        "name",
    ]:
        assert getattr(layer, attr) == getattr(layer2, attr), attr

    print("  [OK ]")


def main():
    check_core_vs_naive()
    check_xla_vs_eager()
    check_identity_init()
    check_cross_attention()
    check_train_smoke()
    check_config_roundtrip()
    print("all done")


if __name__ == "__main__":
    main()
