# ===================================================================
# MIT License
# Copyright (c) 2024 edwardyehuang (https://github.com/edwardyehuang)
# ===================================================================

# Equivalence test: DCNv2_Flash must match DCNv2 (same weights -> same
# outputs/gradients, up to fp32 rounding).

import os
import sys

rootpath = os.path.abspath(os.path.join(os.path.dirname(__file__), os.pardir, os.pardir))
sys.path.insert(1, rootpath)

import tensorflow as tf

from iseg.layers.dcn_v2 import DCNv2
from iseg.layers.dcnv2_flash import DCNv2_Flash


def _build_pair(filters, use_custom_offset, x_shape, dilation=1):
    old = DCNv2(
        filters=filters,
        kernel_size=(3, 3),
        dilation_rate=dilation,
        use_custom_offset=use_custom_offset,
    )
    new = DCNv2_Flash(
        filters=filters,
        kernel_size=(3, 3),
        dilation_rate=dilation,
        use_custom_offset=use_custom_offset,
    )
    if use_custom_offset:
        offset_shape = (x_shape[0], x_shape[1], x_shape[2], x_shape[-1])
        old.build([x_shape, offset_shape])
        new.build([x_shape, offset_shape])
    else:
        old.build(x_shape)
        new.build(x_shape)
    for v_new, v_old in zip(new.trainable_variables, old.trainable_variables):
        v_new.assign(v_old)
    return old, new


def check_equivalence(B=2, H=32, W=24, C=64, F=64, use_custom_offset=False, dilation=1, jit=False):
    x = tf.random.normal([B, H, W, C])
    inputs = [x, x] if use_custom_offset else x
    old, new = _build_pair(F, use_custom_offset, (B, H, W, C), dilation)
    gout = tf.random.normal([B, H, W, F])

    results = {}
    for name, layer in [("old", old), ("new", new)]:
        fn = layer
        if jit:
            fn = tf.function(jit_compile=True)(lambda inp, layer=layer: layer(inp))
        with tf.GradientTape() as tape:
            for t in (inputs if isinstance(inputs, list) else [inputs]):
                tape.watch(t)
            y = fn(inputs)
            loss = tf.reduce_sum(y * gout)
        grads = tape.gradient(loss, (inputs if isinstance(inputs, list) else [inputs]) + layer.trainable_variables)
        results[name] = (y, grads)

    dy = float(tf.reduce_max(tf.abs(results["old"][0] - results["new"][0])))
    assert dy < 5e-3, f"forward mismatch: {dy}"
    for a, b in zip(results["old"][1], results["new"][1]):
        dg = float(tf.reduce_max(tf.abs(a - b)))
        # TF32 matmul rounding shows up on large-reduction weight grads
        scale = float(tf.reduce_max(tf.abs(a)))
        assert dg <= max(5e-3, 2e-3 * scale), f"gradient mismatch: {dg} (scale {scale})"


def test_dcnv2_flash_equivalence():
    for jit in [False, True]:
        check_equivalence(use_custom_offset=False, jit=jit)
        check_equivalence(use_custom_offset=True, jit=jit)
        check_equivalence(dilation=2, jit=jit)
        check_equivalence(H=17, W=19, F=48, jit=jit)
    print("dcnv2_flash equivalence: OK")


if __name__ == "__main__":
    test_dcnv2_flash_equivalence()
