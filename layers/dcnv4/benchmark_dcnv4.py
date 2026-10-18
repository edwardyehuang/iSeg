# ===================================================================
# MIT License
# Copyright (c) 2024 edwardyehuang (https://github.com/edwardyehuang)
# ===================================================================

# Benchmark: iseg.layers.dcnv4.DCNv4 vs iseg.layers.dcn_v2.DCNv2 (XLA).
#
#   python iseg/layers/dcnv4/benchmark_dcnv4.py

import os
import sys
import time

rootpath = os.path.abspath(os.path.join(os.path.dirname(__file__), os.pardir, os.pardir, os.pardir))
sys.path.insert(1, rootpath)

import tensorflow as tf

from iseg.layers.dcnv4 import DCNv4
from iseg.layers.dcn_v2 import DCNv2


def make_fwd(layer, xla):
    def fn(x):
        return layer(x)

    if xla:
        fn = tf.function(jit_compile=True)(fn)
    return fn


def make_fwd_bwd(layer, xla):
    def fn(x):
        with tf.GradientTape() as tape:
            y = layer(x)
            loss = tf.reduce_mean(tf.square(y))
        return tape.gradient(loss, layer.trainable_variables)

    if xla:
        fn = tf.function(jit_compile=True)(fn)
    return fn


def bench(fn, x, warmup=10, iters=50):
    for _ in range(warmup):
        r = fn(x)
    _sync(r)
    t0 = time.perf_counter()
    for _ in range(iters):
        r = fn(x)
    _sync(r)
    dt = (time.perf_counter() - t0) / iters
    return dt * 1000.0  # ms


def _sync(r):
    if isinstance(r, (list, tuple)):
        for t in r:
            _ = tf.reduce_sum(t).numpy()
    else:
        _ = tf.reduce_sum(r).numpy()


def bench_config(B, H, W, C, group=4):
    print(f"\n== B={B} H={H} W={W} C={C} group={group} ==")

    x = tf.random.normal([B, H, W, C])

    layers = {
        "dcnv2": DCNv2(filters=C, kernel_size=(3, 3)),
        "dcnv4": DCNv4(kernel_size=3, group=group),
    }

    # realistic (non-zero) offsets: zero-init offset weights would make all
    # sampling positions fall on the base grid -> unrealistically cache-friendly
    # gathers. randomize offset-producing weights for a realistic access pattern.
    for layer in layers.values():
        y = layer(x)  # build
        for v in layer.trainable_variables:
            if "offset" in v.name and "kernel" in v.name:
                v.assign(tf.random.normal(v.shape, stddev=0.05))

    results = {}

    for name, layer in layers.items():
        for xla in [False, True]:
            key = f"{name} {'xla' if xla else 'eager'}"

            fwd = make_fwd(layer, xla)
            t_fwd = bench(fwd, x)

            fwd_bwd = make_fwd_bwd(layer, xla)
            t_fwd_bwd = bench(fwd_bwd, x)

            results[key] = (t_fwd, t_fwd_bwd)
            print(f"  {key:14s} fwd={t_fwd:8.3f} ms   fwd+bwd={t_fwd_bwd:8.3f} ms")

    for mode, i in [("fwd", 0), ("fwd+bwd", 1)]:
        speedup = results["dcnv2 xla"][i] / results["dcnv4 xla"][i]
        print(f"  >> XLA {mode}: dcnv4 is {speedup:.2f}x vs dcnv2")


def main():
    bench_config(2, 64, 64, 64)
    bench_config(2, 128, 128, 128)
    bench_config(2, 256, 256, 64)
    bench_config(4, 96, 96, 192, group=4)
    bench_config(2, 128, 128, 256, group=4)
    bench_config(1, 256, 256, 256, group=4)


if __name__ == "__main__":
    main()
