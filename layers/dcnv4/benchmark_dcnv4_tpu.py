# ===================================================================
# MIT License
# Copyright (c) 2024 edwardyehuang (https://github.com/edwardyehuang)
# ===================================================================

# TPU benchmark: iseg.layers.dcnv4.DCNv4 vs iseg.layers.dcn_v2.DCNv2.
#
# Runs through TPUStrategy (server-side XLA via tpu.replicate, same as
# production training via main.py). Do NOT use client-side jit_compile here:
# it triggers "no registered transfer manager for platform Host" on the
# tpu-runtime server and eventually crashes it.
#
#   python -u iseg/layers/dcnv4/benchmark_dcnv4_tpu.py [--tpu_name=node4] [--fp32] [--full]
#
# After running, restart the TPU runtime before the next run:
#   sudo systemctl restart tpu-runtime.service

import argparse
import os
import sys
import time

rootpath = os.path.abspath(os.path.join(os.path.dirname(__file__), os.pardir, os.pardir, os.pardir))
sys.path.insert(1, rootpath)

import tensorflow as tf

from iseg.distribution.distribution_utils import get_tpu_strategy
from iseg.layers.dcnv4 import DCNv4
from iseg.layers.dcnv4.dcnv4_tpu import DCNv4_TPU
from iseg.layers.dcn_v2 import DCNv2


def build_layers(C, group):
    return {
        "dcnv2": DCNv2(filters=C, kernel_size=(3, 3)),
        "dcnv4": DCNv4(kernel_size=3, group=group),
        "dcnv4_tpu": DCNv4_TPU(kernel_size=3, group=group, window_radius=2),
        "dcnv4_tpu_r1": DCNv4_TPU(kernel_size=3, group=group, window_radius=1),
    }


def make_run_steps(strategy, layer, fwd_bwd, x_rep):
    """Benchmark loop that keeps the input ON DEVICE between iterations.

    Feeding the input through tf.data each step makes the benchmark
    infeed-bound (measured: 134MB/step -> ~140ms/step at 256x256x256, which
    swamps all layer compute). Instead we capture a per-replica input once
    and make it loop-carried: x <- x + 1e-30 * y. The tiny perturbation
    keeps XLA from hoisting the layer out of the while loop (loop-invariant
    code motion) without changing the measured math.
    """

    if fwd_bwd:

        def step_fn(x):
            with tf.GradientTape() as tape:
                y = layer(x, training=True)
                loss = tf.reduce_mean(tf.square(tf.cast(y, tf.float32)))
            grads = tape.gradient(loss, layer.trainable_variables)
            x_new = x + 1e-30 * tf.cast(y, x.dtype)
            return loss, grads, x_new

    else:

        def step_fn(x):
            y = layer(x, training=True)
            x_new = x + 1e-30 * tf.cast(y, x.dtype)
            return tf.reduce_mean(tf.cast(y, tf.float32)), x_new

    @tf.function(input_signature=[tf.TensorSpec([], tf.int32)])
    def run_steps(n):
        acc = tf.zeros([], tf.float32)
        x = strategy.experimental_distribute_values_from_function(
            lambda ctx: tf.identity(x_rep)
        )

        for _ in tf.range(n):
            out = strategy.run(step_fn, args=(x,))

            if fwd_bwd:
                loss, grads, x = out
                acc += strategy.experimental_local_results(loss)[0]
                grads_r0 = strategy.experimental_local_results(grads)[0]
                acc += tf.add_n(
                    [tf.reduce_sum(tf.cast(g, tf.float32)) for g in grads_r0]
                ) * 1e-30
            else:
                loss, x = out
                acc += strategy.experimental_local_results(loss)[0]

        return acc

    return run_steps


def bench_layer(strategy, layer, global_batch, H, W, C, fwd_bwd, warmup=10, iters=50):
    """Returns per-iteration time in ms, measured with `iters` sequential
    strategy.run steps inside one tf.function (no host dispatch, no infeed)."""

    per_replica = global_batch // strategy.num_replicas_in_sync

    with strategy.scope():
        x_rep = tf.random.normal([per_replica, H, W, C])
        run_steps = make_run_steps(strategy, layer, fwd_bwd, x_rep)

        # build + compile + realistic offsets (non-zero, see GPU benchmark)
        run_steps(tf.constant(1, tf.int32)).numpy()
        for v in layer.trainable_variables:
            if "offset" in v.name and "kernel" in v.name:
                v.assign(tf.random.normal(v.shape, stddev=0.05, dtype=tf.float32))

        run_steps(tf.constant(warmup, tf.int32)).numpy()
        t0 = time.perf_counter()
        run_steps(tf.constant(iters, tf.int32)).numpy()
        dt = (time.perf_counter() - t0) / iters

    return dt * 1000.0


def bench_config(strategy, B, H, W, C, group=4, layer_filter=None):
    print(f"\n== B={B} H={H} W={W} C={C} group={group} ==", flush=True)

    results = {}

    for name, layer in build_layers(C, group).items():
        if layer_filter and name not in layer_filter:
            continue
        t_fwd = bench_layer(strategy, layer, B, H, W, C, fwd_bwd=False)
        print(f"  {name:10s} fwd={t_fwd:8.3f} ms", flush=True)
        t_fwd_bwd = bench_layer(strategy, layer, B, H, W, C, fwd_bwd=True)
        print(f"  {name:10s} fwd+bwd={t_fwd_bwd:8.3f} ms", flush=True)
        results[name] = (t_fwd, t_fwd_bwd)

    if "dcnv2" in results:
        for mode, i in [("fwd", 0), ("fwd+bwd", 1)]:
            for name in results:
                if name == "dcnv2":
                    continue
                speedup = results["dcnv2"][i] / results[name][i]
                print(f"  >> {mode}: {name} is {speedup:.2f}x vs dcnv2", flush=True)


# FaPN-oriented coverage: C=256/512 across pyramid resolutions + batch scaling
FULL_CONFIGS = [
    # (B, H, W, C)
    (None, 128, 128, 128),
    (None, 256, 256, 256),  # P3
    (None, 128, 128, 256),  # P4
    (None, 64, 64, 256),  # P5
    (None, 32, 32, 256),  # P6
    (None, 128, 128, 512),  # C=512
    (None, 64, 64, 512),  # C=512
    ("x2", 128, 128, 256),  # 2 samples per replica
    ("x2", 64, 64, 512),
]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--tpu_name", default=os.environ.get("TPU_NAME", "node4"))
    parser.add_argument("--fp32", action="store_true", help="disable mixed_bfloat16")
    parser.add_argument("--full", action="store_true", help="run all configs (default: 2 quick ones)")
    parser.add_argument("--layers", default=None, help="comma list: dcnv2,dcnv4,dcnv4_tpu,dcnv4_tpu_r1")
    args = parser.parse_args()

    strategy = get_tpu_strategy(args.tpu_name)
    print("num replicas:", strategy.num_replicas_in_sync, flush=True)

    if not args.fp32:
        import keras

        keras.mixed_precision.set_global_policy("mixed_bfloat16")
        print("global policy: mixed_bfloat16", flush=True)

    R = strategy.num_replicas_in_sync
    layer_filter = args.layers.split(",") if args.layers else None

    if args.full:
        for bmul, H, W, C in FULL_CONFIGS:
            B = R * (2 if bmul == "x2" else 1)
            bench_config(strategy, B, H, W, C, layer_filter=layer_filter)
    else:
        bench_config(strategy, R, 128, 128, 128, layer_filter=layer_filter)
        bench_config(strategy, R, 256, 256, 256, layer_filter=layer_filter)

    print("\nbenchmark done", flush=True)

    if args.full:
        bench_config(strategy, B, 64, 64, 64)
        bench_config(strategy, B, 256, 256, 64)
        bench_config(strategy, B, 96, 96, 192)
        bench_config(strategy, B, 128, 128, 256)

    print("\nbenchmark done", flush=True)


if __name__ == "__main__":
    main()
