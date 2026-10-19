# ===================================================================
# MIT License
# Copyright (c) 2024 edwardyehuang (https://github.com/edwardyehuang)
# ===================================================================

# TPU benchmark: iseg.layers.dcnv2 DCNv2 vs DCNv2_Flash vs DCNv2_TPU.
#
# Same harness as benchmark_dcnv4_tpu.py (TPUStrategy, server-side XLA via
# tpu.replicate, on-device loop-carried inputs). No client-side jit_compile.
#
#   python -u iseg/layers/dcnv2/benchmark_dcnv2_tpu.py [--tpu_name=node4] [--full]
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
from iseg.layers.dcnv2 import DCNv2, DCNv2_Flash

try:
    from iseg.layers.dcnv2.dcnv2_tpu import DCNv2_TPU
except ImportError:
    DCNv2_TPU = None


def build_layers(C):
    layers = {
        "dcnv2": DCNv2(filters=C, kernel_size=(3, 3)),
        "dcnv2_flash": DCNv2_Flash(filters=C, kernel_size=(3, 3)),
    }
    if DCNv2_TPU is not None:
        layers["dcnv2_tpu"] = DCNv2_TPU(filters=C, kernel_size=(3, 3), window_radius=2)
        layers["dcnv2_tpu_r1"] = DCNv2_TPU(filters=C, kernel_size=(3, 3), window_radius=1)
    return layers


def make_run_steps(strategy, layer, fwd_bwd, x_rep):
    """See benchmark_dcnv4_tpu.py: on-device loop-carried input."""

    if fwd_bwd:

        def step_fn(x):
            with tf.GradientTape() as tape:
                y = layer(x, training=True)
                loss = tf.reduce_mean(tf.square(tf.cast(y, tf.float32)))
            grads = tape.gradient(loss, layer.trainable_variables)
            x_new = x + tf.cast(1e-30 * tf.reduce_mean(tf.cast(y, tf.float32)), x.dtype)
            return loss, grads, x_new

    else:

        def step_fn(x):
            y = layer(x, training=True)
            x_new = x + tf.cast(1e-30 * tf.reduce_mean(tf.cast(y, tf.float32)), x.dtype)
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
                    [tf.reduce_sum(tf.cast(g, tf.float32)) for g in grads_r0 if g is not None]
                ) * 1e-30
            else:
                loss, x = out
                acc += strategy.experimental_local_results(loss)[0]

        return acc

    return run_steps


def bench_layer(strategy, layer, global_batch, H, W, C, fwd_bwd, warmup=10, iters=50):

    per_replica = global_batch // strategy.num_replicas_in_sync

    with strategy.scope():
        x_rep = tf.random.normal([per_replica, H, W, C])
        run_steps = make_run_steps(strategy, layer, fwd_bwd, x_rep)

        # build + compile + realistic offsets (non-zero offset kernels)
        run_steps(tf.constant(1, tf.int32)).numpy()
        for v in layer.trainable_variables:
            if "offset" in v.name and "kernel" in v.name:
                v.assign(tf.random.normal(v.shape, stddev=0.05, dtype=tf.float32))

        run_steps(tf.constant(warmup, tf.int32)).numpy()
        t0 = time.perf_counter()
        run_steps(tf.constant(iters, tf.int32)).numpy()
        dt = (time.perf_counter() - t0) / iters

    return dt * 1000.0


def bench_config(strategy, B, H, W, C, layer_filter=None):
    print(f"\n== B={B} H={H} W={W} C={C} ==", flush=True)

    results = {}

    for name, layer in build_layers(C).items():
        if layer_filter and name not in layer_filter:
            continue
        t_fwd = bench_layer(strategy, layer, B, H, W, C, fwd_bwd=False)
        print(f"  {name:14s} fwd={t_fwd:8.3f} ms", flush=True)
        t_fwd_bwd = bench_layer(strategy, layer, B, H, W, C, fwd_bwd=True)
        print(f"  {name:14s} fwd+bwd={t_fwd_bwd:8.3f} ms", flush=True)
        results[name] = (t_fwd, t_fwd_bwd)

    base = "dcnv2"
    if base in results:
        for mode, i in [("fwd", 0), ("fwd+bwd", 1)]:
            for name in results:
                if name == base:
                    continue
                speedup = results[base][i] / results[name][i]
                print(f"  >> {mode}: {name} is {speedup:.2f}x vs {base}", flush=True)


FULL_CONFIGS = [
    # (B_mult, H, W, C)
    (None, 128, 128, 256),
    (None, 256, 256, 256),
    (None, 64, 64, 256),
    (None, 128, 128, 512),
    (None, 64, 64, 512),
    ("x2", 128, 128, 256),
]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--tpu_name", default=os.environ.get("TPU_NAME", "node4"))
    parser.add_argument("--fp32", action="store_true", help="disable mixed_bfloat16")
    parser.add_argument("--full", action="store_true")
    parser.add_argument("--layers", default=None)
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
        bench_config(strategy, R, 128, 128, 256, layer_filter=layer_filter)
        bench_config(strategy, R, 256, 256, 256, layer_filter=layer_filter)

    print("\nbenchmark done", flush=True)


if __name__ == "__main__":
    main()
