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
    }


def make_step_fn(layer, fwd_bwd):

    if fwd_bwd:

        def step_fn(x):
            with tf.GradientTape() as tape:
                y = layer(x, training=True)
                loss = tf.reduce_mean(tf.square(tf.cast(y, tf.float32)))
            grads = tape.gradient(loss, layer.trainable_variables)
            return loss, grads

    else:

        def step_fn(x):
            y = layer(x, training=True)
            return tf.reduce_mean(tf.cast(y, tf.float32))

    return step_fn


def make_run_steps(strategy, it, layer, fwd_bwd):

    step_fn = make_step_fn(layer, fwd_bwd)

    # input_signature fixes the trace count to 1 per (layer, mode); passing
    # different tf.constant values for n would otherwise retrace per value
    @tf.function(input_signature=[tf.TensorSpec([], tf.int32)])
    def run_steps(n):
        acc = tf.zeros([], tf.float32)

        for _ in tf.range(n):
            x = next(it)
            out = strategy.run(step_fn, args=(x,))

            if fwd_bwd:
                loss, grads = out
                acc += strategy.reduce(tf.distribute.ReduceOp.SUM, loss, axis=None)
                # force grads to be computed & materialized (no *0.0: XLA
                # would simplify it away and DCE the whole backward)
                grads_r0 = strategy.experimental_local_results(grads)[0]
                acc += tf.add_n(
                    [tf.reduce_sum(tf.cast(g, tf.float32)) for g in grads_r0]
                ) * 1e-30
            else:
                acc += strategy.reduce(tf.distribute.ReduceOp.SUM, out, axis=None)

        return acc

    return run_steps


def bench_layer(strategy, layer, global_batch, H, W, C, fwd_bwd, warmup=10, iters=50):
    """Returns per-iteration time in ms, measured with `iters` sequential
    strategy.run steps inside one tf.function (no host dispatch in the loop)."""

    with strategy.scope():
        x = tf.random.normal([global_batch, H, W, C])
        ds = tf.data.Dataset.from_tensors(x).repeat()
        it = iter(strategy.experimental_distribute_dataset(ds))

        run_steps = make_run_steps(strategy, it, layer, fwd_bwd)

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


def bench_config(strategy, B, H, W, C, group=4):
    print(f"\n== B={B} H={H} W={W} C={C} group={group} ==", flush=True)

    results = {}

    for name, layer in build_layers(C, group).items():
        t_fwd = bench_layer(strategy, layer, B, H, W, C, fwd_bwd=False)
        print(f"  {name:10s} fwd={t_fwd:8.3f} ms", flush=True)
        t_fwd_bwd = bench_layer(strategy, layer, B, H, W, C, fwd_bwd=True)
        print(f"  {name:10s} fwd+bwd={t_fwd_bwd:8.3f} ms", flush=True)
        results[name] = (t_fwd, t_fwd_bwd)

    for mode, i in [("fwd", 0), ("fwd+bwd", 1)]:
        for name in results:
            if name == "dcnv2":
                continue
            speedup = results["dcnv2"][i] / results[name][i]
            print(f"  >> {mode}: {name} is {speedup:.2f}x vs dcnv2", flush=True)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--tpu_name", default=os.environ.get("TPU_NAME", "node4"))
    parser.add_argument("--fp32", action="store_true", help="disable mixed_bfloat16")
    parser.add_argument("--full", action="store_true", help="run all 6 configs (default: 2 quick ones)")
    args = parser.parse_args()

    strategy = get_tpu_strategy(args.tpu_name)
    print("num replicas:", strategy.num_replicas_in_sync, flush=True)

    if not args.fp32:
        import keras

        keras.mixed_precision.set_global_policy("mixed_bfloat16")
        print("global policy: mixed_bfloat16", flush=True)

    B = strategy.num_replicas_in_sync  # 1 sample per replica

    bench_config(strategy, B, 128, 128, 128)
    bench_config(strategy, B, 256, 256, 256)

    if args.full:
        bench_config(strategy, B, 64, 64, 64)
        bench_config(strategy, B, 256, 256, 64)
        bench_config(strategy, B, 96, 96, 192)
        bench_config(strategy, B, 128, 128, 256)

    print("\nbenchmark done", flush=True)


if __name__ == "__main__":
    main()
