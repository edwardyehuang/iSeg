# ===================================================================
# MIT License
# Copyright (c) 2024 edwardyehuang (https://github.com/edwardyehuang)
# ===================================================================

# Stage-by-stage profiling of dcnv4_core_tpu on TPU.
#
# Stages (cumulative):
#   a_offset   - offset decode + soft clip + rel + point_ok (small tensors)
#   b_weights  - + hat weights + k-collapse reduce
#   c_patches  - + pad + window extraction + reshape (no final aggregation)
#   d_full     - + final mul+reduce aggregation
#   c_slices   - c_patches but window extraction via T^2 slices + stack
#   d_slices   - d_full but window extraction via T^2 slices + stack
#
#   python -u iseg/layers/dcnv4/profile_core_tpu.py [--tpu_name=node4]

import argparse
import os
import sys
import time

rootpath = os.path.abspath(os.path.join(os.path.dirname(__file__), os.pardir, os.pardir, os.pardir))
sys.path.insert(1, rootpath)

import tensorflow as tf

from iseg.distribution.distribution_utils import get_tpu_strategy
from iseg.utils import get_tensor_shape


GROUP = 4
WINDOW_RADIUS = 2


def stage_fn(value, offset_mask, stage):

    bs, height, width, channels = get_tensor_shape(value)
    group = GROUP
    group_channels = channels // group
    kernel_size = (3, 3)
    num_points = 9

    hout, wout = height, width

    compute_dtype = tf.float32

    offset_mask = offset_mask[..., : group * num_points * 3]
    offset_mask = tf.cast(offset_mask, compute_dtype)
    offset_mask = tf.reshape(offset_mask, [bs, hout, wout, group, 3 * num_points])

    offsets = tf.reshape(offset_mask[..., : 2 * num_points], [bs, hout, wout, group, num_points, 2])
    mask = offset_mask[..., 2 * num_points :]

    off_x = offsets[..., 0]
    off_y = offsets[..., 1]

    R = float(WINDOW_RADIUS)
    over_x = tf.abs(off_x) - R
    off_x = tf.where(over_x <= 0.0, off_x, tf.sign(off_x) * (R + tf.tanh(over_x)))
    over_y = tf.abs(off_y) - R
    off_y = tf.where(over_y <= 0.0, off_y, tf.sign(off_y) * (R + tf.tanh(over_y)))

    center_off = 1
    point = tf.constant([-1.0, 0.0, 1.0], dtype=compute_dtype)
    point_dx = tf.reshape(tf.tile(point[:, None], [1, 3]), [-1])  # width outer
    point_dy = tf.reshape(tf.tile(point[None, :], [3, 1]), [-1])

    rel_x = off_x + point_dx[None, None, None, None, :]
    rel_y = off_y + point_dy[None, None, None, None, :]

    q = tf.range(hout, dtype=compute_dtype)
    pos_x = q[None, None, :, None, None] + rel_x
    pos_y = q[None, :, None, None, None] + rel_y

    hw_f = tf.cast(height, compute_dtype)
    point_ok = (pos_x > -1.0) & (pos_x < hw_f) & (pos_y > -1.0) & (pos_y < hw_f)
    point_ok = tf.cast(point_ok, compute_dtype)

    if stage == "a_offset":
        return tf.reduce_mean(point_ok) + tf.reduce_mean(rel_x) * 1e-30

    tap_x = tf.range(-1 - WINDOW_RADIUS, 1 + WINDOW_RADIUS + 2, dtype=compute_dtype)  # [T]
    t_h = t_w = 2 * WINDOW_RADIUS + 4
    num_taps = t_h * t_w

    hat_x = rel_x[..., None] - tap_x[None, None, None, None, None, :]
    hat_x = tf.maximum(1.0 - tf.abs(hat_x), 0.0)
    hat_y = rel_y[..., None] - tap_x[None, None, None, None, None, :]
    hat_y = tf.maximum(1.0 - tf.abs(hat_y), 0.0)

    weights = hat_y[..., :, None] * hat_x[..., None, :] * (mask * point_ok)[..., None, None]
    weights = tf.reduce_sum(weights, axis=4)  # [B, H, W, G, T_h, T_w]
    weights = tf.reshape(weights, [bs, hout, wout, group, num_taps])

    if stage == "b_weights":
        return tf.reduce_mean(weights)

    use_slices = stage.endswith("slices")
    want_full = stage.startswith("d")

    # window extraction; taps span [-1-R, R+2] -> asymmetric pad
    pad_top = WINDOW_RADIUS + 1
    pad_bottom = WINDOW_RADIUS + 2
    value_padded = tf.pad(
        value, [[0, 0], [pad_top, pad_bottom], [pad_top, pad_bottom], [0, 0]]
    )

    if use_slices:
        slices = []
        for ty in range(t_h):
            for tx in range(t_w):
                slices.append(
                    value_padded[:, ty : ty + hout, tx : tx + wout, :]
                )
        patches = tf.stack(slices, axis=3)  # [B, H, W, T2, C]
    else:
        patches = tf.image.extract_patches(
            value_padded,
            sizes=[1, t_h, t_w, 1],
            strides=[1, 1, 1, 1],
            rates=[1, 1, 1, 1],
            padding="VALID",
        )

    patches = tf.reshape(patches, [bs, hout, wout, num_taps, group, group_channels])

    if not want_full:
        return tf.reduce_mean(tf.cast(patches, tf.float32))

    weights_bc = tf.transpose(weights, [0, 1, 2, 4, 3])
    weights_bc = tf.cast(weights_bc[..., None], patches.dtype)

    output = tf.reduce_sum(patches * weights_bc, axis=3)

    return tf.reduce_mean(tf.cast(output, tf.float32))


def make_run_steps(strategy, it, stage, fwd_bwd):

    def step_fn(inputs):
        value, offset_mask = inputs

        if fwd_bwd:
            with tf.GradientTape() as tape:
                tape.watch([value, offset_mask])
                y = stage_fn(value, offset_mask, stage)
            grads = tape.gradient(y, [value, offset_mask])
            return y, grads

        return stage_fn(value, offset_mask, stage)

    @tf.function(input_signature=[tf.TensorSpec([], tf.int32)])
    def run_steps(n):
        acc = tf.zeros([], tf.float32)

        for _ in tf.range(n):
            inputs = next(it)
            out = strategy.run(step_fn, args=(inputs,))

            if fwd_bwd:
                loss, grads = out
                acc += strategy.reduce(tf.distribute.ReduceOp.SUM, loss, axis=None)
                grads_r0 = strategy.experimental_local_results(grads)[0]
                grads_r0 = [g for g in grads_r0 if g is not None]
                if grads_r0:
                    acc += tf.add_n(
                        [tf.reduce_sum(tf.cast(g, tf.float32)) for g in grads_r0]
                    ) * 1e-30
            else:
                acc += strategy.reduce(tf.distribute.ReduceOp.SUM, out, axis=None)

        return acc

    return run_steps


def bench_stage(strategy, stage, B, H, W, C, fwd_bwd, warmup=10, iters=50):

    with strategy.scope():
        value = tf.random.normal([B, H, W, C])
        offset_mask = tf.random.normal([B, H, W, GROUP * 9 * 3], stddev=0.5)
        ds = tf.data.Dataset.from_tensors((value, offset_mask)).repeat()
        it = iter(strategy.experimental_distribute_dataset(ds))

        run_steps = make_run_steps(strategy, it, stage, fwd_bwd)

        run_steps(tf.constant(1, tf.int32)).numpy()
        run_steps(tf.constant(warmup, tf.int32)).numpy()
        t0 = time.perf_counter()
        run_steps(tf.constant(iters, tf.int32)).numpy()
        dt = (time.perf_counter() - t0) / iters

    return dt * 1000.0


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--tpu_name", default=os.environ.get("TPU_NAME", "node4"))
    parser.add_argument("--stages", default="a_offset,b_weights,c_patches,d_full,c_slices,d_slices")
    parser.add_argument("--size", default="128,128,128")
    args = parser.parse_args()

    strategy = get_tpu_strategy(args.tpu_name)
    print("num replicas:", strategy.num_replicas_in_sync, flush=True)

    B = strategy.num_replicas_in_sync
    H, W, C = [int(v) for v in args.size.split(",")]

    print(f"config: B={B} H={H} W={W} C={C}", flush=True)

    for stage in args.stages.split(","):
        t_fwd = bench_stage(strategy, stage, B, H, W, C, fwd_bwd=False)
        print(f"  {stage:10s} fwd={t_fwd:8.3f} ms", flush=True)
        t_fwd_bwd = bench_stage(strategy, stage, B, H, W, C, fwd_bwd=True)
        print(f"  {stage:10s} fwd+bwd={t_fwd_bwd:8.3f} ms", flush=True)

    print("done", flush=True)


if __name__ == "__main__":
    main()
