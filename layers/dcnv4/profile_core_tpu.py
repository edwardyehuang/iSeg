# ===================================================================
# MIT License
# Copyright (c) 2024 edwardyehuang (https://github.com/edwardyehuang)
# ===================================================================

# Stage-by-stage profiling of dcnv4_core_tpu on TPU (on-device inputs, no
# infeed bottleneck).
#
# Stages (cumulative unless noted):
#   a_offset   - offset decode + soft clip + rel + point_ok (small tensors)
#   b_weights  - + hat weights + k-collapse reduce
#   c_stack    - + pad + slices+stack window extraction (no aggregation)
#   d_full     - + final mul+reduce aggregation
#   e_agg_only - d_full but patches from a random tensor (no slices stack):
#                isolates mul+reduce from the stack construction
#
#   python -u iseg/layers/dcnv4/profile_core_tpu.py [--size 256,256,256]

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
T_HW = 2 * WINDOW_RADIUS + 4
NUM_TAPS = T_HW * T_HW


def set_radius(r):
    global WINDOW_RADIUS, T_HW, NUM_TAPS
    WINDOW_RADIUS = r
    T_HW = 2 * WINDOW_RADIUS + 4
    NUM_TAPS = T_HW * T_HW


def _loopacc_core(value, offset_mask):
    """Full dcnv4_core_tpu equivalent, but the aggregation accumulates over
    taps in a python loop (no patches tensor is materialized): lets XLA tile
    from the small value buffer instead of an 8.6GB stack."""

    bs, height, width, channels = get_tensor_shape(value)
    group = GROUP
    group_channels = channels // group
    num_points = 9

    hout, wout = height, width

    compute_dtype = tf.float32
    weight_dtype = value.dtype if value.dtype == tf.bfloat16 else compute_dtype

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

    point = tf.constant([-1.0, 0.0, 1.0], dtype=compute_dtype)
    point_dx = tf.reshape(tf.tile(point[:, None], [1, 3]), [-1])
    point_dy = tf.reshape(tf.tile(point[None, :], [3, 1]), [-1])

    rel_x = off_x + point_dx[None, None, None, None, :]
    rel_y = off_y + point_dy[None, None, None, None, :]

    q = tf.range(hout, dtype=compute_dtype)
    pos_x = q[None, None, :, None, None] + rel_x
    pos_y = q[None, :, None, None, None] + rel_y

    hw_f = tf.cast(height, compute_dtype)
    point_ok = (pos_x > -1.0) & (pos_x < hw_f) & (pos_y > -1.0) & (pos_y < hw_f)
    point_ok = tf.cast(point_ok, compute_dtype)

    tap = tf.constant(list(range(-1 - WINDOW_RADIUS, WINDOW_RADIUS + 3)), dtype=weight_dtype)

    rel_x_w = tf.cast(rel_x, weight_dtype)
    rel_y_w = tf.cast(rel_y, weight_dtype)

    hat_x = rel_x_w[..., None] - tap[None, None, None, None, None, :]
    hat_x = tf.maximum(1.0 - tf.abs(hat_x), 0.0)
    hat_y = rel_y_w[..., None] - tap[None, None, None, None, None, :]
    hat_y = tf.maximum(1.0 - tf.abs(hat_y), 0.0)

    weights = (
        hat_y[..., :, None]
        * hat_x[..., None, :]
        * tf.cast(mask * point_ok, weight_dtype)[..., None, None]
    )
    weights = tf.reduce_sum(weights, axis=4)  # [B, H, W, G, T_h, T_w]

    pad_top = WINDOW_RADIUS + 1
    pad_bottom = WINDOW_RADIUS + 2
    value_padded = tf.pad(
        value, [[0, 0], [pad_top, pad_bottom], [pad_top, pad_bottom], [0, 0]]
    )

    out = tf.zeros([bs, hout, wout, channels], dtype=weight_dtype)
    for ty in range(T_HW):
        for tx in range(T_HW):
            w_t = weights[..., ty, tx]  # [B, H, W, G]
            w_t = w_t[..., None]  # [B, H, W, G, 1]
            slice_t = value_padded[:, ty : ty + hout, tx : tx + wout, :]
            slice_t = tf.reshape(slice_t, [bs, hout, wout, group, group_channels])
            out = out + tf.reshape(w_t * slice_t, [bs, hout, wout, channels])

    return tf.reduce_mean(tf.cast(out, tf.float32))


def _separable_core(value, offset_mask):
    """Full dcnv4_core_tpu equivalent via the exactly-separable bilinear hat:
    hat(pos - tap) = hat_y(pos_y - ty) * hat_x(pos_x - tx).

    Correct factorization (weights attach to the OUTPUT position, value
    slides):
      GY[b,h,w,g,k,d] = sum_ty shy[b,h,w,g,k,ty] * Vpad[b,h+ty,w,g,d]
      out[b,h,w,g,d]  = sum_k sum_tx hax[b,h,w,g,k,tx] * GY[b,h,w+tx,g,k,d]
    The big 2-D patches tensor [B,H,W,C,T^2] never exists; the largest
    intermediate is GY [B,H,W,G,K,D] (= 9/64 of it for K=9, T=8).
    """

    bs, height, width, channels = get_tensor_shape(value)
    group = GROUP
    group_channels = channels // group
    num_points = 9

    hout, wout = height, width

    compute_dtype = tf.float32
    weight_dtype = value.dtype if value.dtype == tf.bfloat16 else compute_dtype

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

    point = tf.constant([-1.0, 0.0, 1.0], dtype=compute_dtype)
    point_dx = tf.reshape(tf.tile(point[:, None], [1, 3]), [-1])
    point_dy = tf.reshape(tf.tile(point[None, :], [3, 1]), [-1])

    rel_x = off_x + point_dx[None, None, None, None, :]
    rel_y = off_y + point_dy[None, None, None, None, :]

    q = tf.range(hout, dtype=compute_dtype)
    pos_x = q[None, None, :, None, None] + rel_x
    pos_y = q[None, :, None, None, None] + rel_y

    hw_f = tf.cast(height, compute_dtype)
    point_ok = (pos_x > -1.0) & (pos_x < hw_f) & (pos_y > -1.0) & (pos_y < hw_f)
    point_ok = tf.cast(point_ok, compute_dtype)

    tap = tf.constant(list(range(-1 - WINDOW_RADIUS, WINDOW_RADIUS + 3)), dtype=weight_dtype)

    rel_x_w = tf.cast(rel_x, weight_dtype)
    rel_y_w = tf.cast(rel_y, weight_dtype)

    hat_x = rel_x_w[..., None] - tap[None, None, None, None, None, :]
    hat_x = tf.maximum(1.0 - tf.abs(hat_x), 0.0)  # [B, H, W, G, K, T_w]
    hat_y = rel_y_w[..., None] - tap[None, None, None, None, None, :]
    hat_y = tf.maximum(1.0 - tf.abs(hat_y), 0.0)  # [B, H, W, G, K, T_h]

    s = tf.cast(mask * point_ok, weight_dtype)  # [B, H, W, G, K]
    shy = s[..., None] * hat_y  # [B, H, W, G, K, T_h]

    pad_top = WINDOW_RADIUS + 1
    pad_bottom = WINDOW_RADIUS + 2
    value_padded = tf.pad(
        value, [[0, 0], [pad_top, pad_bottom], [pad_top, pad_bottom], [0, 0]]
    )

    # pass 1 (H window): GY[b,h,w,g,k,d] = sum_ty shy[...,ty] * Vpad[b,h+ty,w,g,d]
    gy = tf.zeros([bs, hout, wout, group, num_points, group_channels], dtype=weight_dtype)
    for ty in range(T_HW):
        shy_t = shy[..., ty][..., None]  # [B, H, W, G, K, 1]
        slice_t = value_padded[:, ty : ty + hout, pad_top : pad_top + wout, :]
        slice_t = tf.reshape(slice_t, [bs, hout, wout, group, 1, group_channels])
        gy = gy + shy_t * slice_t

    # pad GY along W for the tx window
    gy_padded = tf.pad(
        gy, [[0, 0], [0, 0], [pad_top, pad_bottom], [0, 0], [0, 0], [0, 0]]
    )

    # pass 2 (W window): out[b,h,w,g,d] = sum_k sum_tx hax[...,tx] * GY[b,h,w+tx,g,k,d]
    out = tf.zeros([bs, hout, wout, group, group_channels], dtype=weight_dtype)
    for tx in range(T_HW):
        hx_t = hat_x[..., tx][..., None]  # [B, H, W, G, K, 1]
        slice_t = gy_padded[:, :, tx : tx + wout, :, :, :]  # [B, H, W, G, K, D]
        out = out + tf.reduce_sum(hx_t * slice_t, axis=4)  # over K

    out = tf.reshape(out, [bs, hout, wout, channels])
    return tf.reduce_mean(tf.cast(out, tf.float32))


def stage_fn(value, offset_mask, stage, patches_src=None):

    if stage == "v1_full":
        from iseg.layers.dcnv4.dcnv4_tpu import dcnv4_core_tpu

        y = dcnv4_core_tpu(value, offset_mask, group=GROUP, window_radius=WINDOW_RADIUS)
        return tf.reduce_mean(tf.cast(y, tf.float32))

    if stage == "v2_loopacc":
        return _loopacc_core(value, offset_mask)

    if stage == "v3_separable":
        return _separable_core(value, offset_mask)

    bs, height, width, channels = get_tensor_shape(value)
    group = GROUP
    group_channels = channels // group
    num_points = 9

    hout, wout = height, width

    compute_dtype = tf.float32
    weight_dtype = value.dtype if value.dtype == tf.bfloat16 else compute_dtype

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

    tap = tf.constant(list(range(-1 - WINDOW_RADIUS, WINDOW_RADIUS + 3)), dtype=weight_dtype)

    rel_x_w = tf.cast(rel_x, weight_dtype)
    rel_y_w = tf.cast(rel_y, weight_dtype)

    hat_x = rel_x_w[..., None] - tap[None, None, None, None, None, :]
    hat_x = tf.maximum(1.0 - tf.abs(hat_x), 0.0)
    hat_y = rel_y_w[..., None] - tap[None, None, None, None, None, :]
    hat_y = tf.maximum(1.0 - tf.abs(hat_y), 0.0)

    weights = (
        hat_y[..., :, None]
        * hat_x[..., None, :]
        * tf.cast(mask * point_ok, weight_dtype)[..., None, None]
    )
    weights = tf.reduce_sum(weights, axis=4)  # [B, H, W, G, T_h, T_w]
    weights = tf.reshape(weights, [bs, hout, wout, group, NUM_TAPS])

    if stage == "b_weights":
        return tf.reduce_mean(tf.cast(weights, tf.float32))

    if stage == "e_agg_only":
        patches = tf.cast(patches_src, weight_dtype)
    else:
        pad_top = WINDOW_RADIUS + 1
        pad_bottom = WINDOW_RADIUS + 2
        value_padded = tf.pad(
            value, [[0, 0], [pad_top, pad_bottom], [pad_top, pad_bottom], [0, 0]]
        )

        slices = []
        for ty in range(T_HW):
            for tx in range(T_HW):
                slices.append(value_padded[:, ty : ty + hout, tx : tx + wout, :])

        patches = tf.stack(slices, axis=3)  # [B, H, W, T2, C]

    patches = tf.reshape(patches, [bs, hout, wout, NUM_TAPS, group, group_channels])

    if stage == "c_stack":
        return tf.reduce_mean(tf.cast(patches, tf.float32))

    weights_bc = tf.transpose(weights, [0, 1, 2, 4, 3])
    weights_bc = tf.cast(weights_bc[..., None], patches.dtype)

    output = tf.reduce_sum(patches * weights_bc, axis=3)

    return tf.reduce_mean(tf.cast(output, tf.float32))


def make_run_steps(strategy, stage, fwd_bwd, value_rep, offset_rep, patches_rep):

    def step_fn(value, offset_mask):

        if fwd_bwd:
            with tf.GradientTape() as tape:
                tape.watch([value, offset_mask])
                y = stage_fn(value, offset_mask, stage, patches_rep)
            grads = tape.gradient(y, [value, offset_mask])
            x_new = value + 1e-30 * tf.cast(y, value.dtype)
            return y, grads, x_new

        y = stage_fn(value, offset_mask, stage, patches_rep)
        x_new = value + 1e-30 * tf.cast(y, value.dtype)
        return y, x_new

    @tf.function(input_signature=[tf.TensorSpec([], tf.int32)])
    def run_steps(n):
        acc = tf.zeros([], tf.float32)
        inputs = strategy.experimental_distribute_values_from_function(
            lambda ctx: (tf.identity(value_rep), tf.identity(offset_rep))
        )

        for _ in tf.range(n):
            value, offset_mask = inputs
            out = strategy.run(step_fn, args=(value, offset_mask))

            if fwd_bwd:
                loss, grads, x_new = out
                acc += strategy.experimental_local_results(loss)[0]
                grads_r0 = strategy.experimental_local_results(grads)[0]
                grads_r0 = [g for g in grads_r0 if g is not None]
                if grads_r0:
                    acc += tf.add_n(
                        [tf.reduce_sum(tf.cast(g, tf.float32)) for g in grads_r0]
                    ) * 1e-30
            else:
                loss, x_new = out
                acc += strategy.experimental_local_results(loss)[0]

            inputs = (x_new, offset_mask)

        return acc

    return run_steps


def bench_stage(strategy, stage, B, H, W, C, fwd_bwd, warmup=10, iters=50):

    per_replica = B // strategy.num_replicas_in_sync

    with strategy.scope():
        value_rep = tf.random.normal([per_replica, H, W, C])
        offset_rep = tf.random.normal([per_replica, H, W, GROUP * 9 * 3], stddev=0.5)
        patches_rep = tf.random.normal([per_replica, H, W, NUM_TAPS * C]) if stage == "e_agg_only" else None

        run_steps = make_run_steps(strategy, stage, fwd_bwd, value_rep, offset_rep, patches_rep)

        run_steps(tf.constant(1, tf.int32)).numpy()
        run_steps(tf.constant(warmup, tf.int32)).numpy()
        t0 = time.perf_counter()
        run_steps(tf.constant(iters, tf.int32)).numpy()
        dt = (time.perf_counter() - t0) / iters

    return dt * 1000.0


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--tpu_name", default=os.environ.get("TPU_NAME", "node4"))
    parser.add_argument("--stages", default="a_offset,b_weights,c_stack,d_full,e_agg_only")
    parser.add_argument("--size", default="128,128,128")
    parser.add_argument("--radius", type=int, default=2)
    parser.add_argument("--batch_per_replica", type=int, default=1)
    args = parser.parse_args()

    strategy = get_tpu_strategy(args.tpu_name)
    print("num replicas:", strategy.num_replicas_in_sync, flush=True)

    set_radius(args.radius)
    B = strategy.num_replicas_in_sync * args.batch_per_replica
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
