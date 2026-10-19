# ===================================================================
# MIT License
# Copyright (c) 2024 edwardyehuang (https://github.com/edwardyehuang)
# ===================================================================

# Ablation profiling of dcnv4_core on TPU: where does the time go?
#
# Variants (all receive the same realistic value/offset inputs):
#   full            - dcnv4_core as-is
#   const_idx       - gathers use constant zero indices (index math removed)
#   static_slice    - gathers replaced by static slices (gather removed)
#   fused_gather    - one [K*4] gather instead of 4x [K] gathers
#   gather_full_c   - gather full C-wide rows (wide rows, G x bytes)
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
from iseg.layers.dcnv4.dcnv4 import dcnv4_core
from iseg.utils import get_tensor_shape


def _grid(offset_mask, group, num_points, kernel_size, stride, pad, dilation, offset_scale, height, width):
    """Shared index/weight math. Returns (indices[4], weights[4]) each
    [B, Hout, Wout, G, K], plus shapes."""

    kh, kw = kernel_size
    sh, sw = stride
    ph, pw = pad
    dh, dw = dilation

    center_h, center_w = kh // 2, kw // 2

    points = [(i, j) for i in range(kw) for j in range(kh)]
    assert len(points) == num_points

    bs, hout, wout, _ = get_tensor_shape(offset_mask)

    compute_dtype = tf.float32

    offset_mask = offset_mask[..., : group * num_points * 3]
    offset_mask = tf.reshape(offset_mask, [bs, hout, wout, group, 3 * num_points])

    offsets = tf.reshape(offset_mask[..., : 2 * num_points], [bs, hout, wout, group, num_points, 2])
    mask = offset_mask[..., 2 * num_points :]

    off_x = offsets[..., 0] * offset_scale
    off_y = offsets[..., 1] * offset_scale

    center_off_h = (dh * (kh - 1)) // 2
    center_off_w = (dw * (kw - 1)) // 2

    point_dx = tf.constant([(i * dw - center_off_w) * offset_scale for (i, _) in points], dtype=compute_dtype)
    point_dy = tf.constant([(j * dh - center_off_h) * offset_scale for (_, j) in points], dtype=compute_dtype)

    qx = tf.range(wout, dtype=compute_dtype)
    qy = tf.range(hout, dtype=compute_dtype)

    base_x = (center_off_w - pw) + qx * sw
    base_y = (center_off_h - ph) + qy * sh

    base_x = base_x[:, None] + point_dx[None, :]
    base_y = base_y[:, None] + point_dy[None, :]

    x_pos = base_x[None, None, :, None, :] + off_x
    y_pos = base_y[None, :, None, None, :] + off_y

    ix0 = tf.floor(x_pos)
    iy0 = tf.floor(y_pos)
    ix1 = ix0 + 1.0
    iy1 = iy0 + 1.0

    lx = x_pos - ix0
    ly = y_pos - iy0
    hx = 1.0 - lx
    hy = 1.0 - ly

    height_f = tf.cast(height, compute_dtype)
    width_f = tf.cast(width, compute_dtype)

    point_ok = (x_pos > -1.0) & (x_pos < width_f) & (y_pos > -1.0) & (y_pos < height_f)
    point_ok = tf.cast(point_ok, compute_dtype)

    c1_ok = tf.cast((ix0 >= 0.0) & (iy0 >= 0.0), compute_dtype)
    c2_ok = tf.cast((ix1 < width_f) & (iy0 >= 0.0), compute_dtype)
    c3_ok = tf.cast((ix0 >= 0.0) & (iy1 < height_f), compute_dtype)
    c4_ok = tf.cast((ix1 < width_f) & (iy1 < height_f), compute_dtype)

    base_w = mask * point_ok
    weights = [
        base_w * (hy * hx * c1_ok),
        base_w * (hy * lx * c2_ok),
        base_w * (ly * hx * c3_ok),
        base_w * (ly * lx * c4_ok),
    ]

    width_i32 = tf.cast(width, tf.int32)

    def _flat_index(ix, iy):
        ix = tf.cast(tf.clip_by_value(ix, 0.0, width_f - 1.0), tf.int32)
        iy = tf.cast(tf.clip_by_value(iy, 0.0, height_f - 1.0), tf.int32)
        return iy * width_i32 + ix

    indices = [
        _flat_index(ix0, iy0),
        _flat_index(ix1, iy0),
        _flat_index(ix0, iy1),
        _flat_index(ix1, iy1),
    ]

    return indices, weights


def core_variant(value, offset_mask, variant, group=4, kernel_size=(3, 3), stride=(1, 1), pad=(1, 1), dilation=(1, 1), offset_scale=1.0):

    bs, height, width, channels = get_tensor_shape(value)
    group_channels = channels // group
    num_points = kernel_size[0] * kernel_size[1]

    hout = get_tensor_shape(offset_mask)[1]
    wout = get_tensor_shape(offset_mask)[2]

    indices, weights = _grid(
        offset_mask, group, num_points, kernel_size, stride, pad, dilation, offset_scale, height, width
    )

    group_index = tf.range(group, dtype=tf.int32)[None, None, None, :, None]
    batch_index = (
        tf.range(bs, dtype=tf.int32)[:, None, None, None, None]
        * tf.cast(height * width * group, tf.int32)
    )

    value_flat = tf.reshape(value, [bs * height * width * group, group_channels])

    output = tf.zeros([bs, hout, wout, group, group_channels], dtype=value.dtype)

    if variant == "static_slice":
        # static slices instead of gathers: value[:, :hout, :wout] viewed per group
        v = tf.reshape(value[:, :hout, :wout, :], [bs, hout, wout, group, 1, group_channels])
        v = tf.broadcast_to(v, [bs, hout, wout, group, num_points, group_channels])
        for c in range(4):
            w_c = tf.cast(weights[c], v.dtype)
            output = output + tf.reduce_sum(v * w_c[..., None], axis=-2)
        return tf.reshape(output, [bs, hout, wout, channels])

    if variant == "const_idx":
        indices = [tf.zeros_like(i) for i in indices]

    if variant == "fused_gather":
        idx = tf.stack(indices, axis=-1)  # [B, Hout, Wout, G, K, 4]
        w = tf.stack(weights, axis=-1)  # [B, Hout, Wout, G, K, 4]
        idx = idx * group + group_index[..., None] + batch_index[..., None]
        idx = tf.reshape(idx, [bs, hout, wout, group, num_points * 4])
        gathered = tf.gather(value_flat, idx)  # [B, Hout, Wout, G, K*4, D]
        w = tf.reshape(w, [bs, hout, wout, group, num_points * 4])
        output = tf.reduce_sum(gathered * tf.cast(w, gathered.dtype)[..., None], axis=-2)
        return tf.reshape(output, [bs, hout, wout, channels])

    if variant == "gather_full_c":
        idx = tf.stack(indices, axis=-1)  # [B, Hout, Wout, G, K, 4]
        w = tf.stack(weights, axis=-1)  # [B, Hout, Wout, G, K, 4]
        idx = idx + batch_index[..., None] // group * tf.cast(width, tf.int32) * 0  # keep shape; batch offset below
        idx = idx + (
            tf.range(bs, dtype=tf.int32)[:, None, None, None, None, None]
            * tf.cast(height * width, tf.int32)
        )
        value_flat_c = tf.reshape(value, [bs * height * width, channels])
        gathered = tf.gather(value_flat_c, idx)  # [B, Hout, Wout, G, K, 4, C]
        outs = []
        for g in range(group):
            gg = gathered[:, :, :, g]  # [B, Hout, Wout, K, 4, C]
            gg = gg[..., g * group_channels : (g + 1) * group_channels]
            w_g = w[:, :, :, g]  # [B, Hout, Wout, K, 4]
            outs.append(
                tf.reduce_sum(gg * tf.cast(w_g, gg.dtype)[..., None], axis=(-4, -3))
            )
        output = tf.stack(outs, axis=3)  # [B, Hout, Wout, G, D]
        return tf.reshape(output, [bs, hout, wout, channels])

    # default: separate gathers (full / const_idx)
    for c in range(4):
        idx_c = indices[c] * group + group_index + batch_index
        gathered_c = tf.gather(value_flat, idx_c)
        w_c = tf.cast(weights[c], gathered_c.dtype)
        output = output + tf.reduce_sum(gathered_c * w_c[..., None], axis=-2)

    return tf.reshape(output, [bs, hout, wout, channels])


def make_step_fn(variant, value_shape, offset_shape, fwd_bwd):

    group = 4

    def compute(value, offset_mask):
        if variant == "full":
            return dcnv4_core(value, offset_mask, group=group)
        return core_variant(value, offset_mask, variant, group=group)

    if fwd_bwd:

        def step_fn(inputs):
            value, offset_mask = inputs
            with tf.GradientTape() as tape:
                tape.watch([value, offset_mask])
                y = compute(value, offset_mask)
                loss = tf.reduce_mean(tf.square(tf.cast(y, tf.float32)))
            grads = tape.gradient(loss, [value, offset_mask])
            return loss, grads

    else:

        def step_fn(inputs):
            value, offset_mask = inputs
            y = compute(value, offset_mask)
            return tf.reduce_mean(tf.cast(y, tf.float32))

    return step_fn


def make_run_steps(strategy, it, step_fn, fwd_bwd):

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
                acc += tf.add_n(
                    [tf.reduce_sum(tf.cast(g, tf.float32)) for g in grads_r0]
                ) * 1e-30
            else:
                acc += strategy.reduce(tf.distribute.ReduceOp.SUM, out, axis=None)

        return acc

    return run_steps


def bench_variant(strategy, variant, B, H, W, C, fwd_bwd, warmup=10, iters=50):

    with strategy.scope():
        value = tf.random.normal([B, H, W, C])
        offset_mask = tf.random.normal([B, H, W, 4 * 9 * 3], stddev=0.5)
        ds = tf.data.Dataset.from_tensors((value, offset_mask)).repeat()
        it = iter(strategy.experimental_distribute_dataset(ds))

        step_fn = make_step_fn(variant, None, None, fwd_bwd)
        run_steps = make_run_steps(strategy, it, step_fn, fwd_bwd)

        run_steps(tf.constant(1, tf.int32)).numpy()
        run_steps(tf.constant(warmup, tf.int32)).numpy()
        t0 = time.perf_counter()
        run_steps(tf.constant(iters, tf.int32)).numpy()
        dt = (time.perf_counter() - t0) / iters

    return dt * 1000.0


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--tpu_name", default=os.environ.get("TPU_NAME", "node4"))
    parser.add_argument("--variants", default="full,const_idx,static_slice,fused_gather,gather_full_c")
    args = parser.parse_args()

    strategy = get_tpu_strategy(args.tpu_name)
    print("num replicas:", strategy.num_replicas_in_sync, flush=True)

    B = strategy.num_replicas_in_sync
    H = W = 128
    C = 128

    variants = args.variants.split(",")

    print(f"config: B={B} H={H} W={W} C={C} (fp32 core inputs)", flush=True)

    for variant in variants:
        t_fwd = bench_variant(strategy, variant, B, H, W, C, fwd_bwd=False)
        print(f"  {variant:15s} fwd={t_fwd:8.3f} ms", flush=True)
        t_fwd_bwd = bench_variant(strategy, variant, B, H, W, C, fwd_bwd=True)
        print(f"  {variant:15s} fwd+bwd={t_fwd_bwd:8.3f} ms", flush=True)

    print("done", flush=True)


if __name__ == "__main__":
    main()
