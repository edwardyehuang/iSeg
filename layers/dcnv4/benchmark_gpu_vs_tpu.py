"""GPU-side speed comparison: DCNv2 / DCNv4 / DCNv4_TPU(R=2) / DCNv4_TPU(R=1) under XLA."""
import os, sys, time
rootpath = os.path.abspath(os.path.join(os.path.dirname(__file__), os.pardir, os.pardir, os.pardir))
sys.path.insert(1, rootpath)

import tensorflow as tf

from iseg.layers.dcnv2 import DCNv2
from iseg.layers.dcnv4 import DCNv4, DCNv4_TPU


def make_fwd(layer):
    @tf.function(jit_compile=True)
    def fn(x):
        return layer(x)
    return fn


def make_fwd_bwd(layer):
    @tf.function(jit_compile=True)
    def fn(x):
        with tf.GradientTape() as tape:
            y = layer(x)
            loss = tf.reduce_mean(tf.square(y))
        return tape.gradient(loss, layer.trainable_variables)
    return fn


def bench(fn, x, warmup=10, iters=50):
    for _ in range(warmup):
        r = fn(x)
    _sync(r)
    t0 = time.perf_counter()
    for _ in range(iters):
        r = fn(x)
    _sync(r)
    return (time.perf_counter() - t0) / iters * 1000.0


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
        "dcnv4(gpu)": DCNv4(kernel_size=3, group=group),
        "dcnv4_tpu_r2": DCNv4_TPU(kernel_size=3, group=group, window_radius=2),
        "dcnv4_tpu_r1": DCNv4_TPU(kernel_size=3, group=group, window_radius=1),
    }

    # realistic (non-zero) offsets for a representative gather pattern
    for layer in layers.values():
        layer(x)  # build
        for v in layer.trainable_variables:
            if "offset" in v.name and "kernel" in v.name:
                v.assign(tf.random.normal(v.shape, stddev=0.05))

    results = {}
    for name, layer in layers.items():
        t_fwd = bench(make_fwd(layer), x)
        t_bwd = bench(make_fwd_bwd(layer), x)
        results[name] = (t_fwd, t_bwd)
        print(f"  {name:14s} fwd={t_fwd:8.3f} ms   fwd+bwd={t_bwd:8.3f} ms")

    base = results["dcnv4(gpu)"]
    for name in ["dcnv4_tpu_r2", "dcnv4_tpu_r1"]:
        sf = base[0] / results[name][0]
        sb = base[1] / results[name][1]
        print(f"  >> {name} vs dcnv4(gpu): fwd {sf:.2f}x, fwd+bwd {sb:.2f}x")


def main():
    bench_config(2, 64, 64, 64)
    bench_config(2, 128, 128, 128)
    bench_config(2, 256, 256, 64)
    bench_config(4, 96, 96, 192)
    bench_config(2, 128, 128, 256)
    bench_config(1, 256, 256, 256)


if __name__ == "__main__":
    main()
