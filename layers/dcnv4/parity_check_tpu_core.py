import os
import sys

rootpath = os.path.abspath(os.path.join(os.path.dirname(__file__), os.pardir, os.pardir, os.pardir))
sys.path.insert(1, rootpath)

import numpy as np
import tensorflow as tf

from iseg.layers.dcnv4.dcnv4 import dcnv4_core
from iseg.layers.dcnv4.dcnv4_tpu import dcnv4_core_tpu, _soft_clip

tf.random.set_seed(7)

B, H, W, C, G = 2, 24, 20, 32, 4

value = tf.random.normal([B, H, W, C], dtype=tf.float64)


def run_pair(offset_mask, window_radius=2, **kw):
    ref = dcnv4_core(value, offset_mask, group=G, **kw)
    new = dcnv4_core_tpu(value, offset_mask, group=G, window_radius=window_radius, **kw)
    diff = np.max(np.abs(ref.numpy() - new.numpy()))
    return diff


# 1) offsets well inside the window -> must match
om_small = tf.random.normal([B, H, W, G * 9 * 3], stddev=0.3, dtype=tf.float64)
print("small offsets, R=2: max_abs_diff =", run_pair(om_small))

# 2) offsets at the window edge -> must match
om_edge = tf.random.uniform([B, H, W, G * 9 * 3], minval=-1.9, maxval=1.9, dtype=tf.float64)
print("edge offsets (|off|<1.9), R=2: max_abs_diff =", run_pair(om_edge))

# 3) large offsets -> expected to differ (soft-clamped)
om_large = tf.random.normal([B, H, W, G * 9 * 3], stddev=5.0, dtype=tf.float64)
print("large offsets (stddev 5), R=2: max_abs_diff =", run_pair(om_large), "(expected nonzero)")

# 4) large R covers the offsets entirely -> should match closely
om_cov = tf.random.uniform([B, H, W, G * 9 * 3], minval=-7.9, maxval=7.9, dtype=tf.float64)
print("offsets in +-7.9, R=8: max_abs_diff =", run_pair(om_cov, window_radius=8))

# 5) dilation / remove_center / offset_scale variants
om_v = tf.random.normal([B, H, W, G * 9 * 3], stddev=0.3, dtype=tf.float64)
print("dilation=2, R=2:", run_pair(om_v, dilation=(2, 2)))
om_rc = tf.random.normal([B, H, W, G * 8 * 3], stddev=0.3, dtype=tf.float64)
print("remove_center, R=2:", run_pair(om_rc, remove_center=True))
print("offset_scale=0.5, R=2:", run_pair(om_v, offset_scale=0.5))

# 6) gradient flow through soft clip for out-of-window offsets
om_big = tf.Variable(tf.random.normal([B, H, W, G * 9 * 3], stddev=6.0, dtype=tf.float32))
v32 = tf.cast(value, tf.float32)
with tf.GradientTape() as tape:
    y = dcnv4_core_tpu(v32, om_big, group=G, window_radius=2)
    loss = tf.reduce_mean(y)
g = tape.gradient(loss, om_big)
frac_nonzero = float(tf.reduce_mean(tf.cast(tf.abs(g) > 0, tf.float32)))
print(f"out-of-window grad coverage: {frac_nonzero:.4f} (hard clip would be ~0)")

# 7) gradient parity inside the window
om_in = tf.Variable(tf.random.normal([B, H, W, G * 9 * 3], stddev=0.3, dtype=tf.float64))
with tf.GradientTape() as tape:
    tape.watch(value)
    y = dcnv4_core(value, om_in, group=G)
    loss = tf.reduce_sum(y * y)
g_ref = tape.gradient(loss, [value, om_in])
with tf.GradientTape() as tape:
    tape.watch(value)
    y2 = dcnv4_core_tpu(value, om_in, group=G, window_radius=2)
    loss2 = tf.reduce_sum(y2 * y2)
g_new = tape.gradient(loss2, [value, om_in])
print("grad value diff:", float(tf.reduce_max(tf.abs(g_ref[0] - g_new[0]))))
print("grad offset_mask diff:", float(tf.reduce_max(tf.abs(g_ref[1] - g_new[1]))))

print("CPU parity checks done")
