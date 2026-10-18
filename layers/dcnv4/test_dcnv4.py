# ===================================================================
# MIT License
# Copyright (c) 2024 edwardyehuang (https://github.com/edwardyehuang)
# ===================================================================

# TF-side parity check of iseg.layers.dcnv4 against the official DCNv4
# (PyTorch CUDA op) ground truth.
#
# Step 1 (dcnv4 conda env):
#   python iseg/layers/dcnv4/parity_check_torch.py --out_dir /tmp/dcnv4_parity
# Step 2 (tf215 conda env):
#   python iseg/layers/dcnv4/test_dcnv4.py --parity_dir /tmp/dcnv4_parity
#
# Without --parity_dir, only self-contained checks (shape/grad/train) run.

import argparse
import ast
import os
import sys

import numpy as np

rootpath = os.path.abspath(os.path.join(os.path.dirname(__file__), os.pardir, os.pardir, os.pardir))
sys.path.insert(1, rootpath)

import tensorflow as tf

from iseg.layers.dcnv4 import DCNv4


LAYER_KWARGS = [
    "kernel_size",
    "group",
    "offset_scale",
    "remove_center",
    "center_feature_scale",
    "dw_kernel_size",
    "without_pointwise",
    "dilation",
]


def build_layer_and_assign(cfg, weights):
    kwargs = {k: cfg[k] for k in LAYER_KWARGS}
    layer = DCNv4(**kwargs)

    B, H, W, C = cfg["BHWC"]
    layer.build((B, H, W, C))

    def assign(attr, value):
        getattr(layer, attr).assign(value)

    assign("offset_mask_kernel", weights["offset_mask.weight"].T)
    assign("offset_mask_bias", weights["offset_mask.bias"])

    if cfg["dw_kernel_size"] is not None:
        # torch depthwise weight [C, 1, kh, kw] -> tf [kh, kw, C, 1]
        assign("offset_mask_dw_kernel", np.transpose(weights["offset_mask_dw.weight"], (2, 3, 0, 1)))
        assign("offset_mask_dw_bias", weights["offset_mask_dw.bias"])

    if not cfg["without_pointwise"]:
        assign("value_proj_kernel", weights["value_proj.weight"].T)
        assign("value_proj_bias", weights["value_proj.bias"])
        assign("output_proj_kernel", weights["output_proj.weight"].T)
        if "output_proj.bias" in weights:
            assign("output_proj_bias", weights["output_proj.bias"])

    if cfg["center_feature_scale"]:
        assign(
            "center_feature_scale_kernel",
            weights["center_feature_scale_proj_weight"].T,
        )
        assign("center_feature_scale_bias", weights["center_feature_scale_proj_bias"])

    return layer


def report(name, a, b, atol):
    a = np.asarray(a, dtype=np.float64)
    b = np.asarray(b, dtype=np.float64)
    max_abs = np.max(np.abs(a - b))
    denom = np.maximum(np.abs(b), 1e-3)
    max_rel = np.max(np.abs(a - b) / denom)
    status = "OK " if max_abs < atol else "FAIL"
    print(f"  [{status}] {name:35s} max_abs={max_abs:.3e} max_rel={max_rel:.3e}")
    return max_abs < atol


def run_parity(parity_dir, dtype="float32"):
    files = sorted(f for f in os.listdir(parity_dir) if f.endswith(".npz"))
    assert len(files) > 0, f"no npz dumps found in {parity_dir}"

    is64 = dtype == "float64"
    out_atol = 1e-5 if is64 else 5e-3
    ref_atol = 1e-5 if is64 else 2e-3
    # fp32 matmuls run with TF32 on both frameworks (default), so fp32 grads
    # carry TF32-level relative noise (~1e-3); float64 proves exact math.
    grad_atol = 2e-4 if is64 else 1e-2
    grad_rtol = 0.0 if is64 else 2e-3

    all_ok = True

    for f in files:
        data = dict(np.load(os.path.join(parity_dir, f), allow_pickle=True))
        cfg = ast.literal_eval(str(data["config"][0]))
        B, H, W, C = cfg["BHWC"]

        print(f"config: {cfg['name']}")

        weights = {k: v for k, v in data.items() if k not in ("x", "ref_out", "ext_out", "gout", "grad_x", "grad_x_ext", "config") and not k.endswith(".grad")}

        layer = build_layer_and_assign(cfg, weights)

        x = data["x"].reshape(B, H, W, C).astype(dtype)
        gout = data["gout"].astype(dtype)

        x_var = tf.Variable(x)
        with tf.GradientTape() as tape:
            out = layer(x_var)
            loss = tf.reduce_sum(out * gout)

        out_np = out.numpy()

        if "ext_out" in data:
            all_ok &= report("out vs official ext", out_np, data["ext_out"], atol=out_atol)
        all_ok &= report("out vs pure-torch ref", out_np, data["ref_out"], atol=ref_atol)

        # gradients: TF must be within the noise envelope of the two torch
        # implementations, i.e. diff(tf, ext) <= max(3 * diff(ref, ext), atol)
        grad_targets = [x_var]
        grad_targets += [layer.offset_mask_kernel, layer.offset_mask_bias]
        if cfg["dw_kernel_size"] is not None:
            grad_targets += [layer.offset_mask_dw_kernel, layer.offset_mask_dw_bias]
        if not cfg["without_pointwise"]:
            grad_targets += [layer.value_proj_kernel, layer.value_proj_bias, layer.output_proj_kernel]
            if "output_proj.bias" in weights:
                grad_targets += [layer.output_proj_bias]
        if cfg["center_feature_scale"]:
            grad_targets += [layer.center_feature_scale_kernel, layer.center_feature_scale_bias]

        grads = tape.gradient(loss, grad_targets)
        grad_x = grads[0].numpy().reshape(B, H * W, C)

        def maxabs(a, b):
            return float(np.max(np.abs(np.asarray(a, np.float64) - np.asarray(b, np.float64))))

        def check3(name, tf_grad, ref_grad, ext_grad, atol=grad_atol):
            nonlocal all_ok
            d_ref = maxabs(tf_grad, ref_grad)
            # fp32: TF32-noise is relative to grad magnitude; f64: strict atol
            ref_scale = float(np.max(np.abs(np.asarray(ref_grad, np.float64))))
            tol = max(atol, grad_rtol * ref_scale)
            if ext_grad is not None:
                d_ext = maxabs(tf_grad, ext_grad)
                d_ref_ext = maxabs(ref_grad, ext_grad)
                ok = d_ref <= tol or d_ext <= max(3.0 * d_ref_ext, atol)
                status = "OK " if ok else "FAIL"
                print(
                    f"  [{status}] grad {name:32s} tf~ref={d_ref:.2e} tf~ext={d_ext:.2e} ref~ext={d_ref_ext:.2e} |ref|max={ref_scale:.1e}"
                )
            else:
                ok = d_ref <= tol
                status = "OK " if ok else "FAIL"
                print(f"  [{status}] grad {name:32s} tf~ref={d_ref:.2e} |ref|max={ref_scale:.1e}")
            all_ok &= ok

        def ext_grad_of(key):
            k = key + ".ext.grad"
            return data[k] if k in data else None

        check3("x", grad_x, data["grad_x"], data["grad_x_ext"] if "grad_x_ext" in data else None)

        gi = 1

        def check_weight_grad(name, key, transpose):
            nonlocal gi
            tf_grad = grads[gi].numpy()
            gi += 1
            ref = data[key + ".grad"]
            ref = ref.T if transpose else ref
            ext = ext_grad_of(key)
            ext = ext.T if (ext is not None and transpose) else ext
            check3(name, tf_grad, ref, ext)

        check_weight_grad("offset_mask_kernel", "offset_mask.weight", True)
        check_weight_grad("offset_mask_bias", "offset_mask.bias", False)

        if cfg["dw_kernel_size"] is not None:
            # torch depthwise grad [C, 1, kh, kw] -> tf [kh, kw, C, 1]
            tf_grad = np.transpose(grads[gi].numpy(), (2, 3, 0, 1)); gi += 1
            check3(
                "offset_mask_dw_kernel",
                tf_grad,
                data["offset_mask_dw.weight.grad"],
                ext_grad_of("offset_mask_dw.weight"),
            )
            check_weight_grad("offset_mask_dw_bias", "offset_mask_dw.bias", False)

        if not cfg["without_pointwise"]:
            check_weight_grad("value_proj_kernel", "value_proj.weight", True)
            check_weight_grad("value_proj_bias", "value_proj.bias", False)
            check_weight_grad("output_proj_kernel", "output_proj.weight", True)
            if "output_proj.bias" in weights:
                check_weight_grad("output_proj_bias", "output_proj.bias", False)

        if cfg["center_feature_scale"]:
            check_weight_grad(
                "center_feature_scale_kernel",
                "center_feature_scale_proj_weight",
                True,
            )
            check_weight_grad(
                "center_feature_scale_bias",
                "center_feature_scale_proj_bias",
                False,
            )

    return all_ok


def run_self_checks():
    print("self checks (no ground truth)")

    x = tf.random.normal([2, 32, 24, 64])

    # XLA vs eager consistency
    for cfg in [dict(), dict(remove_center=True), dict(center_feature_scale=True), dict(dw_kernel_size=5), dict(dilation=2)]:
        layer_eager = DCNv4(kernel_size=3, group=4, use_jit_compile=False, **cfg)
        y_eager = layer_eager(x)

        layer_xla = DCNv4(kernel_size=3, group=4, use_jit_compile=True, **cfg)
        y_xla = layer_xla(x)

        # same weights
        for v_xla, v_eager in zip(layer_xla.trainable_variables, layer_eager.trainable_variables):
            v_xla.assign(v_eager)
        y_xla = layer_xla(x)

        max_abs = np.max(np.abs(y_xla.numpy() - y_eager.numpy()))
        # TF32 matmuls make XLA and eager differ by ~1e-3 on fp32
        status = "OK " if max_abs < 2e-3 else "FAIL"
        print(f"  [{status}] xla vs eager {cfg}: max_abs={max_abs:.3e}")
        assert max_abs < 2e-3

    # training smoke: loss decreases on a tiny overfit task
    layer = DCNv4(kernel_size=3, group=4, center_feature_scale=True, dw_kernel_size=5)
    target = tf.random.normal([2, 32, 24, 64])
    optimizer = tf.keras.optimizers.SGD(learning_rate=0.05)

    losses = []
    for step in range(30):
        with tf.GradientTape() as tape:
            y = layer(x, training=True)
            loss = tf.reduce_mean(tf.square(y - target))
        grads = tape.gradient(loss, layer.trainable_variables)
        assert all(g is not None and tf.reduce_all(tf.math.is_finite(g)) for g in grads)
        optimizer.apply_gradients(zip(grads, layer.trainable_variables))
        losses.append(float(loss))

    print(f"  train smoke: loss {losses[0]:.4f} -> {losses[-1]:.4f}")
    assert losses[-1] < losses[0], "loss did not decrease"


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--parity_dir", default=None)
    parser.add_argument("--dtype", default="float32", choices=["float32", "float64"])
    parser.add_argument("--skip_self_checks", action="store_true")
    args = parser.parse_args()

    if not args.skip_self_checks:
        run_self_checks()

    if args.parity_dir is not None:
        ok = run_parity(args.parity_dir, dtype=args.dtype)
        print("parity:", "PASS" if ok else "FAIL")
        if not ok:
            sys.exit(1)

    print("all done")


if __name__ == "__main__":
    main()
