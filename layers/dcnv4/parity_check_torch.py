# ===================================================================
# MIT License
# Copyright (c) 2024 edwardyehuang (https://github.com/edwardyehuang)
# ===================================================================

# Torch-side ground-truth dumper for TF DCNv4 parity check.
# Run with the dcnv4 conda env:
#   python iseg/layers/dcnv4/parity_check_torch.py --out_dir /tmp/dcnv4_parity
#
# If the official CUDA extension is available it is used as ground truth,
# otherwise a pure-torch reference that mirrors the CUDA kernel semantics
# (dcnv4_im2col_cuda.cuh) is used.

import argparse
import math
import os

import numpy as np
import torch
import torch.nn.functional as F

# strict fp32 ground truth: disable TF32 so the reference is not
# reduced-precision (TF32 has only 10 mantissa bits)
torch.backends.cuda.matmul.allow_tf32 = False
torch.backends.cudnn.allow_tf32 = False


def build_points(kh, kw, remove_center):
    center_h, center_w = kh // 2, kw // 2
    return [
        (i, j)
        for i in range(kw)
        for j in range(kh)
        if not (remove_center and i == center_w and j == center_h)
    ]


def dcnv4_core_ref(
    value, offset_mask, kernel_size, stride, pad, dilation, group, offset_scale, remove_center
):
    """Pure torch reference mirroring dcnv4_im2col_cuda.cuh forward semantics.

    value:       [B, H, W, C]  (channels-last, like the TF port)
    offset_mask: [B, Hout, Wout, >= group * K * 3]
    """
    kh, kw = kernel_size
    sh, sw = stride
    ph, pw = pad
    dh, dw = dilation

    B, H, W, C = value.shape
    Hout, Wout = offset_mask.shape[1], offset_mask.shape[2]
    D = C // group

    points = build_points(kh, kw, remove_center)
    K = len(points)

    center_off_h = (dh * (kh - 1)) // 2
    center_off_w = (dw * (kw - 1)) // 2

    om = offset_mask[..., : group * K * 3].reshape(B, Hout, Wout, group, 3 * K)

    output = torch.zeros(B, Hout, Wout, C, dtype=value.dtype, device=value.device)

    qy = torch.arange(Hout, dtype=value.dtype, device=value.device)[:, None]
    qx = torch.arange(Wout, dtype=value.dtype, device=value.device)

    for g in range(group):
        om_g = om[:, :, :, g]  # [B, Hout, Wout, 3K]
        offs = om_g[..., : 2 * K].reshape(B, Hout, Wout, K, 2)
        # DCNv4 uses raw mask logits (official dcnv4_func.py passes softmax=False)
        mask = om_g[..., 2 * K :]  # [B, Hout, Wout, K]

        for p, (i, j) in enumerate(points):
            off_x = offs[..., p, 0] * offset_scale
            off_y = offs[..., p, 1] * offset_scale

            # w_im = p0_w_ + (i * dw + off) * scale
            # p0_w_ = center_off_w - pw + qx * sw - center_off_w * scale
            x_pos = (
                (center_off_w - pw)
                + qx * sw
                - center_off_w * offset_scale
                + i * dw * offset_scale
                + off_x
            )  # [B, Hout, Wout]
            y_pos = (
                (center_off_h - ph)
                + qy * sh
                - center_off_h * offset_scale
                + j * dh * offset_scale
                + off_y
            )

            ix0 = torch.floor(x_pos).detach()
            iy0 = torch.floor(y_pos).detach()
            ix1 = ix0 + 1.0
            iy1 = iy0 + 1.0

            lx = x_pos - ix0
            ly = y_pos - iy0

            w1 = (1.0 - lx) * (1.0 - ly)
            w2 = lx * (1.0 - ly)
            w3 = (1.0 - lx) * ly
            w4 = lx * ly

            point_ok = ((x_pos > -1) & (x_pos < W) & (y_pos > -1) & (y_pos < H)).float()
            c1_ok = ((ix0 >= 0) & (iy0 >= 0)).float()
            c2_ok = ((ix1 < W) & (iy0 >= 0)).float()
            c3_ok = ((ix0 >= 0) & (iy1 < H)).float()
            c4_ok = ((ix1 < W) & (iy1 < H)).float()

            attn = mask[..., p] * point_ok

            ix0c = ix0.clamp(0, W - 1).long()
            ix1c = ix1.clamp(0, W - 1).long()
            iy0c = iy0.clamp(0, H - 1).long()
            iy1c = iy1.clamp(0, H - 1).long()

            v_g = value[..., g * D : (g + 1) * D].reshape(B, H * W, D)  # [B, H*W, D]

            def _gather(iy, ix):
                idx = (iy * W + ix).reshape(B, Hout * Wout, 1).expand(-1, -1, D)
                return torch.gather(v_g, 1, idx).reshape(B, Hout, Wout, D)

            v1 = _gather(iy0c, ix0c)
            v2 = _gather(iy0c, ix1c)
            v3 = _gather(iy1c, ix0c)
            v4 = _gather(iy1c, ix1c)

            sample = (
                w1[..., None] * c1_ok[..., None] * v1
                + w2[..., None] * c2_ok[..., None] * v2
                + w3[..., None] * c3_ok[..., None] * v3
                + w4[..., None] * c4_ok[..., None] * v4
            )

            output[..., g * D : (g + 1) * D] += attn[..., None] * sample

    return output


def ref_forward(module, x, cfg):
    """Full-module forward using the pure-torch reference core (channels-last)."""
    B, H, W, C = cfg["BHWC"]
    kh = kw = cfg["kernel_size"]
    group = cfg["group"]
    offset_scale = cfg["offset_scale"]
    remove_center = cfg["remove_center"]
    center_feature_scale = cfg["center_feature_scale"]
    dw_kernel_size = cfg["dw_kernel_size"]
    without_pointwise = cfg["without_pointwise"]
    dilation = cfg["dilation"]

    x4 = x.reshape(B, H, W, C)

    if without_pointwise:
        value = x4
    else:
        value = F.linear(x, module.value_proj.weight, module.value_proj.bias).reshape(B, H, W, C)

    if dw_kernel_size is not None:
        om_in = F.conv2d(
            x4.permute(0, 3, 1, 2),
            module.offset_mask_dw.weight,
            module.offset_mask_dw.bias,
            stride=1,
            padding=(dw_kernel_size - 1) // 2,
            groups=C,
        ).permute(0, 2, 3, 1)
    else:
        om_in = x4

    offset_mask = F.linear(om_in, module.offset_mask.weight, module.offset_mask.bias)

    out = dcnv4_core_ref(
        value,
        offset_mask,
        (kh, kw),
        (1, 1),
        ((dilation * (kh - 1)) // 2, (dilation * (kw - 1)) // 2),
        (dilation, dilation),
        group,
        offset_scale,
        remove_center,
    )

    if center_feature_scale:
        scale = F.linear(
            out.reshape(B, H * W, C),
            module.center_feature_scale_proj_weight,
            module.center_feature_scale_proj_bias,
        ).sigmoid()
        scale = scale[..., None].repeat(1, 1, 1, C // group).flatten(-2).reshape(B, H, W, C)
        out = out * (1 - scale) + value * scale

    if not without_pointwise:
        out = F.linear(
            out.reshape(B, H * W, C), module.output_proj.weight, module.output_proj.bias
        ).reshape(B, H, W, C)

    return out


def run_config(cfg, out_dir, use_ext, dtype):
    from DCNv4.modules.dcnv4 import DCNv4 as TorchDCNv4

    seed = cfg["seed"]
    torch.manual_seed(seed)

    B, H, W, C = cfg["BHWC"]
    kh = kw = cfg["kernel_size"]
    group = cfg["group"]
    offset_scale = cfg["offset_scale"]
    remove_center = cfg["remove_center"]
    center_feature_scale = cfg["center_feature_scale"]
    dw_kernel_size = cfg["dw_kernel_size"]
    without_pointwise = cfg["without_pointwise"]
    dilation = cfg["dilation"]

    name = cfg["name"]

    module = TorchDCNv4(
        channels=C,
        kernel_size=kh,
        stride=1,
        pad=(dilation * (kh - 1)) // 2,
        dilation=dilation,
        group=group,
        offset_scale=offset_scale,
        dw_kernel_size=dw_kernel_size,
        center_feature_scale=center_feature_scale,
        remove_center=remove_center,
        output_bias=True,
        without_pointwise=without_pointwise,
    ).cuda().to(dtype)

    x = torch.randn(B, H * W, C, dtype=dtype, device="cuda")

    # replace random weights with seeded random ones (offset_mask stays zeros
    # in official init; randomize it so offsets/masks are actually exercised)
    with torch.no_grad():
        for p in module.parameters():
            p.copy_(torch.randn_like(p) * 0.1)

    # ------- reference (pure torch) -------
    with torch.no_grad():
        ref_out = ref_forward(module, x, cfg)

    # upstream gradient for backward parity
    gout = torch.randn(B, H, W, C, dtype=dtype, device="cuda")

    module.zero_grad()
    x_req = x.clone().requires_grad_(True)
    ref_out_b = ref_forward(module, x_req, cfg)
    (ref_out_b * gout).sum().backward()
    grad_x = x_req.grad

    ext_out = None
    if use_ext:
        with torch.no_grad():
            ext_out = module(x, shape=(H, W)).reshape(B, H, W, C)
        diff = (ext_out - ref_out).abs().max().item()
        print(f"[{name}] official-ext vs pure-torch-ref max abs diff: {diff:.3e}")

    dump = {
        "x": x.cpu().numpy(),
        "ref_out": ref_out.detach().cpu().numpy(),
        "gout": gout.cpu().numpy(),
        "grad_x": grad_x.cpu().numpy(),
        "config": np.array([str(cfg)]),
    }
    if ext_out is not None:
        dump["ext_out"] = ext_out.cpu().numpy()

    named_params = {
        "offset_mask": module.offset_mask,
        "offset_mask_dw": getattr(module, "offset_mask_dw", None),
        "value_proj": getattr(module, "value_proj", None),
        "output_proj": getattr(module, "output_proj", None),
    }
    for wname, p in named_params.items():
        if p is None:
            continue
        dump[f"{wname}.weight"] = p.weight.detach().cpu().numpy()
        if p.weight.grad is not None:
            dump[f"{wname}.weight.grad"] = p.weight.grad.detach().cpu().numpy()
        if p.bias is not None:
            dump[f"{wname}.bias"] = p.bias.detach().cpu().numpy()
            if p.bias.grad is not None:
                dump[f"{wname}.bias.grad"] = p.bias.grad.detach().cpu().numpy()

    if center_feature_scale:
        for wname, p in [
            ("center_feature_scale_proj_weight", module.center_feature_scale_proj_weight),
            ("center_feature_scale_proj_bias", module.center_feature_scale_proj_bias),
        ]:
            dump[wname] = p.detach().cpu().numpy()
            if p.grad is not None:
                dump[f"{wname}.grad"] = p.grad.detach().cpu().numpy()

    # ------- official ext backward -------
    if use_ext:
        module.zero_grad()
        x_req2 = x.clone().requires_grad_(True)
        ext_out_b = module(x_req2, shape=(H, W))
        (ext_out_b.reshape(B, H, W, C) * gout).sum().backward()
        dump["grad_x_ext"] = x_req2.grad.cpu().numpy()
        for wname, p in named_params.items():
            if p is None:
                continue
            if p.weight.grad is not None:
                dump[f"{wname}.weight.ext.grad"] = p.weight.grad.detach().cpu().numpy()
            if p.bias is not None and p.bias.grad is not None:
                dump[f"{wname}.bias.ext.grad"] = p.bias.grad.detach().cpu().numpy()
        if center_feature_scale:
            for wname, p in [
                ("center_feature_scale_proj_weight", module.center_feature_scale_proj_weight),
                ("center_feature_scale_proj_bias", module.center_feature_scale_proj_bias),
            ]:
                if p.grad is not None:
                    dump[f"{wname}.ext.grad"] = p.grad.detach().cpu().numpy()

    np.savez(os.path.join(out_dir, f"{name}.npz"), **dump)
    print(f"[{name}] dumped to {out_dir}/{name}.npz")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--out_dir", default="/tmp/dcnv4_parity")
    parser.add_argument("--double", action="store_true", help="dump float64 ground truth (reference only)")
    args = parser.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)

    dtype = torch.float64 if args.double else torch.float32

    use_ext = False
    if not args.double:
        try:
            from DCNv4 import ext  # noqa: F401

            use_ext = True
            print("official CUDA extension available")
        except Exception as e:
            print(f"official CUDA extension NOT available ({e}), using pure-torch reference")

    configs = [
        dict(
            name="default_g4_c64",
            seed=11,
            BHWC=(2, 32, 24, 64),
            kernel_size=3,
            group=4,
            offset_scale=1.0,
            remove_center=False,
            center_feature_scale=False,
            dw_kernel_size=None,
            without_pointwise=False,
            dilation=1,
        ),
        dict(
            name="remove_center",
            seed=22,
            BHWC=(2, 32, 24, 64),
            kernel_size=3,
            group=4,
            offset_scale=1.0,
            remove_center=True,
            center_feature_scale=False,
            dw_kernel_size=None,
            without_pointwise=False,
            dilation=1,
        ),
        dict(
            name="center_feature_scale",
            seed=33,
            BHWC=(2, 32, 24, 64),
            kernel_size=3,
            group=4,
            offset_scale=0.5,
            remove_center=False,
            center_feature_scale=True,
            dw_kernel_size=None,
            without_pointwise=False,
            dilation=1,
        ),
        dict(
            name="dw5_offset05",
            seed=44,
            BHWC=(2, 32, 24, 64),
            kernel_size=3,
            group=4,
            offset_scale=0.5,
            remove_center=False,
            center_feature_scale=False,
            dw_kernel_size=5,
            without_pointwise=False,
            dilation=1,
        ),
        dict(
            name="without_pointwise",
            seed=55,
            BHWC=(2, 32, 24, 64),
            kernel_size=3,
            group=4,
            offset_scale=1.0,
            remove_center=False,
            center_feature_scale=False,
            dw_kernel_size=None,
            without_pointwise=True,
            dilation=1,
        ),
        dict(
            name="dilation2_g8_c128",
            seed=66,
            BHWC=(2, 32, 24, 128),
            kernel_size=3,
            group=8,
            offset_scale=1.0,
            remove_center=False,
            center_feature_scale=False,
            dw_kernel_size=None,
            without_pointwise=False,
            dilation=2,
        ),
    ]

    for cfg in configs:
        run_config(cfg, args.out_dir, use_ext, dtype)


if __name__ == "__main__":
    main()
