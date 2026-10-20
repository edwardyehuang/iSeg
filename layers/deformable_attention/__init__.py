# ===================================================================
# MIT License
# Copyright (c) 2026 edwardyehuang (https://github.com/edwardyehuang)
# ===================================================================

# Deformable attention (Deformable DETR style, single-scale, NHWC).
# Gather-based core (GPU-friendly); runs on TPU as well, but the gather
# backward lowers to scatter there, so expect reduced throughput.

from iseg.layers.deformable_attention.deformable_attention import (
    DeformableAttention,
    deformable_attention_core,
)
