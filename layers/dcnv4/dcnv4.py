# ===================================================================
# MIT License
# Copyright (c) 2024 edwardyehuang (https://github.com/edwardyehuang)
# ===================================================================

# Auto-routing DCNv4: picks the aggregation core from the distribution context
#   - TPU strategy  -> dense window-based core (dcnv4_tpu.dcnv4_core_tpu)
#   - otherwise     -> gather-based core (dcnv4_base.dcnv4_core, GPU-friendly)
#
# The routing is a method override resolved at trace time, so DCNv4 stays a
# normal keras layer class (isinstance / subclassing / serialization all keep
# working) and one layer instance (same weights) can serve both backends.
#
# For an explicitly pinned implementation, use DCNv4_Base (gather) or
# DCNv4_TPU (window-dense) directly.

import tensorflow as tf

from iseg.layers.dcnv4.dcnv4_base import DCNv4_Base, dcnv4_core  # re-exported
from iseg.layers.dcnv4.dcnv4_tpu import dcnv4_core_tpu

# Memory safety valve: the window core materializes a patches tensor of
# value_volume x T^2 (up to 64x); at very large feature volumes this alone
# can dominate TPU HBM in a full model. Above this per-replica value volume,
# fall back to the gather core (no amplification). FaPN-scale production
# (56x56x256 at 64/replica -> 12.8M) stays on the window core.
_WINDOW_CORE_MAX_VOLUME = 1 << 25  # 33.5M


def _is_tpu_context():
    """True when the ambient distribution strategy targets TPUs."""
    if not tf.distribute.has_strategy():
        return False
    strategy = tf.distribute.get_strategy()
    if isinstance(strategy, tf.distribute.TPUStrategy):
        return True
    # DTensor-backed strategies expose their mesh; check the device type
    mesh = getattr(strategy, "_mesh", None)
    if mesh is not None and mesh.device_type() == "TPU":
        return True
    return False


class DCNv4(DCNv4_Base):
    """DCNv4 that routes its aggregation core to the current backend.

    window_radius: only used by the TPU core (see dcnv4_tpu.dcnv4_core_tpu);
        "exact" forces the reference gather core even on TPU.
    """

    def __init__(self, *args, window_radius=2, **kwargs):
        super().__init__(*args, **kwargs)
        self.window_radius = window_radius

    def _aggregate_core(self, value, offset_mask):
        if self.window_radius != "exact" and _is_tpu_context():
            bs, h, w, c = value.shape
            volume = int(bs) * int(h) * int(w) * int(c)
            # Memory safety valve: the window core materializes a patches
            # tensor of value x T^2 (up to 64x). Only fall back to the gather
            # core at huge volumes AND small spatial maps -- the gather core's
            # scatter backward is catastrophically slow at large spatial maps,
            # so it is never a win there; the valve exists to keep a full
            # model's TPU HBM from blowing up at small-map/huge-batch extremes.
            use_window = volume <= _WINDOW_CORE_MAX_VOLUME or max(h, w) > 64
            if use_window:
                return dcnv4_core_tpu(
                    value,
                    offset_mask,
                    kernel_size=self.kernel_size,
                    stride=self.stride,
                    pad=self.pad,
                    dilation=self.dilation,
                    group=self.group,
                    offset_scale=self.offset_scale,
                    remove_center=self.remove_center,
                    window_radius=self.window_radius,
                )
        return super()._aggregate_core(value, offset_mask)

    def call(self, inputs, training=None):
        # client-side jit_compile is irrelevant (and harmful) on TPU remote
        # clients; TPU compilation happens server-side via tpu.replicate
        if _is_tpu_context():
            if self.use_custom_offset:
                x, offset = tuple(inputs)
            else:
                x = offset = inputs
            return self._forward(x, offset)
        return super().call(inputs, training=training)

    def get_config(self):
        config = {"window_radius": self.window_radius}
        base_config = super().get_config()
        return dict(list(base_config.items()) + list(config.items()))
