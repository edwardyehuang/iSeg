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
