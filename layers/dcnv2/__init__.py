# ===================================================================
# MIT License
# Copyright (c) 2024 edwardyehuang (https://github.com/edwardyehuang)
# ===================================================================

# DCNv2 interface with selectable implementation:
#   DCNv2(..., implementation="flash")    -> XLA-optimized path (default)
#   DCNv2(..., implementation="fallback") -> reference implementation
# Both share identical weight names/shapes, so checkpoints and configs are
# interchangeable. Pin the implementation explicitly via DCNv2_Flash /
# DCNv2_Base when needed.

from iseg.distribution.distribution_utils import is_tpu_context
from iseg.layers.dcnv2.dcnv2 import DCNv2_Base
from iseg.layers.dcnv2.dcnv2_flash import DCNv2_Flash


class DCNv2(DCNv2_Flash):
    """DCNv2 with a selectable implementation.

    implementation:
      - "flash" (default): XLA-optimized forward (DCNv2_Flash)
      - "auto": automatically select implementation based on context
      - "fallback": reference implementation (DCNv2_Base)
    """

    def __init__(self, *args, implementation="auto", **kwargs):
        if implementation not in ("auto", "flash", "fallback"):
            raise ValueError(
                f"implementation must be 'auto', 'flash' or 'fallback', got {implementation!r}"
            )

        if not is_tpu_context() and implementation == "auto":
            implementation = "fallback"
        
        super().__init__(*args, **kwargs)
        self.implementation = implementation

    def _forward(self, x, offset):
        if self.implementation == "fallback":
            return DCNv2_Base._forward(self, x, offset)
        return super()._forward(x, offset)

    def get_config(self):
        config = {"implementation": self.implementation}
        base_config = super().get_config()
        return dict(list(base_config.items()) + list(config.items()))
