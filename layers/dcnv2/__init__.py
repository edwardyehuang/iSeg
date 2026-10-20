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

from iseg.layers.dcnv2.dcnv2 import DCNv2_Base
from iseg.layers.dcnv2.dcnv2_flash import DCNv2_Flash


class DCNv2(DCNv2_Flash):
    """DCNv2 with a selectable implementation.

    implementation:
      - "flash" (default): XLA-optimized forward (DCNv2_Flash)
      - "fallback": reference implementation (DCNv2_Base)
    """

    def __init__(self, *args, implementation="flash", **kwargs):
        if implementation not in ("flash", "fallback"):
            raise ValueError(
                f"implementation must be 'flash' or 'fallback', got {implementation!r}"
            )
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
