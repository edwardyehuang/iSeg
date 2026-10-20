# ===================================================================
# MIT License
# Copyright (c) 2024 edwardyehuang (https://github.com/edwardyehuang)
# ===================================================================

# Compatibility shim: DCNv2 moved to iseg/layers/dcnv2/ (package).
# Prefer `from iseg.layers.dcnv2 import DCNv2`; this module re-exports it so
# legacy `iseg.layers.dcn_v2` imports keep working.

from iseg.layers.dcnv2 import DCNv2, DCNv2_Base, DCNv2_Flash  # noqa: F401
