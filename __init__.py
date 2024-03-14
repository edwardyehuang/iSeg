# ====================================================================
# MIT License
# Copyright (c) 2024 edwardyehuang (https://github.com/edwardyehuang)
# ====================================================================

name = "iseg"
from iseg.core_model import SegBase, SegFoundation
from iseg.utils.value_check import check_numerics, set_check_numerics_level

import iseg.utils.tensor_convert_utils