# ================================================================
# MIT License
# Copyright (c) 2021 edwardyehuang (https://github.com/edwardyehuang)
# ================================================================

import os
import tensorflow as tf

from iseg.utils.common import set_random_seed, enable_mixed_precision
from iseg.utils.distribution_utils import get_distribution_strategy

def common_env_setup (run_eagerly = False,
                      gpu_memory_growth = True,
                      cuda_visible_devices = None,
                      tpu_name = None,
                      random_seed = 0,
                      mixed_precision = True,
                      use_deterministic = True):

    if use_deterministic:
        os.environ['TF_DETERMINISTIC_OPS'] = '1'
        os.environ['TF_DISABLE_SEGMENT_REDUCTION_OP_DETERMINISM_EXCEPTIONS'] = '1'


    tf.config.run_functions_eagerly(run_eagerly)

    strategy = get_distribution_strategy(gpu_memory_growth,
                                         cuda_visible_devices,
                                         tpu_name is not None,
                                         tpu_name)


    set_random_seed(random_seed)

    if mixed_precision:
        enable_mixed_precision(use_tpu = tpu_name is not None)


    return strategy

    

    