# ================================================================
# MIT License
# Copyright (c) 2021 edwardyehuang (https://github.com/edwardyehuang)
# ================================================================

import os
import tensorflow as tf

from distutils.version import LooseVersion

from iseg.utils.common import set_random_seed, enable_mixed_precision
from iseg.utils.distribution_utils import get_distribution_strategy


def common_env_setup(
    run_eagerly=False,
    gpu_memory_growth=True,
    cuda_visible_devices=None,
    tpu_name=None,
    random_seed=0,
    mixed_precision=True,
    use_deterministic=True,
    numpy_behavior=False,
):

    if use_deterministic:
        if LooseVersion(tf.version.VERSION) < LooseVersion("2.8.0"):
            os.environ["TF_DETERMINISTIC_OPS"] = "1"
            os.environ["TF_DISABLE_SEGMENT_REDUCTION_OP_DETERMINISM_EXCEPTIONS"] = "1"  # For 2.5.0+
        else:
            tf.config.experimental.enable_op_determinism()

        if LooseVersion(tf.version.VERSION) >= LooseVersion("2.5.0"):
            os.environ["TF_CUDNN_USE_FRONTEND"] = "1"

        if tpu_name is not None:
            tf.config.threading.set_intra_op_parallelism_threads(1)
            tf.config.threading.set_inter_op_parallelism_threads(1)

    set_random_seed(random_seed)

    tf.config.run_functions_eagerly(run_eagerly)

    if run_eagerly and LooseVersion(tf.version.VERSION) >= LooseVersion("2.8.0"):
        tf.data.experimental.enable_debug_mode()

    strategy = get_distribution_strategy(gpu_memory_growth, cuda_visible_devices, tpu_name is not None, tpu_name)

    if mixed_precision:
        enable_mixed_precision(use_tpu=tpu_name is not None)

    if numpy_behavior:
        tf.experimental.numpy.experimental_enable_numpy_behavior(True)
        print("Enable experimental numpy behavior")

    return strategy
