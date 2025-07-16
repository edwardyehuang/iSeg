# ================================================================
# MIT License
# Copyright (c) 2021 edwardyehuang (https://github.com/edwardyehuang)
# ================================================================

import os
import tensorflow as tf

from distutils.version import LooseVersion

from iseg.utils.common import set_random_seed, enable_mixed_precision
from iseg.utils.distribution_utils import get_distribution_strategy, shutdown_tpu_system


def common_env_setup(
    run_eagerly=False,
    gpu_memory_growth=True,
    cuda_visible_devices=None,
    use_one_device_strategy=False,
    tpu_name=None,
    random_seed=0,
    mixed_precision=True,
    use_deterministic=True,
    num_op_parallelism_threads=-1,
    numpy_behavior=False,
    soft_device_placement=False,
):
    
    set_random_seed(random_seed)

    tf.get_logger().setLevel(0)

    use_tpu = tpu_name is not None

    print(f"Using TPU: {use_tpu}")
    print(f"use_deterministic = {use_deterministic}")

    if use_deterministic and not use_tpu:
        if LooseVersion(tf.version.VERSION) < LooseVersion("2.8.0"):
            os.environ["TF_DETERMINISTIC_OPS"] = "1"
            os.environ["TF_DISABLE_SEGMENT_REDUCTION_OP_DETERMINISM_EXCEPTIONS"] = "1"  # For 2.5.0+
        else:
            tf.config.experimental.enable_op_determinism()

        if LooseVersion(tf.version.VERSION) >= LooseVersion("2.5.0"):
            os.environ["TF_CUDNN_USE_FRONTEND"] = "1"

    if num_op_parallelism_threads is not None and num_op_parallelism_threads > 0:
        tf.config.threading.set_inter_op_parallelism_threads(num_op_parallelism_threads)
        tf.config.threading.set_intra_op_parallelism_threads(num_op_parallelism_threads)

    if soft_device_placement:
        tf.config.set_soft_device_placement(soft_device_placement)

    tf.config.run_functions_eagerly(run_eagerly)

    if run_eagerly and LooseVersion(tf.version.VERSION) >= LooseVersion("2.8.0"):
        tf.data.experimental.enable_debug_mode()

    strategy = get_distribution_strategy(
        gpu_memory_growth=gpu_memory_growth, 
        cuda_visible_devices=cuda_visible_devices, 
        use_tpu=use_tpu, 
        tpu_name=tpu_name,
        use_one_device_strategy=use_one_device_strategy,
    )

    if mixed_precision:
        enable_mixed_precision(use_tpu=tpu_name is not None)

    if numpy_behavior:
        tf.experimental.numpy.experimental_enable_numpy_behavior(True)
        print("Enable experimental numpy behavior")

    return strategy


def common_env_clean (strategy):

    if isinstance(strategy, tf.distribute.TPUStrategy):
        shutdown_tpu_system(strategy)