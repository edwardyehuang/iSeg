# ================================================================
# MIT License
# Copyright (c) 2021 edwardyehuang (https://github.com/edwardyehuang)
# ================================================================

"""Mesh and DTensor utilities for distributed training.

This module builds on TensorFlow 2.15's native DTensor support and exposes a
`tf.distribute.Strategy` backed by DTensor (`tf.python.distribute.experimental.
mirrored_strategy.MirroredStrategy`). The returned strategy can be used as a
drop-in replacement for the classic `MirroredStrategy` / `TPUStrategy` in the
rest of the codebase.
"""

import tensorflow as tf

from iseg.distribution.distribution_utils import (
    initialize_tpu_system,
    list_gpus,
    set_gpu_memory_growth,
)

try:
    from tensorflow.python.distribute.experimental.mirrored_strategy import (
        MirroredStrategy as DTensorMirroredStrategy,
    )
except ImportError as e:
    raise ImportError(
        "DTensor-backed MirroredStrategy is not available. "
        "Please ensure TensorFlow >= 2.15 is installed."
    ) from e


def get_dtensor_strategy(
    gpu_memory_growth=True,
    cuda_visible_devices=None,
    use_tpu=False,
    tpu_name=None,
    use_one_device_strategy=False,
):
    """Create a DTensor-backed `tf.distribute.Strategy`.

    The strategy is synchronous data-parallel: the mesh is 1-D along the
    `"batch"` dimension, inputs are batch-sharded, and variables are replicated.
    This matches the behavior of classic `MirroredStrategy` / `TPUStrategy` but
    is implemented on top of DTensor SPMD.

    Args:
      gpu_memory_growth: Whether to enable GPU memory growth. Only used when
        the accelerator type is GPU.
      cuda_visible_devices: Ignored in mesh mode; device visibility is expected
        to be controlled via the `CUDA_VISIBLE_DEVICES` environment variable
        before the process starts.
      use_tpu: Whether to target TPU devices.
      tpu_name: The TPU name or grpc target, passed to the existing TPU
        initialization helper to connect to the TPU system.
      use_one_device_strategy: Ignored in mesh mode; a mesh can be trivially
        1-D with a single device if only one device is available.

    Returns:
      A `tf.python.distribute.experimental.mirrored_strategy.MirroredStrategy`
      instance backed by a DTensor mesh.
    """
    device_type = _get_accelerator_type(use_tpu=use_tpu)

    if device_type == "TPU":
        # Reuse the existing TPU connection helper so that the `tpu_name` flag
        # keeps working. After initialization, logical TPU devices are visible
        # and can be used to build a DTensor mesh.
        initialize_tpu_system(tpu_name)
        tpu_devices = tf.config.list_logical_devices("TPU")
        mesh = tf.experimental.dtensor.create_mesh(
            [("batch", len(tpu_devices))],
            devices=[d.name for d in tpu_devices],
        )
        return DTensorMirroredStrategy(mesh=mesh)

    if device_type == "GPU":
        set_gpu_memory_growth(gpu_memory_growth)
        gpu_devices = tf.config.list_logical_devices("GPU")
        devices = [d.name for d in gpu_devices]
    else:
        devices = [d.name for d in tf.config.list_logical_devices("CPU")]

    if not devices:
        raise RuntimeError(
            f"No {device_type} devices found for mesh training."
        )

    return DTensorMirroredStrategy(devices=devices)


def get_data_parallel_layout_map(mesh, batch_dim="batch"):
    """Create a LayoutMap for pure data parallelism.

    In pure data parallelism all weights are replicated and only the input
    batch dimension is sharded by the strategy. An empty `LayoutMap` with the
    mesh set will cause Keras to assign a default replicated layout to every
    variable created inside `layout_map.scope()`.

    This is a building block for future model-parallel support: callers can add
    regex-based entries to shard specific weights once they move beyond pure
    data parallelism.

    Args:
      mesh: A `tf.experimental.dtensor.Mesh` instance.
      batch_dim: Name of the batch dimension in the mesh (unused in pure data
        parallelism, kept for API consistency).

    Returns:
      A `tf.keras.dtensor.experimental.LayoutMap` configured for the mesh.
    """
    del batch_dim  # Unused in pure data parallelism.
    from tensorflow.keras.dtensor.experimental import LayoutMap

    return LayoutMap(mesh=mesh)


def get_compute_device_list(use_tpu=False):
    """Return the list of physical devices to use for mesh creation."""
    if use_tpu:
        return tf.config.list_physical_devices("TPU")

    gpus = list_gpus()
    if gpus:
        return gpus

    return tf.config.list_physical_devices("CPU")


def _get_accelerator_type(use_tpu=False):
    """Infer the accelerator type for mesh training."""
    if use_tpu:
        return "TPU"
    if list_gpus():
        return "GPU"
    return "CPU"


# Backwards-compatible alias used by the existing `use_mesh` branch in
# `iseg/core_env.py`. It now returns a Strategy instead of a LayoutMap.
get_mesh_layout_map = get_dtensor_strategy
