# ================================================================
# MIT License
# Copyright (c) 2021 edwardyehuang (https://github.com/edwardyehuang)
# ================================================================

import tensorflow as tf
import os

from distutils.version import LooseVersion
from platform import uname


def get_tpu_strategy(name=None):

    if LooseVersion(tf.version.VERSION) >= LooseVersion("2.19.0"):
        print("Set TPU name to None for 2.19.0+")
        name = None

    cluster_resolver = tf.distribute.cluster_resolver.TPUClusterResolver(name)

    print("Connecting to TPU cluster ...")

    tf.config.experimental_connect_to_cluster(cluster_resolver)
    
    print("Initializing TPU system ...")

    tf.tpu.experimental.initialize_tpu_system(cluster_resolver)
    print("TPU Device:", cluster_resolver.master())

    print("Creating TPU strategy ...")

    if LooseVersion(tf.version.VERSION) < LooseVersion("2.4.0"):
        strategy = tf.distribute.experimental.TPUStrategy(cluster_resolver)
    else:
        strategy = tf.distribute.TPUStrategy(cluster_resolver)

    print(f"TPU worker_devices: {strategy.extended.worker_devices}")

    return strategy


def shutdown_tpu_system (strategy):

    cluster_resolver = strategy.cluster_resolver
    tf.tpu.experimental.shutdown_tpu_system(cluster_resolver)


def get_cpu_strategy():
    return tf.distribute.OneDeviceStrategy("/cpu:0")


def build_one_device_strategy(device=None):

    if device is None or len(device) == 0:
        gpus = list_gpus()
        if gpus:
            device = gpus[0].name
        else:
            device = "/cpu:0"
    else:
        device = device[0]

    print(f"Using OneDeviceStrategy with device: {device}")

    return tf.distribute.OneDeviceStrategy(device=device)


def build_mirrored_strategy(
    dist_devices=None,
):
    cross_device_ops = None

    if os.name == "nt" or "microsoft-standard" in uname().release:
        cross_device_ops = tf.distribute.HierarchicalCopyAllReduce()
    
    if LooseVersion(tf.version.VERSION) >= LooseVersion("2.21.0"):
        strategy = tf.distribute.experimental.MultiWorkerMirroredStrategy(
            communication=tf.distribute.experimental.CollectiveCommunication.RING
        )
    else:
        strategy = tf.distribute.MirroredStrategy(
            devices=dist_devices, cross_device_ops=cross_device_ops
        )
    # issue 41539 may be fixed: https://github.com/tensorflow/tensorflow/issues/41539
    # strategy = tf.distribute.experimental.MultiWorkerMirroredStrategy(communication= tf.distribute.experimental.CollectiveCommunication.RING)
    # New issue 62234 : https://github.com/tensorflow/tensorflow/issues/62234

    return strategy


def get_distribution_strategy(
    gpu_memory_growth=True, 
    cuda_visible_devices=None, 
    use_tpu=False, 
    tpu_name=None,
    use_one_device_strategy=False,
):

    if use_tpu:
        if tpu_name == "colab":
            tpu_name = None

        print(f"Use TPU strategy with name={tpu_name}")

        return get_tpu_strategy(tpu_name)
    
    dist_devices = _handle_cuda_visible_devices(cuda_visible_devices)

    set_gpu_memory_growth(gpu_memory_growth)

    if use_one_device_strategy:
        return build_one_device_strategy(dist_devices)

    return build_mirrored_strategy(dist_devices)


def list_gpus():

    if LooseVersion(tf.version.VERSION) < LooseVersion("2.8.0"):
        gpus = tf.config.experimental.list_physical_devices("GPU")
    else:
        gpus = tf.config.list_physical_devices("GPU")

    return gpus


def set_gpu_memory_growth(growth=False):

    gpus = list_gpus()

    if gpus:
        try:
            for gpu in gpus:
                tf.config.experimental.set_memory_growth(gpu, growth)

        except RuntimeError as e:
            print(e)


def get_gpu_counts():

    gpus = list_gpus()

    if gpus:
        return len(gpus)
    else:
        return 0
    


def all_reduce_values (
    vars,
    reduce_op=tf.distribute.ReduceOp.SUM
):
    replica_ctx : tf.distribute.ReplicaContext = tf.distribute.get_replica_context()
    
    reduced = replica_ctx.all_reduce(
        reduce_op=reduce_op, 
        value=vars,
    )

    return reduced


def _handle_cuda_visible_devices(cuda_visible_devices=None):

    if cuda_visible_devices is None:
        return None
    
    if isinstance(cuda_visible_devices, str):
        cuda_visible_devices = cuda_visible_devices.split(",")

    if isinstance(cuda_visible_devices, tuple):
        cuda_visible_devices = list(cuda_visible_devices)

    return cuda_visible_devices