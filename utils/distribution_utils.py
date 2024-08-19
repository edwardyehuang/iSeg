# ================================================================
# MIT License
# Copyright (c) 2021 edwardyehuang (https://github.com/edwardyehuang)
# ================================================================

import tensorflow as tf
import os

from distutils.version import LooseVersion
from platform import uname

from iseg.utils.keras3_utils import is_keras3, is_keras2_15

def get_tpu_strategy(name=None):

    if LooseVersion(tf.version.VERSION) < LooseVersion("2.16.0"):
        cluster_resolver = tf.distribute.cluster_resolver.TPUClusterResolver(name)
        tf.config.experimental_connect_to_cluster(cluster_resolver)
        tf.tpu.experimental.initialize_tpu_system(cluster_resolver)
    else:
        cluster_resolver = tf.distribute.cluster_resolver.TPUClusterResolver.connect(name)
        print("TPU Device:", cluster_resolver.master())

    if LooseVersion(tf.version.VERSION) < LooseVersion("2.4.0"):
        strategy = tf.distribute.experimental.TPUStrategy(cluster_resolver)
    else:
        strategy = tf.distribute.TPUStrategy(cluster_resolver)

    return strategy


def shutdown_tpu_system (strategy):

    cluster_resolver = strategy.cluster_resolver
    tf.tpu.experimental.shutdown_tpu_system(cluster_resolver)


def get_cpu_strategy():
    return tf.distribute.OneDeviceStrategy("/cpu:0")


def get_distribution_strategy(
    gpu_memory_growth=True, 
    cuda_visible_devices=None, 
    use_tpu=False, 
    tpu_name=None
):

    if use_tpu:
        if tpu_name == "colab":
            tpu_name = None

        return get_tpu_strategy(tpu_name)
    
    dist_devices = None

    if cuda_visible_devices is not None:
        dist_devices = [cuda_visible_devices]

    set_gpu_memory_growth(gpu_memory_growth)

    cross_device_ops = None

    if os.name == "nt" or "microsoft-standard" in uname().release:
        cross_device_ops = tf.distribute.HierarchicalCopyAllReduce()
    
    if LooseVersion(tf.version.VERSION) >= LooseVersion("2.19.0"):
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