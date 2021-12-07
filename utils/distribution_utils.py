# ================================================================
# MIT License
# Copyright (c) 2021 edwardyehuang (https://github.com/edwardyehuang)
# ================================================================

import tensorflow as tf
import os

from distutils.version import LooseVersion

def get_tpu_strategy(name = None):

    cluster_resolver = tf.distribute.cluster_resolver.TPUClusterResolver(name)
    tf.config.experimental_connect_to_cluster(cluster_resolver)
    tf.tpu.experimental.initialize_tpu_system(cluster_resolver)

    if LooseVersion(tf.version.VERSION) < LooseVersion("2.4.0"):
        strategy = tf.distribute.experimental.TPUStrategy(cluster_resolver)
    else:
        strategy = tf.distribute.TPUStrategy(cluster_resolver)

    return strategy


def get_cpu_strategy():
    return tf.distribute.OneDeviceStrategy("/cpu:0")


def get_distribution_strategy (gpu_memory_growth = True,
                               cuda_visible_devices = None, 
                               use_tpu = False, 
                               tpu_name = None):

    if use_tpu:
        if tpu_name == "colab":
            tpu_name = None
            
        return get_tpu_strategy(tpu_name)

    if cuda_visible_devices is not None:
        os.environ["CUDA_VISIBLE_DEVICES"] = cuda_visible_devices

    set_gpu_memory_growth(gpu_memory_growth)

    gpu_counts = get_gpu_counts()

    cross_device_ops = None

    if os.name == "nt":
        cross_device_ops = tf.distribute.HierarchicalCopyAllReduce()

    if gpu_counts == 1:
        # strategy = tf.distribute.OneDeviceStrategy("/gpu:0")
        strategy = tf.distribute.MirroredStrategy()
    else:
        # issue 41539 may be fixed: https://github.com/tensorflow/tensorflow/issues/41539
        strategy = tf.distribute.MirroredStrategy(cross_device_ops = cross_device_ops)
        # strategy = tf.distribute.experimental.MultiWorkerMirroredStrategy(communication= tf.distribute.experimental.CollectiveCommunication.RING)


    return strategy


def set_gpu_memory_growth (growth = False):

    gpus = tf.config.experimental.list_physical_devices('GPU')

    if gpus:
        try:
            for gpu in gpus:
                tf.config.experimental.set_memory_growth(gpu, growth)

        except RuntimeError as e:
            print(e)



def get_gpu_counts():

    gpus = tf.config.experimental.list_physical_devices('GPU')

    if gpus:
        return len(gpus)
    else:
        return 0