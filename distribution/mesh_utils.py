# Mesh imp

import tensorflow as tf
import keras

from iseg.distribution.distribution_utils import set_gpu_memory_growth, list_gpus

dtensor = tf.experimental.dtensor


def get_mesh_layout_map(
    gpu_memory_growth=True, 
    cuda_visible_devices=None, 
    use_tpu=False, 
    tpu_name=None,
    use_one_device_strategy=False,
):
    set_gpu_memory_growth(gpu_memory_growth)

    devices = get_compute_device_list(use_tpu=use_tpu)
    
    mesh = dtensor.create_mesh([("batch", len(devices))], devices=devices)

    layout_map = keras.distribution.LayoutMap(mesh)

    return layout_map



def get_compute_device_list (use_tpu=False):

    if use_tpu:
        return tf.config.list_physical_devices("TPU")
    else:
        gpus = list_gpus()

        if len(gpus) > 0:
            return gpus

    return tf.config.list_physical_devices("CPU")