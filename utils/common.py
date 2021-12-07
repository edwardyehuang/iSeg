# ================================================================
# MIT License
# Copyright (c) 2021 edwardyehuang (https://github.com/edwardyehuang)
# ================================================================

import iseg.static_strings as ss
import os
import random

import numpy as np
import tensorflow as tf

from distutils.version import LooseVersion

DEFAULT_IMAGE_RESIZE_METHOD = "bilinear"

def set_random_seed(seed = 0):

    print("Use the random seed \"{}\"".format(seed))
    tf.random.set_seed(seed)
    os.environ['PYTHONHASHSEED']=str(seed)
    random.seed(seed)
    np.random.seed(seed)


def enable_mixed_precision(use_tpu = False):

    if LooseVersion(tf.version.VERSION) < LooseVersion("2.4.0"):
        if use_tpu:
            tf.keras.mixed_precision.experimental.set_policy('mixed_bfloat16')
        else:
            tf.keras.mixed_precision.experimental.set_policy('mixed_float16')
    else:
        if use_tpu:
            tf.keras.mixed_precision.set_global_policy('mixed_bfloat16')
        else:
            tf.keras.mixed_precision.set_global_policy('mixed_float16')


def resize_image (images, size, method = None, name = None):

    if method is None:
        method = DEFAULT_IMAGE_RESIZE_METHOD

    if isinstance(method, str):
        method = method.lower()

    if method == "bilinear" or method == tf.image.ResizeMethod.BILINEAR:
        method = tf.image.ResizeMethod.BILINEAR
    elif method == "nearest" or method == tf.image.ResizeMethod.NEAREST_NEIGHBOR:
        method = tf.image.ResizeMethod.NEAREST_NEIGHBOR
    elif method == "bicubic" or method == tf.image.ResizeMethod.BICUBIC:
        method = tf.image.ResizeMethod.BICUBIC
    else:
        raise ValueError("Not support")

    x = tf.compat.v1.image.resize(images, size, method = method, align_corners = True, name = name)
    x = tf.cast(x, images.dtype)

    return x


def down_size_image_by_scale (images, scale, method = None, name = None):

    image_shape = tf.shape(images)
    image_height = image_shape[-3]
    image_width = image_shape[-2]

    is_height_odd = tf.where(image_height % scale > 0, 1, 0)
    is_width_odd = tf.where(image_width % scale > 0, 1, 0)

    target_height = image_height // scale + is_height_odd
    target_width = image_width // scale + is_width_odd

    return resize_image(images = images,
                        size = (target_height, target_width),
                        method = method,
                        name = name)



def simple_load_image (image_path, min_size_unit = 1):

    image_tensor = tf.image.decode_jpeg(tf.io.read_file(image_path), channels = 3)
    image_tensor = tf.expand_dims(tf.cast(image_tensor, tf.float32), axis = 0)  # [1, H, W, 3]

    image_size = tf.shape(image_tensor)[1:3]

    pad_height = tf.cast(tf.math.ceil(image_size[0] / 32) * 32, tf.int32)
    pad_width = tf.cast(tf.math.ceil(image_size[1] / 32) * 32, tf.int32)

    pad_height = pad_height if pad_height % 2 != 0 else pad_height + 1
    pad_width = pad_height if pad_width % 2 != 0 else pad_width + 1

    from iseg.data_process.utils import pad_to_bounding_box, normalize_value_range

    pad_image_tensor = pad_to_bounding_box(image_tensor, 0, 0, pad_height, pad_width, pad_value = [127.5, 127.5, 127.5])
    pad_image_tensor = normalize_value_range(pad_image_tensor)

    return pad_image_tensor, image_size