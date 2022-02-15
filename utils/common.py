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


def set_random_seed(seed=0):

    print('Use the random seed "{}"'.format(seed))
    tf.random.set_seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)


def enable_mixed_precision(use_tpu=False):

    if LooseVersion(tf.version.VERSION) < LooseVersion("2.4.0"):
        if use_tpu:
            tf.keras.mixed_precision.experimental.set_policy("mixed_bfloat16")
        else:
            tf.keras.mixed_precision.experimental.set_policy("mixed_float16")
    else:
        if use_tpu:
            tf.keras.mixed_precision.set_global_policy("mixed_bfloat16")
        else:
            tf.keras.mixed_precision.set_global_policy("mixed_float16")


def resize_image(images, size, method=None, name=None):

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

    x = tf.compat.v1.image.resize(images, size, method=method, align_corners=True, name=name)
    x = tf.cast(x, images.dtype)

    return x


def down_size_image_by_scale(images, scale, method=None, name=None):

    image_shape = tf.shape(images)
    image_height = image_shape[-3]
    image_width = image_shape[-2]

    is_height_odd = tf.where(image_height % scale > 0, 1, 0)
    is_width_odd = tf.where(image_width % scale > 0, 1, 0)

    target_height = image_height // scale + is_height_odd
    target_width = image_width // scale + is_width_odd

    return resize_image(images=images, size=(target_height, target_width), method=method, name=name)