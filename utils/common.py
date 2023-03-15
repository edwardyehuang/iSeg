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
DEFAULT_ALIGN_CORNERS = True


def set_random_seed(seed=0):

    print('Use the random seed "{}"'.format(seed))
    tf.random.set_seed(seed)
    tf.keras.utils.set_random_seed(seed)
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
    
    align_corners = DEFAULT_ALIGN_CORNERS

    print(f"align_corners = {align_corners}")

    x = tf.compat.v1.image.resize(images, size, method=method, align_corners=align_corners, name=name)
    x = tf.cast(x, images.dtype)

    return x


def get_scaled_size(inputs, scale_rate, pad_mode=0):

    if pad_mode == 0:
        pad_size = -1
    elif pad_mode == 1:
        pad_size = 1
    else:
        raise ValueError(f"Not supported pad_mode = {pad_mode}")

    inputs_size = tf.shape(inputs)[1:3]

    inputs_size_h = inputs_size[0]
    inputs_size_w = inputs_size[1]

    target_size_h = tf.cast(scale_rate * tf.cast(inputs_size_h, tf.float32), tf.int32)
    target_size_w = tf.cast(scale_rate * tf.cast(inputs_size_w, tf.float32), tf.int32)

    target_size_h = tf.where((target_size_h % 2 == 0) & (inputs_size_h % 2 != 0), target_size_h + pad_size, target_size_h)
    target_size_w = tf.where((target_size_w % 2 == 0) & (inputs_size_w % 2 != 0), target_size_w + pad_size, target_size_w)

    sizes = [target_size_h, target_size_w]

    return sizes


def down_size_image_by_scale(images, scale, method=None, name=None):

    image_shape = tf.shape(images)
    image_height = image_shape[-3]
    image_width = image_shape[-2]

    is_height_odd = tf.where(image_height % scale > 0, 1, 0)
    is_width_odd = tf.where(image_width % scale > 0, 1, 0)

    target_height = image_height // scale + is_height_odd
    target_width = image_width // scale + is_width_odd

    return resize_image(images=images, size=(target_height, target_width), method=method, name=name)