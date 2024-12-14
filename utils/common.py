# ====================================================================
# MIT License
# Copyright (c) 2024 edwardyehuang (https://github.com/edwardyehuang)
# ====================================================================

import iseg.static_strings as ss
import os
import random

import numpy as np
import tensorflow as tf

from iseg.utils.distribution_utils import list_gpus


DEFAULT_IMAGE_RESIZE_METHOD = "bilinear"
DEFAULT_ALIGN_CORNERS = False


def set_random_seed(seed=0):

    print('Use the random seed "{}"'.format(seed))
    tf.random.set_seed(seed)
    tf.keras.utils.set_random_seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)


def enable_mixed_precision(use_tpu=False):
    
    if use_tpu:
        tf.keras.mixed_precision.set_global_policy("mixed_bfloat16")
    else:

        gpus = list_gpus()

        support_bfloat16 = False

        if gpus:

            support_bfloat16 = True

            try:
                for gpu in gpus:
                    details = tf.config.experimental.get_device_details(gpu)
                    compute_capability = details["compute_capability"]

                    support_bfloat16 = support_bfloat16 and compute_capability[0] >= 8

            except RuntimeError as e:
                print(e)

        if support_bfloat16:
            print("GPU supports mixed_bfloat16 !")
            tf.keras.mixed_precision.set_global_policy("mixed_bfloat16")
        else:
            print("GPU does not support mixed_bfloat16, use mixed_float16 instead !")
            tf.keras.mixed_precision.set_global_policy("mixed_float16")


def get_tensor_shape(x, return_list=False):

    shapes = list(x.shape)
    dynamic_shapes = tf.shape(x)

    for i in range(len(shapes)):
        if shapes[i] is None:
            shapes[i] = dynamic_shapes[i]

    if return_list:
        return shapes

    return tuple(shapes)

def isinstance_all (inputs, t):

    for x in inputs:
        if not isinstance(x, t):
            return False
        
    return True


def smart_where(cond, x, y, name=None):
    if name is None:
        name = "smart_where"

    if isinstance(cond, bool):
        if cond:
            return x
        else:
            return y

    return tf.where(cond, x, y, name=name)


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

    # print(f"align_corners = {align_corners}")
    if align_corners:
        x = tf.compat.v1.image.resize(images, size, method=method, align_corners=align_corners, name=name)
    else:
        x = tf.image.resize(images, size, method=method, name=name)
        
    x = tf.cast(x, images.dtype)

    return x


def get_scaled_size_v2 (size, scale_rate=1.0):

    size_h = size[0]
    size_w = size[1]

    pad_h = tf.math.floormod(size_h, 2)
    pad_w = tf.math.floormod(size_w, 2)

    size_h -= pad_h
    size_w -= pad_w

    target_size_h = tf.cast(scale_rate * tf.cast(size_h, tf.float32), tf.int32)
    target_size_w = tf.cast(scale_rate * tf.cast(size_w, tf.float32), tf.int32)

    target_size_h += pad_h
    target_size_w += pad_w

    sizes = [target_size_h, target_size_w]

    return sizes


def get_scaled_size(inputs, scale_rate, pad_mode=0):

    inputs_size = get_tensor_shape(inputs, return_list=True)[1:3]

    if pad_mode == 0:
        return get_scaled_size_v2(inputs_size, scale_rate)
    elif pad_mode == 1:
        pad_size = 1
    else:
        raise ValueError(f"Not supported pad_mode = {pad_mode}")
    
    inputs_size_h = inputs_size[0]
    inputs_size_w = inputs_size[1]

    if isinstance_all([inputs_size_h, inputs_size_w, scale_rate], int):
        target_size_h = int(scale_rate * inputs_size_h)
        target_size_w = int(scale_rate * inputs_size_w)
    else:
        target_size_h = tf.cast(scale_rate * tf.cast(inputs_size_h, tf.float32), tf.int32)
        target_size_w = tf.cast(scale_rate * tf.cast(inputs_size_w, tf.float32), tf.int32)

    cond_h = (target_size_h % 2 == 0) & (inputs_size_h % 2 != 0)
    cond_w = (target_size_w % 2 == 0) & (inputs_size_w % 2 != 0)

    target_size_h = smart_where(cond_h, target_size_h + pad_size, target_size_h)
    target_size_w = smart_where(cond_w, target_size_w + pad_size, target_size_w)

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


def resample_absolute_position_embedding (
    position_embedding,
    target_size,
    source_size=None,
    num_prefix_tokens=1,
    method="bicubic",
    antialias=False,
):
    with tf.name_scope("resample_absolute_position_embedding"):
    
        batch_size, num_pos_tokens, channels = get_tensor_shape(position_embedding)

        num_new_tokens = target_size[0] * target_size[1] + num_prefix_tokens

        # if num_new_tokens == num_pos_tokens and target_size[0] == target_size[1]:
        # return position_embedding
        
        if num_prefix_tokens:
            num_spatial_pos_tokens = num_pos_tokens - num_prefix_tokens
            spatial_pos_embedding = position_embedding[:, num_prefix_tokens:] # [N, num_prefix_tokens + HW, C] -> [N, HW, C]
            prefix_pos_embedding = position_embedding[:, :num_prefix_tokens]
        else:
            num_spatial_pos_tokens = num_pos_tokens
            spatial_pos_embedding = position_embedding
            prefix_pos_embedding = None
        
        if source_size is None:
            hw = tf.sqrt(tf.cast(num_spatial_pos_tokens, tf.float32))
            hw = tf.cast(hw, tf.int32)
            source_size = hw, hw

        orginal_dtype = position_embedding.dtype

        spatial_pos_embedding = tf.reshape(
            spatial_pos_embedding, 
            [batch_size, source_size[0], source_size[1], channels]
        )

        spatial_pos_embedding = tf.image.resize(
            spatial_pos_embedding,
            target_size,
            method=method,
            antialias=antialias,
        )

        output_embedding = tf.reshape(
            spatial_pos_embedding,
            [batch_size, -1, channels]
        )

        output_embedding = tf.cast(output_embedding, orginal_dtype)

        if prefix_pos_embedding is not None:
            output_embedding = tf.concat([prefix_pos_embedding, output_embedding], axis=1)
        
        tf.assert_equal(tf.shape(output_embedding)[1], num_new_tokens)

        return output_embedding