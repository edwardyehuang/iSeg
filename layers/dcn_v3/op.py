# ===================================================================
# MIT License
# Copyright (c) 2024 edwardyehuang (https://github.com/edwardyehuang)
# ===================================================================

import tensorflow as tf

from iseg.layers.dcn_v3.utils import get_reference_points, generate_dilation_grids, bilinear_sampler
from iseg.layers.model_builder import get_tensor_shape_v2


def dcnv3_op (
    x, 
    offset, 
    mask, 
    kernel_size, 
    strides, 
    padding, 
    dilation_rate, 
    groups, 
    group_channels, 
    offset_scale
):
    
    if not isinstance(padding, str):
        raise TypeError("padding must be a string in 'SAME' or 'VALID'")
    
    padding = padding.upper()
    
    if padding == "SAME":
        padding = (kernel_size[0] // 2, kernel_size[1] // 2)
    elif padding == "VALID":
        padding = (0, 0)
    else:
        raise ValueError("padding must be 'SAME' or 'VALID'")
    
    pad_h, pad_w = padding
    kernel_h, kernel_w = kernel_size
    dilation_h, dilation_w = dilation_rate
    stride_h, stride_w = strides
    
    x = tf.pad(x, [[0, 0], [pad_h, pad_h], [pad_w, pad_w], [0, 0]])

    input_shape = get_tensor_shape_v2(x)

    batch_size, height_in, width_in, channels = input_shape
    _, height_out, width_out, _ = get_tensor_shape_v2(offset)

    ref = get_reference_points(
        spatial_shapes=input_shape,
        kernel_h=kernel_h,
        kernel_w=kernel_w,
        dilation_h=dilation_h,
        dilation_w=dilation_w,
        stride_h=stride_h,
        stride_w=stride_w,
    )

    grid = generate_dilation_grids(
        spatial_shapes=input_shape,
        kernel_h=kernel_h,
        kernel_w=kernel_w,
        dilation_h=dilation_h,
        dilation_w=dilation_w,
        group=groups,
    )

    spatial_norm = tf.constant([width_in, height_in], dtype=tf.float32)
    spatial_norm = tf.reshape(spatial_norm, [1, 1, 1, 2])
    spatial_norm = tf.tile(spatial_norm, [1, 1, 1, groups * kernel_h * kernel_w])

    sampling_locations = ref + grid * offset_scale
    sampling_locations = tf.tile(sampling_locations, [batch_size, 1, 1, 1, 1]) # CHECK: is this correct?
    sampling_locations = tf.reshape(sampling_locations, [batch_size, height_out, width_out, groups * kernel_h * kernel_w * 2])

    sampling_locations += offset * offset_scale / spatial_norm

    P_ = kernel_h * kernel_w
    sampling_grids = 2 * sampling_locations - 1

    x = tf.reshape(x, [batch_size, height_in * width_in, groups, group_channels])
    x = tf.transpose(x, [0, 2, 1, 3])
    x = tf.reshape(x, [batch_size * groups, height_in, width_in, group_channels])

    sampling_grids = tf.reshape(sampling_grids, [batch_size, height_out * width_out, groups, P_, 2])
    sampling_grids = tf.transpose(sampling_grids, [0, 2, 1, 3, 4])
    sampling_grids = tf.reshape(sampling_grids, [batch_size * groups, height_out * width_out, P_, 2])

    sample_inputs = bilinear_sampler(x, sampling_grids, dtype=x.dtype)

    mask = tf.reshape(mask, [batch_size, height_out * width_out, groups, P_])
    mask = tf.transpose(mask, [0, 2, 1, 3])
    mask = tf.reshape(mask, [batch_size * groups, height_out * width_out, P_, 1])

    output = sample_inputs * mask
    output = tf.reduce_sum(output, axis=-2) # [batch_size * groups, height_out * width_out, group_channels]
    output = tf.reshape(output, [batch_size, groups, height_out, width_out, group_channels])
    output = tf.transpose(output, [0, 2, 3, 1, 4])
    output = tf.reshape(output, [batch_size, height_out, width_out, groups * group_channels])

    return output

