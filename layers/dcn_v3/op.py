# ===================================================================
# MIT License
# Copyright (c) 2024 edwardyehuang (https://github.com/edwardyehuang)
# ===================================================================

import tensorflow as tf

from iseg.layers.dcn_v3.utils import get_reference_points, generate_dilation_grids, dcnv3_bilinear_sampler
from iseg.layers.model_builder import get_tensor_shape_v2

@tf.function(
    jit_compile=True,
    autograph=False,
    reduce_retracing=False,
)
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
    offset_scale,
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
        dtype=x.dtype,
    )

    grid = generate_dilation_grids(
        spatial_shapes=input_shape,
        kernel_h=kernel_h,
        kernel_w=kernel_w,
        dilation_h=dilation_h,
        dilation_w=dilation_w,
        group=groups,
        dtype=x.dtype,
    )

    P_ = kernel_h * kernel_w

    spatial_norm = tf.stack([width_in, height_in], axis=0)
    spatial_norm = tf.reshape(spatial_norm, [1, 1, 1, 2])
    spatial_norm = tf.tile(spatial_norm, [1, 1, 1, groups * P_])
    spatial_norm = tf.cast(spatial_norm, dtype=x.dtype)

    sampling_locations = ref + grid * offset_scale
    sampling_locations = tf.reshape(sampling_locations, [1, height_out, width_out, groups * P_ * 2]) # [1, H, W, groups * kh * kw * 2]

    sampling_locations += offset * offset_scale / spatial_norm # [N, H, W, groups * kh * kw * 2]

    sampling_grids = 2 * sampling_locations - 1 # [N, H, W, groups * kh * kw * 2]

    x = tf.reshape(x, [batch_size, height_in, width_in, groups, group_channels])
    x = tf.transpose(x, [0, 3, 1, 2, 4])
    x = tf.reshape(x, [batch_size * groups, height_in, width_in, group_channels])

    sampling_grids = tf.reshape(sampling_grids, [batch_size, height_out * width_out, groups, P_, 2]) # [N, HW, groups, kh*hw, 2]
    sampling_grids = tf.transpose(sampling_grids, [3, 0, 2, 1, 4]) # [kh*kw, N, groups, HW, 2]
    sampling_grids = tf.reshape(sampling_grids, [P_, batch_size * groups, height_out * width_out, 2]) # [kh*kw, N * groups, HW, 2]

    mask = tf.reshape(mask, [batch_size, height_out * width_out, groups, P_]) # [N, HW, groups, kh*kw]
    mask = tf.transpose(mask, [3, 0, 2, 1]) # [kh*kw, N, groups, HW]
    mask = tf.reshape(mask, [P_, batch_size * groups, height_out * width_out, 1]) # [kh*kw, N * groups, HW, 1]

    # output = x

    output = dcnv3_bilinear_sampler(x, sampling_grids, mask) # [N * groups * HW, C]

    output = tf.reshape(output, [batch_size, groups, height_out, width_out, group_channels])
    output = tf.transpose(output, [0, 2, 3, 1, 4])
    output = tf.reshape(output, [batch_size, height_out, width_out, groups * group_channels])

    return output

