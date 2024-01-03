# ===================================================================
# MIT License
# Copyright (c) 2024 edwardyehuang (https://github.com/edwardyehuang)
# ===================================================================

import tensorflow as tf

from iseg.layers.model_builder import get_tensor_shape_v2

@tf.function(
    jit_compile=True,
    autograph=False,
)
def get_reference_points (
    spatial_shapes,
    kernel_h,
    kernel_w,
    dilation_h,
    dilation_w,
    stride_h=1,
    stride_w=1,
    dtype=tf.float32
):
    _, H_, W_, _ = spatial_shapes
    
    H_out = (H_ - (dilation_h * (kernel_h - 1) + 1)) // stride_h + 1
    W_out = (W_ - (dilation_w * (kernel_w - 1) + 1)) // stride_w + 1

    y_start = (dilation_h * (kernel_h - 1)) // 2 + 0.5
    y_end = (dilation_h * (kernel_h - 1)) // 2 + 0.5 + tf.cast((H_out - 1) * stride_h, dtype=tf.float32)

    x_start =  (dilation_w * (kernel_w - 1)) // 2 + 0.5
    x_end =  (dilation_w * (kernel_w - 1)) // 2 + 0.5 + tf.cast((W_out - 1) * stride_w, dtype=tf.float32)

    y_linespace = tf.linspace(y_start, y_end, H_out)
    x_linespace = tf.linspace(x_start, x_end, W_out)

    ref_y, ref_x = tf.meshgrid(y_linespace, x_linespace, indexing='ij')

    ref_y = tf.cast(ref_y, dtype=dtype)
    ref_x = tf.cast(ref_x, dtype=dtype)

    ref_y = tf.reshape(ref_y, [-1])
    ref_x = tf.reshape(ref_x, [-1])

    ref_y = tf.expand_dims(ref_y, axis=0)
    ref_x = tf.expand_dims(ref_x, axis=0)

    ref_y /= tf.cast(H_, dtype=dtype)
    ref_x /= tf.cast(W_, dtype=dtype)

    ref =  tf.stack([ref_y, ref_x], axis=-1)
    ref = tf.reshape(
        ref,
        [1, H_out, W_out, 1, 2]
    )

    return tf.stop_gradient(ref)


@tf.function(
    jit_compile=True,
    autograph=False,
)
def generate_dilation_grids(
    spatial_shapes, 
    kernel_h, 
    kernel_w, 
    dilation_h, 
    dilation_w, 
    group,
    dtype=tf.float32
):
    
    _, H_, W_, _ = spatial_shapes

    x, y = tf.meshgrid(
        tf.linspace(
            -((dilation_w * (kernel_w - 1)) // 2),
            -((dilation_w * (kernel_w - 1)) // 2) +
            (kernel_w - 1) * dilation_w, kernel_w,
        ),
        tf.linspace(
            -((dilation_h * (kernel_h - 1)) // 2),
            -((dilation_h * (kernel_h - 1)) // 2) +
            (kernel_h - 1) * dilation_h, kernel_h,
        ),
        indexing='ij',
    )

    x = tf.cast(x, dtype=dtype)
    y = tf.cast(y, dtype=dtype)

    x /= tf.cast(W_, dtype=dtype)
    y /= tf.cast(H_, dtype=dtype)

    grid = tf.stack([x, y], axis=-1)
    grid = tf.reshape(grid, [-1, 1, 2])
    grid = tf.tile(grid, [1, group, 1])
    grid = tf.transpose(grid, [1, 0, 2])
    grid = tf.reshape(grid, [1, 1, 1, group * kernel_h * kernel_w, 2])

    return tf.stop_gradient(grid)


@tf.function(
    jit_compile=True,
    autograph=False,
)
def _get_pixel_value(img, x, y):
    """
    Utility function to get pixel value for coordinate
    vectors x and y from a  4D tensor image.

    Input
    -----
    - img: tensor of shape (N, H, W, C)
    - x: flattened tensor of shape (4, kh*kw, N, groups, H*W,)
    - y: flattened tensor of shape (4, kh*kw, N, groups, H*W,)

    Returns
    -------
    - output: tensor of shape (N, H, W, C)
    """

    grid_shapes = get_tensor_shape_v2(x)

    kernel_size = grid_shapes[1]
    batch_size = grid_shapes[2]
    num_groups = grid_shapes[3]
    num_points = grid_shapes[4]

    batch_idx = tf.range(0, batch_size) # [N * groups]
    batch_idx = tf.reshape(batch_idx, (1, 1, batch_size, 1, 1))
    b = tf.tile(batch_idx, (4, kernel_size, 1, num_groups, num_points)) # [4, kh*kw, N, groups, H*W]

    indices = tf.stack([b, y, x], -1)

    return tf.raw_ops.GatherNd(params=img, indices=indices)


@tf.function(
    jit_compile=True,
    autograph=False,
)
def dcnv3_bilinear_sampler(img, grid, mask):
    """
    Performs bilinear sampling of the input images according to the
    normalized coordinates provided by the sampling grid. Note that
    the sampling is done identically for each channel of the input.

    To test if the function works properly, output image should be
    identical to input image when theta is initialized to identity
    transform.

    Input
    -----
    - img: batch of images in (N, H, W, C) layout.
    - grid: coords

    Returns
    -------
    - out: interpolated images according to grids. Same size as grid.
    """

    dtype = img.dtype

    H = tf.shape(img)[1]
    W = tf.shape(img)[2]
    
    max_y = tf.cast(H - 1, dtype=tf.int32)
    max_x = tf.cast(W - 1, dtype=tf.int32)

    # rescale x and y to [0, W-1/H-1]
    x, y = grid[:, ..., 0], grid[:, ..., 1]
    x = tf.cast(x, dtype=dtype)
    y = tf.cast(y, dtype=dtype)
    x = 0.5 * ((x + 1.0) * tf.cast(max_x-1, dtype=dtype))
    y = 0.5 * ((y + 1.0) * tf.cast(max_y-1, dtype=dtype))

    # grab 4 nearest corner points for each (x_i, y_i)
    x0 = tf.cast(tf.floor(x), dtype=tf.int32)
    x1 = x0 + 1
    y0 = tf.cast(tf.floor(y), dtype=tf.int32)
    y1 = y0 + 1

    # clip to range [0, H-1/W-1] to not violate img boundaries
    x0 = tf.clip_by_value(x0, 0, max_x)
    x1 = tf.clip_by_value(x1, 0, max_x)
    y0 = tf.clip_by_value(y0, 0, max_y)
    y1 = tf.clip_by_value(y1, 0, max_y)

    # recast as float for delta calculation
    x0f = tf.cast(x0, dtype=dtype)
    x1f = tf.cast(x1, dtype=dtype)
    y0f = tf.cast(y0, dtype=dtype)
    y1f = tf.cast(y1, dtype=dtype)

    dx0f = x - x0f
    dx1f = x1f - x
    dy0f = y - y0f
    dy1f = y1f - y

    # calculate deltas
    wa = dx1f * dy1f
    wb = dx1f * dy0f
    wc = dx0f * dy1f
    wd = dx0f * dy0f

    deltas = tf.stack([wa, wb, wc, wd], axis=0) # [4, kh*kw, N, groups, H*W]
    deltas = tf.expand_dims(deltas, axis=-1) # [4, kh*kw, N, groups, H*W, 1]

    all_x = tf.stack([x0, x0, x1, x1], axis=0) # [4, kh*kw, N, groups, H*W]
    all_y = tf.stack([y0, y1, y0, y1], axis=0) # [4, kh*kw, N, groups, H*W]

    grid_shapes = get_tensor_shape_v2(all_x)

    kernel_size = grid_shapes[1]
    batch_size = grid_shapes[2]
    num_groups = grid_shapes[3]
    num_points = grid_shapes[4]

    batch_idx = tf.range(0, batch_size) # [N * groups]
    batch_idx = tf.reshape(batch_idx, (1, 1, batch_size, 1, 1))
    b = tf.tile(batch_idx, (4, kernel_size, 1, num_groups, num_points)) # [4, kh*kw, N, groups, H*W]

    indices = tf.stack([b, all_y, all_x], -1)

    pixel_values = tf.raw_ops.GatherNd(params=img, indices=indices)
    pixel_values *= deltas

    y = tf.reduce_sum(pixel_values, axis=0) # [kh*kw, N, groups, H*W, C]

    y *= mask

    y = tf.reduce_sum(y, axis=0) # [N, groups, H*W, C]

    return y