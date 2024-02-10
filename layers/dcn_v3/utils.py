# ===================================================================
# MIT License
# Copyright (c) 2024 edwardyehuang (https://github.com/edwardyehuang)
# ===================================================================

import tensorflow as tf

from iseg.utils import get_tensor_shape

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

    return ref


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

    return grid


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
    - out: interpolated images according to grids.
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

    deltas = tf.stack([wa, wb, wc, wd], axis=1) # [kh*kw, 4, N, H*W]
    deltas = tf.expand_dims(deltas, axis=-1) # [kh*kw, 4, N, H*W, 1]

    all_x = tf.stack([x0, x0, x1, x1], axis=1) # [kh*kw, 4, N, H*W]
    all_y = tf.stack([y0, y1, y0, y1], axis=1) # [kh*kw, 4, N, H*W]

    grid_shapes = get_tensor_shape(all_x)

    kernel_size = grid_shapes[0]
    batch_size = grid_shapes[2]
    num_points = grid_shapes[3]

    deltas = tf.reshape(deltas, [kernel_size, 4, batch_size * num_points, 1]) # [kh*kw, 4, N*H*W, 1]
    mask = tf.reshape(mask, [kernel_size, batch_size * num_points, 1]) # [kh*kw, N*H*W, 1]

    y = tf.zeros([batch_size * num_points, img.shape[-1]], dtype=dtype)

    batch_idx = tf.range(0, batch_size) # [N]
    batch_idx = tf.reshape(batch_idx, (1, batch_size, 1))
    b = tf.tile(batch_idx, (4, 1, num_points)) # [4, N, H*W]
    
    for i in range(kernel_size):

        indices = tf.raw_ops.Pack(values=[b, all_y[i], all_x[i]], axis=-1) # [4, N, H*W, 3]
        indices = tf.reshape(indices, [4, -1, 3]) # [4, N *H*W, 3]

        pixel_values = tf.raw_ops.GatherNd(params=img, indices=indices)
        pixel_values = tf.raw_ops.Mul(x=pixel_values, y=deltas[i]) # [4, N*H*W, 3]
        pixel_values = tf.reduce_sum(pixel_values, axis=0) # [N*H*W, 3]
    
        pixel_values = tf.raw_ops.Mul(x=pixel_values, y=mask[i])

        y += pixel_values

    y = tf.reshape(y, [batch_size, num_points, img.shape[-1]]) # [N, H*W, C]

    return y