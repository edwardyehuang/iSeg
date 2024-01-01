# ===================================================================
# MIT License
# Copyright (c) 2024 edwardyehuang (https://github.com/edwardyehuang)
# ===================================================================

import tensorflow as tf

def get_reference_points (
    spatial_shapes,
    kernel_h,
    kernel_w,
    dilation_h,
    dilation_w,
    stride_h=1,
    stride_w=1,
):
    _, H_, W_, _ = spatial_shapes
    
    H_out = (H_ - (dilation_h * (kernel_h - 1) + 1)) // stride_h + 1
    W_out = (W_ - (dilation_w * (kernel_w - 1) + 1)) // stride_w + 1


    ref_y, ref_x = tf.meshgrid(
        tf.linspace(
            (dilation_h * (kernel_h - 1)) // 2 + 0.5,
            (dilation_h * (kernel_h - 1)) // 2 + 0.5 + (H_out - 1) * stride_h,
            H_out
        ),
        tf.linspace(
            (dilation_w * (kernel_w - 1)) // 2 + 0.5,
            (dilation_w * (kernel_w - 1)) // 2 + 0.5 + (W_out - 1) * stride_w,
            W_out,
        ),
        indexing='ij'
    )

    ref_y = tf.reshape(ref_y, [-1])
    ref_x = tf.reshape(ref_x, [-1])

    ref_y = tf.expand_dims(ref_y, axis=0)
    ref_x = tf.expand_dims(ref_x, axis=0)

    ref_y /= H_
    ref_x /= W_

    ref =  tf.stack([ref_y, ref_x], axis=-1)
    ref = tf.reshape(
        ref,
        [1, H_out, W_out, 1, 2]
    )

    return ref



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

    grid = tf.stack([x / W_, y / H_], axis=-1)
    grid = tf.reshape(grid, [-1, 1, 2])
    grid = tf.tile(grid, [1, group, 1])
    grid = tf.transpose(grid, [1, 0, 2])
    grid = tf.reshape(grid, [1, 1, 1, group * kernel_h * kernel_w, 2])

    return grid



def _get_pixel_value(img, x, y):
    """
    Utility function to get pixel value for coordinate
    vectors x and y from a  4D tensor image.

    Input
    -----
    - img: tensor of shape (N, H, W, C)
    - x: flattened tensor of shape (N*H*W,)
    - y: flattened tensor of shape (N*H*W,)

    Returns
    -------
    - output: tensor of shape (N, H, W, C)
    """
    shape = tf.shape(x)
    batch_size = shape[0]
    height = shape[1]
    width = shape[2]

    batch_idx = tf.range(0, batch_size)
    batch_idx = tf.reshape(batch_idx, (batch_size, 1, 1))
    b = tf.tile(batch_idx, (1, height, width))

    indices = tf.stack([b, y, x], 3)

    return tf.gather_nd(img, indices)

@tf.function(
    jit_compile=True,
    autograph=False,
)
def bilinear_sampler(img, grid, dtype=tf.float32):
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
    H = tf.shape(img)[1]
    W = tf.shape(img)[2]
    max_y = tf.cast(H - 1, dtype=tf.int32)
    max_x = tf.cast(W - 1, dtype=tf.int32)
    zero = tf.zeros([], dtype=tf.int32)

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
    x0 = tf.clip_by_value(x0, zero, max_x)
    x1 = tf.clip_by_value(x1, zero, max_x)
    y0 = tf.clip_by_value(y0, zero, max_y)
    y1 = tf.clip_by_value(y1, zero, max_y)

    # get pixel value at corner coords
    Ia = _get_pixel_value(img, x0, y0)
    Ib = _get_pixel_value(img, x0, y1)
    Ic = _get_pixel_value(img, x1, y0)
    Id = _get_pixel_value(img, x1, y1)

    # recast as float for delta calculation
    x0 = tf.cast(x0, dtype=dtype)
    x1 = tf.cast(x1, dtype=dtype)
    y0 = tf.cast(y0, dtype=dtype)
    y1 = tf.cast(y1, dtype=dtype)

    # calculate deltas
    wa = (x1-x) * (y1-y)
    wb = (x1-x) * (y-y0)
    wc = (x-x0) * (y1-y)
    wd = (x-x0) * (y-y0)

    # add dimension for addition
    wa = tf.expand_dims(wa, axis=-1)
    wb = tf.expand_dims(wb, axis=-1)
    wc = tf.expand_dims(wc, axis=-1)
    wd = tf.expand_dims(wd, axis=-1)

    # compute output
    out = tf.add_n([wa*Ia, wb*Ib, wc*Ic, wd*Id])

    return out