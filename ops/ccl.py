# ===================================================================
# MIT License
# Copyright (c) 2023 edwardyehuang (https://github.com/edwardyehuang)
# ===================================================================

# This implementation of connected components labeling supports TPU XLA.
# Which even faster than GPU connected components labeling (via custom OP) sometimes.

import tensorflow as tf

NEIGHBORS_COORDS = [
    [-1, 0],
    [0, -1],
    [0, 1],
    [1, 0],
]

NEIGHBORS_COORDS  = tf.constant(NEIGHBORS_COORDS, tf.int32)
NEIGHBORS_COORDS_V = NEIGHBORS_COORDS[:, 0]
NEIGHBORS_COORDS_U = NEIGHBORS_COORDS[:, 1]


@tf.function(
    jit_compile=True,
    input_signature=[tf.TensorSpec([None, None, None], tf.int32)],
    autograph=False,
)
def label_components(binary_img):

    label_img = tf.map_fn(
        _label_components, 
        binary_img,
        parallel_iterations=2000,
    )

    return label_img


@tf.function(
    jit_compile=True,
    input_signature=[
        tf.TensorSpec([None, None], tf.int32),
    ],
    autograph=False,
)
def _label_components(binary_img):

    binary_sum = tf.reduce_sum(binary_img)

    return tf.cond(
        pred=tf.greater(binary_sum, 1),
        true_fn=lambda: find_components(-binary_img, tf.constant(0, binary_img.dtype)),
        false_fn=lambda: binary_img,
    )


@tf.function(
    jit_compile=True,
    input_signature=[
        tf.TensorSpec([None, None], tf.int32),
        tf.TensorSpec((), tf.int32),
    ],
    autograph=False,
)
def find_components(label_img, label):

    height = tf.shape(label_img)[-2]
    width = tf.shape(label_img)[-1]

    max_depth = hw = tf.raw_ops.Mul(x=height, y=width)

    def cond (i, label_img__, label__):
        return tf.less(i, hw)

    def body(i, label_img__, label__):
        
        row = tf.raw_ops.FloorDiv(x=i, y=width)
        col = tf.math.floormod(i, width)

        c = tf.equal(label_img__[row, col], -1)

        label__ += tf.raw_ops.Cast(
            x=c, DstT=label_img__.dtype
        ) # bool -> cast -> 1 or 0
        
        label_img__ = tf.cond(
            pred=c,
            true_fn=lambda: search(label_img__, label__, i, max_depth),
            false_fn=lambda: label_img__,
        )

        return tf.raw_ops.AddV2(x=i, y=1), label_img__, label__
    

    _, label_img, label = tf.while_loop(
        cond, 
        body, 
        [0, label_img, label],
        parallel_iterations=2000,
        maximum_iterations=hw,
    )

    return label_img



@tf.function(
    jit_compile=True,
    input_signature=[
        tf.TensorSpec([None, None], tf.int32),
        tf.TensorSpec((), tf.int32),
        tf.TensorSpec((), tf.int32),
        tf.TensorSpec((), tf.int32),
    ],
    autograph=False,
)
def search(label_img, label, coord, max_depth):

    stack_tensor_indexs = tf.range(max_depth, dtype=tf.int32)
    stack_tensor_array = tf.fill([max_depth], coord)
    autual_size = tf.constant(1, tf.int32)
    

    def while_cond (label_img_inner, autual_size_inner, stack_tensor_array_inner):
        return tf.greater(autual_size_inner, 0)
    
    def while_body (label_img_inner, autual_size_inner, stack_tensor_array_inner):

        autual_size_inner -= 1
        coord = stack_tensor_array_inner[autual_size_inner]

        label_img_inner, new_target_list = search_neighbors(label_img_inner, label, coord)

        for new_target in tf.unstack(new_target_list, 4):

            stack_tensor_array_inner = tf.raw_ops.SelectV2(
                condition=tf.raw_ops.Equal(x=autual_size_inner, y=stack_tensor_indexs),
                t=new_target, 
                e=stack_tensor_array_inner
            )

            autual_size_inner += tf.raw_ops.Cast(
                x=tf.raw_ops.NotEqual(x=new_target, y=-1), 
                DstT=autual_size_inner.dtype
            )


        return label_img_inner, autual_size_inner, stack_tensor_array_inner
    

    label_img, _, stack_tensor_array = tf.compat.v1.while_loop(
        while_cond, 
        while_body, 
        [label_img, autual_size, stack_tensor_array],
        parallel_iterations=2000,
        maximum_iterations=max_depth,
    )


    return label_img


@tf.function(
    jit_compile=True,
    input_signature=[
        tf.TensorSpec([None, None], tf.int32),
        tf.TensorSpec((), tf.int32),
        tf.TensorSpec((), tf.int32),
    ],
    autograph=False,
)
def search_neighbors (label_img, label, coord):

    height = tf.shape(label_img, out_type=coord.dtype)[-2]
    width = tf.shape(label_img, out_type=coord.dtype)[-1]

    i = tf.raw_ops.FloorDiv(x=coord, y=width)
    j = tf.math.floormod(coord, width)

    label_img = tf.tensor_scatter_nd_update(label_img, [[i, j]], [label])

    n = tf.raw_ops.Cast(x=NEIGHBORS_COORDS_V, DstT=i.dtype)
    m = tf.raw_ops.Cast(x=NEIGHBORS_COORDS_U, DstT=j.dtype)

    target_row = tf.raw_ops.AddV2(x=i, y=n)
    target_col = tf.raw_ops.AddV2(x=j, y=m)

    target_row_c = tf.logical_and(tf.greater_equal(target_row, 0), tf.less(target_row, height))
    target_col_c = tf.logical_and(tf.greater_equal(target_col, 0), tf.less(target_col, width))

    target_row = tf.raw_ops.Mul(
        x=target_row, y=tf.raw_ops.Cast(x=target_row_c, DstT=label_img.dtype)
    )
    target_col = tf.raw_ops.Mul(
        x=target_col, y=tf.raw_ops.Cast(x=target_col_c, DstT=label_img.dtype)
    )

    indices = tf.raw_ops.Pack(values=[target_row, target_col], axis=-1) # [4, 2]
    value_c = tf.raw_ops.GatherNd(params=label_img, indices=indices) == -1

    c = value_c & (target_row_c & target_col_c)

    target_list = tf.raw_ops.AddV2(
        x=tf.raw_ops.Mul(x=target_row, y=width),
        y=target_col
    )

    target_list = tf.raw_ops.SelectV2(
        condition=c, 
        t=target_list, 
        e=-1
    )

    return label_img, target_list