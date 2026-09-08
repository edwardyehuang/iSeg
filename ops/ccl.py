# ===================================================================
# MIT License
# Copyright (c) 2023 edwardyehuang (https://github.com/edwardyehuang)
# ===================================================================

# Connected components labeling (CCL), XLA/TPU-compatible.
#
# This file hosts two implementations with different trade-offs:
#
# 1. `label_components` (the original): exact flood fill over binary images.
#    Its outer while-loop scans every pixel (H*W sequential iterations of
#    scalar ops), which is latency-bound — measured ~16 s for a 337x500
#    image on an L20. Fine for occasional inference-time post-processing
#    (its original use), too slow for training input pipelines.
#
# 2. `label_segments_by_propagation`: vectorized min-label propagation over
#    multi-class semantic maps. Every iteration is a handful of elementwise
#    ops on [B, H, W] tensors, so it parallelizes across classes AND batch
#    elements (~0.13 s per 512x512 image, ~10 ms per image when batched).
#    Reaching the iteration cap only SPLITS a component into several ids —
#    it never merges different components — which downstream
#    segment-assignment consumers tolerate.

import tensorflow as tf

# The single source of 4-neighborhood offsets for both implementations.
NEIGHBOR_OFFSETS = [
    [-1, 0],
    [0, -1],
    [0, 1],
    [1, 0],
]

NEIGHBORS_COORDS = tf.constant(NEIGHBOR_OFFSETS, tf.int32)
NEIGHBORS_COORDS_V = NEIGHBORS_COORDS[:, 0]
NEIGHBORS_COORDS_U = NEIGHBORS_COORDS[:, 1]


# =====================================================================
# 1. Exact flood-fill CCL over binary images (original implementation)
# =====================================================================


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


# =====================================================================
# 2. Fast multi-class CCL via min-label propagation
# =====================================================================
#
# Every foreground pixel starts with its own flat index; each iteration
# takes the same-class 4-neighbor minimum, so a component converges to its
# smallest index. Convergence needs as many iterations as the longest
# same-class geodesic path (Manhattan diameter for blob-like segments).
#
# Segment ids are compacted to 1..K via root ordering (roots are pixels
# whose propagated label equals their own flat index; a root's compact id
# is its rank among roots, obtained with a cumsum — no dynamic-shape
# tf.unique).


def label_segments_by_propagation(
    class_map, ignore_label, max_iterations=-1, xla_bucket_size=256
):
    """Label connected same-class regions of semantic map(s).

    Args:
        class_map: `[H, W]` or `[B, H, W]` int32 raw class ids.
        ignore_label: Class id that receives no segments (background/void).
        max_iterations: Propagation cap; a non-positive value (default) uses
            height + width, a sound bound for blob-like segments.
        xla_bucket_size: The map is padded (with the ignore label) up to a
            multiple of this size before entering the XLA-compiled kernel
            and cropped back afterwards, so XLA compiles once per size
            bucket instead of once per distinct image size. Padding pixels
            carry the ignore label and never form segments.

    Returns:
        Segment ids with the same rank as the input: 0 = no segment,
        1..K = connected single-class components (per batch element).
    """

    class_map = tf.convert_to_tensor(class_map, tf.int32)

    input_rank = class_map.shape.rank
    if input_rank == 2:
        class_map = class_map[None]  # [1, H, W]
    elif input_rank != 3:
        raise ValueError(
            f"class_map must have rank 2 ([H, W]) or 3 ([B, H, W]), "
            f"got shape {class_map.shape}"
        )

    height = tf.shape(class_map)[1]
    width = tf.shape(class_map)[2]

    bucket = int(xla_bucket_size)
    padded_height = tf.math.floordiv(height + bucket - 1, bucket) * bucket
    padded_width = tf.math.floordiv(width + bucket - 1, bucket) * bucket

    padded = tf.pad(
        class_map,
        [[0, 0], [0, padded_height - height], [0, padded_width - width]],
        constant_values=ignore_label,
    )

    segment_ids = _propagation_xla(
        padded,
        tf.constant(ignore_label, tf.int32),
        tf.constant(max_iterations, tf.int32),
    )

    segment_ids = segment_ids[:, :height, :width]

    if input_rank == 2:
        segment_ids = segment_ids[0]

    return segment_ids


@tf.function(
    jit_compile=True,
    input_signature=[
        tf.TensorSpec([None, None, None], tf.int32),
        tf.TensorSpec([], tf.int32),
        tf.TensorSpec([], tf.int32),
    ],
    autograph=False,
)
def _propagation_xla(class_map, ignore_label, max_iterations):

    batch_size = tf.shape(class_map)[0]
    height = tf.shape(class_map)[1]
    width = tf.shape(class_map)[2]
    num_pixels = height * width

    # Non-positive cap = default bound (kept inside the graph so the python
    # wrapper stays shape-agnostic).
    max_iterations = tf.where(max_iterations <= 0, height + width, max_iterations)

    foreground = tf.not_equal(class_map, ignore_label)  # [B, H, W]

    # Initialize every foreground pixel with its own flat index; background
    # gets a sentinel larger than any index. Propagation takes the same-class
    # 4-neighbor minimum, so each component converges to its smallest index.
    flat_index = tf.reshape(tf.range(num_pixels, dtype=tf.int32), [1, height, width])
    big = num_pixels

    init_labels = tf.where(foreground, flat_index, big)

    def _cond(i, labels, changed):
        return (i < max_iterations) & changed

    def _body(i, labels, changed):

        # Pad with the sentinel (labels) / ignore class (classes) so border
        # neighbors never match and never win the minimum.
        padded_labels = tf.pad(
            labels, [[0, 0], [1, 1], [1, 1]], constant_values=big
        )
        padded_classes = tf.pad(
            class_map, [[0, 0], [1, 1], [1, 1]], constant_values=ignore_label
        )

        neighbor_min = tf.fill([batch_size, height, width], big)

        for di, dj in NEIGHBOR_OFFSETS:
            shifted_labels = padded_labels[
                :, 1 + di : 1 + di + height, 1 + dj : 1 + dj + width
            ]
            shifted_classes = padded_classes[
                :, 1 + di : 1 + di + height, 1 + dj : 1 + dj + width
            ]
            same_class = tf.equal(shifted_classes, class_map)
            neighbor_min = tf.minimum(
                neighbor_min, tf.where(same_class, shifted_labels, big)
            )

        new_labels = tf.minimum(labels, neighbor_min)
        changed = tf.reduce_any(tf.not_equal(new_labels, labels))

        return i + 1, new_labels, changed

    _, labels, _ = tf.while_loop(
        _cond,
        _body,
        [tf.constant(0, tf.int32), init_labels, tf.constant(True)],
        parallel_iterations=2000,
    )

    # Compact ids per batch element: a component's final label is the flat
    # index of its smallest pixel, which is a root (label == own index).
    # Ranking roots by scan order via a per-row cumsum gives static-shape
    # compact ids.
    is_root = foreground & tf.equal(labels, flat_index)  # [B, H, W]
    root_order = tf.cumsum(
        tf.reshape(tf.cast(is_root, tf.int32), [batch_size, num_pixels]), axis=1
    )

    gather_index = tf.minimum(
        tf.reshape(labels, [batch_size, num_pixels]), num_pixels - 1
    )
    compact = tf.gather(root_order, gather_index, batch_dims=1)

    segment_ids = tf.where(
        tf.reshape(foreground, [batch_size, num_pixels]),
        compact,
        tf.zeros_like(compact),
    )

    return tf.reshape(segment_ids, [batch_size, height, width])
