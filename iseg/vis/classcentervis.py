# ================================================================
# MIT License
# Copyright (c) 2021 edwardyehuang (https://github.com/edwardyehuang)
# ================================================================

import tensorflow as tf


def get_attention (query):

    channels = query.shape[-1]
    key = tf.transpose(query, [0, 2 ,1]) # [N, C, HW]
    
    energy = tf.matmul(query, key) # [N, HW, HW]
    energy = energy / tf.sqrt(tf.cast(channels, query.dtype))

    attention = tf.nn.softmax(energy)

    return attention


def get_label_mask (label_query):

    label_key = tf.transpose(label_query, [0, 2, 1]) # [N, 1, HW]
    label_mask = label_query == label_key # [N, HW, HW]
    return label_mask


def get_class_center_2d_clustering (feature_map, label):

    # feature_map # [N, H, W, C]
    # label # [N, H, W] or [N, H, W, 1]

    feature_shape = tf.shape(feature_map)
    batch_size = feature_shape[0]
    height = feature_shape[1]
    width = feature_shape[2]
    channels = feature_map.shape[-1]

    if len(label.shape) == 3:
        label = tf.expand_dims(label, axis = -1)
    
    assert len(label.shape) == 4, f"Dims of label must be 3 or 4, found {len(label.shape)}"

    label = tf.cast(label, tf.int32)
    label = tf.compat.v1.image.resize(label, size = (height, width), method = "nearest", align_corners = True)

    feature_map = tf.reshape(feature_map, [batch_size, height * width, channels])
    label = tf.reshape(label, [batch_size, height, * width, 1])

    attention = get_attention(feature_map) # [N, HW, HW]
    label_mask = get_label_mask(label) # [N, HW, HW]

    









