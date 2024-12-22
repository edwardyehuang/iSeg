# ================================================================
# MIT License
# Copyright (c) 2021 edwardyehuang (https://github.com/edwardyehuang)
# ================================================================

import tensorflow as tf
import iseg.data_process.utils as dataprocess

from iseg.data_process.augments.data_augment_base import DataAugmentationBase
from iseg.utils.common import get_tensor_shape


class ResizeAugment(DataAugmentationBase):
    def __init__(self, max_resize_height, max_resize_width, name=None):
        super().__init__(name=name)

        self.max_resize_height = max_resize_height
        self.max_resize_width = max_resize_width


    
    def compuate_target_size (self, height, width):

        height_float = tf.cast(height, tf.float32)
        width_float = tf.cast(width, tf.float32)

        target_height = tf.minimum(self.max_resize_height, height)

        target_height_float = tf.cast(target_height, tf.float32)

        target_width = width_float * target_height_float / height_float
        target_width = tf.cast(target_width, tf.int32)

        target_width = tf.minimum(self.max_resize_width, target_width)
        target_width = tf.minimum(width, target_width)

        target_width_float = tf.cast(target_width, tf.float32)

        target_height = height_float * target_width_float / width_float
        target_height = tf.cast(target_height, tf.int32)

        return target_height, target_width


    def call(self, image, label):

        has_label = label is not None

        if has_label:
            label_rank = len(label.shape)
        
        height, width, _ = get_tensor_shape(image)

        if not has_label:
            label = tf.zeros([height, width, 1], dtype=image.dtype)
        elif label_rank == 2:
            label = tf.expand_dims(label, axis=-1) # [H, W] -> [H, W, 1]

        target_height, target_width = self.compuate_target_size(height, width)

        image = tf.image.resize(image, [target_height, target_width], method=tf.image.ResizeMethod.BILINEAR)
        label = tf.image.resize(label, [target_height, target_width], method=tf.image.ResizeMethod.NEAREST_NEIGHBOR)

        if not has_label:
            return image, None
        
        if label_rank == 2:
            label = tf.squeeze(label, axis=-1)

        return image, label