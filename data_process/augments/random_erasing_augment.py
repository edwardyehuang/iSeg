# ================================================================
# MIT License
# Copyright (c) 2021 edwardyehuang (https://github.com/edwardyehuang)
# ================================================================

import tensorflow as tf

from iseg.data_process.augments.data_augment_base import DataAugmentationBase, random_execute_helper


class RandomErasingAugment(DataAugmentationBase):
    def __init__(
        self, 
        prob=0.25, 
        min_area_size=0,
        max_area_size=0.25,
        min_area_count=1,
        max_area_count=3,
        ignore_label=255,
        name=None,
    ):

        super().__init__(name=name)

        self.prob = prob

        min_area_size = float(min_area_size)
        max_area_size = float(max_area_size)

        assert min_area_size >= 0 and min_area_size <= 1, "min_area should be in [0, 1]"
        assert max_area_size >= 0 and max_area_size <= 1, "max_area should be in [0, 1]"
        assert min_area_size <= max_area_size, "min_area should be less than or equal to max_area"

        self.min_area_size = min_area_size
        self.max_area_size = max_area_size

        self.min_area_count = min_area_count
        self.max_area_count = max_area_count

        self.ignore_label = ignore_label


    def call(self, image, label):

        return random_execute_helper(
            self.prob,
            lambda: self._execute_branch(image, label),
            lambda: (image, label),
        )

    
    def _execute_branch(self, image, label):

        image_shape = tf.shape(image)
        height = image_shape[0]
        width = image_shape[1]

        height_float = tf.cast(height, tf.float32)
        width_float = tf.cast(width, tf.float32)

        min_area_height = tf.cast(height_float * self.min_area_size, tf.int32)
        max_area_height = tf.cast(height_float * self.max_area_size, tf.int32)

        min_area_width = tf.cast(width_float * self.min_area_size, tf.int32)
        max_area_width = tf.cast(width_float * self.max_area_size, tf.int32)

        num_area = tf.random.uniform([], minval=self.min_area_count, maxval=self.max_area_count, dtype=tf.int32)
        
        def inner_loop (_i, _image, _label):

            area_height = tf.random.uniform([], minval=min_area_height, maxval=max_area_height, dtype=tf.int32)
            area_width = tf.random.uniform([], minval=min_area_width, maxval=max_area_width, dtype=tf.int32)

            area_height = tf.clip_by_value(area_height, 1, height)
            area_width = tf.clip_by_value(area_width, 1, width)

            area_y_max = height - area_height
            area_x_max = width - area_width

            area_y = tf.random.uniform([], minval=0, maxval=area_y_max, dtype=tf.int32)
            area_x = tf.random.uniform([], minval=0, maxval=area_x_max, dtype=tf.int32)

            area_mask = tf.ones([area_height, area_width], dtype=tf.int32)
            area_mask = tf.pad(area_mask, [[area_y, height - area_y - area_height], [area_x, width - area_x - area_width]])
            area_mask = tf.cast(area_mask, tf.bool)
            area_mask = tf.expand_dims(area_mask, axis=-1)

            _image = tf.where(area_mask, 0.0, _image)
            _label = tf.where(area_mask, self.ignore_label, _label)

            return _i + 1, _image, _label
        
        _, image, label = tf.while_loop(
            lambda _i, _image, _label: _i < num_area,
            inner_loop,
            [0, image, label],
            maximum_iterations=self.max_area_count,
        )
        
        return image, label