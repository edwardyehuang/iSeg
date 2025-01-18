# ================================================================
# MIT License
# Copyright (c) 2021 edwardyehuang (https://github.com/edwardyehuang)
# ================================================================

import tensorflow as tf
import iseg.data_process.utils as dataprocess

from iseg.data_process.augments.data_augment_base import DataAugmentationBase


class RandomScaleAugment(DataAugmentationBase):
    def __init__(
        self, 
        min_scale_factor=0.5, 
        max_scale_factor=2.0, 
        scale_factor_step_size=0.1,
        break_aspect_ratio=False,
        name=None
    ):

        super().__init__(name=name)

        self.min_scale_factor = min_scale_factor
        self.max_scale_factor = max_scale_factor
        self.scale_factor_step_size = scale_factor_step_size

        self.break_aspect_ratio = break_aspect_ratio

        print(f"RandomScaleAugment : break_aspect_ratio = {self.break_aspect_ratio}")

    def call(self, image, label):

        scale_h = dataprocess.get_random_scale(self.min_scale_factor, self.max_scale_factor, self.scale_factor_step_size)

        if self.break_aspect_ratio:
            scale_list = [scale_h - self.scale_factor_step_size, scale_h, scale_h + self.scale_factor_step_size]
            scale_list = tf.random.shuffle(scale_list)

            scale_h = scale_list[0]
            scale_w = scale_list[1]

            scale_h = tf.clip_by_value(scale_h, self.min_scale_factor, self.max_scale_factor)
            scale_w = tf.clip_by_value(scale_w, self.min_scale_factor, self.max_scale_factor)


        image, label = dataprocess.randomly_scale_image_and_label(image, label, scale_h, scale_w)
        image.set_shape([None, None, 3])

        return image, label