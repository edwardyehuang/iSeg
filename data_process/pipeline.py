# ================================================================
# MIT License
# Copyright (c) 2021 edwardyehuang (https://github.com/edwardyehuang)
# ================================================================

import tensorflow as tf
from iseg.data_process.augments import *


class AugmentationsPipeLine(object):
    def __init__(self, target_height=None, target_width=None, augments=[], name=None) -> None:
        super().__init__()

        if name is None:
            name = type(self).__name__

        self.name = name
        self.augments = augments

        self.target_height = target_height if target_height > 0 else None
        self.target_width = target_width if target_width > 0 else None

    def post_process(self, image, label):

        image = tf.cast(image, tf.float32)

        has_target_size = self.target_height is not None and self.target_width is not None

        if has_target_size:
            image.set_shape([self.target_height, self.target_width, image.shape[-1]])

        if label is not None:
            label = tf.cast(label, tf.int32)

            if has_target_size:
                label.set_shape([self.target_height, self.target_width, label.shape[-1]])

            label = tf.squeeze(label)

        return image, label

    def process(self, *inputs):

        processed_arugments = []

        for augment in self.augments:

            try:
                inputs = augment(*inputs)

            except:
                print(f"Error : {augment.name}")

            processed_arugments.append(augment.name)

        print(f"Processed augments = {processed_arugments}")

        return self.post_process(*inputs)

    def __call__(self, ds):

        if ds is None:
            return ds

        return ds.map(self.process, num_parallel_calls=tf.data.experimental.AUTOTUNE)


class StandardAugmentationsPipeline(AugmentationsPipeLine):
    def __init__(
        self,
        training=False,
        mean_pixel=[127.5, 127.5, 127.5],
        ignore_label=255,
        min_resize_value=None,
        max_resize_value=None,
        crop_height=513,
        crop_width=513,
        eval_crop_height=None,
        eval_crop_width=None,
        prob_of_flip=0.5,
        min_scale_factor=0.5,
        max_scale_factor=2.0,
        scale_factor_step_size=0.1,
        random_brightness=False,
        photo_metric_distortions=False,
        name=None,
    ):

        if eval_crop_height is None:
            eval_crop_height = crop_height

        if eval_crop_width is None:
            eval_crop_width = crop_width

        if not training:
            crop_height = eval_crop_height
            crop_width = eval_crop_width

        super().__init__(target_height=crop_height, target_width=crop_width, name=name)

        augments = []

        if max_resize_value or min_resize_value:
            augments.append(ResizeAugment(min_resize_value, max_resize_value))

        if training:
            augments.append(RandomScaleAugment(min_scale_factor, max_scale_factor, scale_factor_step_size))

        pad_value = tf.reshape(mean_pixel, [1, 1, 3])

        if training:
            if photo_metric_distortions:
                augments.append(RandomContrastAugment(0.5, 1.5, execute_prob=0.5))
                # augments.append(RandomHueArgument())
                augments.append(RandomSaturationAugment(0.5, 1.5, execute_prob=0.5))

            if random_brightness:
                augments.append(RandomBrightnessAugment(execute_prob=0.5))

        augments.append(PadAugment(crop_height, crop_width, pad_value, ignore_label))

        if training:
            augments.append(RandomCropAugment(crop_height, crop_width))
            augments.append(RandomFlipAugment(prob_of_flip))

        self.augments = augments
