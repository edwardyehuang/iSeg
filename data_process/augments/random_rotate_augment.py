# ================================================================
# MIT License
# Copyright (c) 2021 edwardyehuang (https://github.com/edwardyehuang)
# ================================================================

import numpy as np

import tensorflow as tf
import keras
import iseg.data_process.utils as dataprocess

from iseg.data_process.augments.data_augment_base import DataAugmentationBase

from keras.src.layers.preprocessing.image_preprocessing import get_rotation_matrix, transform

H_AXIS = -3
W_AXIS = -2


class RandomRotateAugment(DataAugmentationBase):
    def __init__(
        self, 
        prob_of_rotate=0.5,
        fill_constant_color=[0, 0, 0],
        ignore_label=255, 
        name=None
    ):

        super().__init__(name=name)

        self.prob_of_rotate = prob_of_rotate

        self.rotate_layer = keras.layers.RandomRotation()

        self._random_generator = keras.backend.RandomGenerator()

        self.fill_constant_color = fill_constant_color
        self.ignore_label = ignore_label


    def call(self, image, label):

        random_value = tf.random.uniform([])

        return tf.cond(
            random_value <= self.prob_of_rotate,
            lambda: self._execute_branch(image, label),
            lambda: (image, label),
        )
    
    
    def _execute_branch(self, image, label):

        '''

        num_rotate = tf.random.uniform([], minval=1, maxval=3, dtype=tf.int32)

        image = tf.image.rot90(image, k=num_rotate)

        if label is not None:
            label = tf.image.rot90(label, k=num_rotate)

        '''

        image, label = self.random_rotated_inputs(
            image,
            labels=label,
        )

        return image, label



    def random_rotated_inputs(
        self,
        images,
        labels=None,
        lower=0.0, 
        upper=1.0, 
    ):
        """Rotated inputs with random ops."""

        original_image_shape = images.shape

        unbatched = images.shape.rank == 3
        # The transform op only accepts rank 4 inputs,
        # so if we have an unbatched image,
        # we need to temporarily expand dims to a batch.
        if unbatched:
            images = tf.expand_dims(images, 0)
            
            if labels is not None:
                labels = tf.expand_dims(labels, 0)

        inputs_shape = tf.shape(images)
        batch_size = inputs_shape[0]
        img_hd = tf.cast(inputs_shape[H_AXIS], tf.float32)
        img_wd = tf.cast(inputs_shape[W_AXIS], tf.float32)
        min_angle = lower * 2.0 * np.pi
        max_angle = upper * 2.0 * np.pi
        angles = self._random_generator.random_uniform(
            shape=[batch_size], minval=min_angle, maxval=max_angle
        )

        rotation_matrix = get_rotation_matrix(angles, img_hd, img_wd)

        output_images = transform(
            images,
            rotation_matrix,
            fill_mode="constant",
            fill_value=self.fill_constant_color,
            interpolation="bilinear",
        )

        if labels is not None:
            output_labels = transform(
                labels,
                rotation_matrix,
                fill_mode="constant",
                fill_value=self.ignore_label,
                interpolation="nearest",
            )

        if unbatched:
            output_images = tf.squeeze(output_images, 0)

            if labels is not None:
                output_labels = tf.squeeze(output_labels, 0)

        output_images.set_shape(original_image_shape)

        if labels is not None:
            output_labels.set_shape(labels.shape)

            return output_images, output_labels

        return output_images