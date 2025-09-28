# ================================================================
# MIT License
# Copyright (c) 2021 edwardyehuang (https://github.com/edwardyehuang)
# ================================================================

import numpy as np

import tensorflow as tf
import keras
import iseg.data_process.utils as dataprocess

from iseg.utils.keras3_utils import is_keras3

from iseg.data_process.augments.data_augment_base import DataAugmentationBase


H_AXIS = -3
W_AXIS = -2

def transform(
    images,
    transforms,
    fill_mode="reflect",
    fill_value=0.0,
    interpolation="bilinear",
    output_shape=None,
    name=None,
):
    """Applies the given transform(s) to the image(s).

    Args:
        images: A tensor of shape
            `(num_images, num_rows, num_columns, num_channels)` (NHWC).
            The rank must be statically known
            (the shape is not `TensorShape(None)`).
        transforms: Projective transform matrix/matrices.
            A vector of length 8 or tensor of size N x 8.
            If one row of transforms is [a0, a1, a2, b0, b1, b2,
            c0, c1], then it maps the *output* point `(x, y)`
            to a transformed *input* point
            `(x', y') = ((a0 x + a1 y + a2) / k, (b0 x + b1 y + b2) / k)`, where
            `k = c0 x + c1 y + 1`. The transforms are *inverted* compared to the
            transform mapping input points to output points.
            Note that gradients are not backpropagated
            into transformation parameters.
        fill_mode: Points outside the boundaries of the input are filled
            according to the given mode
            (one of `{"constant", "reflect", "wrap", "nearest"}`).
        fill_value: a float represents the value to be filled outside
            the boundaries when `fill_mode="constant"`.
        interpolation: Interpolation mode. Supported values: `"nearest"`,
            `"bilinear"`.
        output_shape: Output dimension after the transform, `[height, width]`.
            If `None`, output is the same size as input image.
        name: The name of the op.

    Fill mode behavior for each valid value is as follows:

    - `"reflect"`: `(d c b a | a b c d | d c b a)`
    The input is extended by reflecting about the edge of the last pixel.

    - `"constant"`: `(k k k k | a b c d | k k k k)`
    The input is extended by filling all
    values beyond the edge with the same constant value k = 0.

    - `"wrap"`: `(a b c d | a b c d | a b c d)`
    The input is extended by wrapping around to the opposite edge.

    - `"nearest"`: `(a a a a | a b c d | d d d d)`
    The input is extended by the nearest pixel.

    Input shape:
        4D tensor with shape: `(samples, height, width, channels)`,
            in `"channels_last"` format.

    Output shape:
        4D tensor with shape: `(samples, height, width, channels)`,
            in `"channels_last"` format.

    Returns:
        Image(s) with the same type and shape as `images`, with the given
        transform(s) applied. Transformed coordinates outside of the input image
        will be filled with zeros.
    """
    with tf.name_scope(name or "transform"):
        if output_shape is None:
            output_shape = tf.shape(images)[1:3]
            if not tf.executing_eagerly():
                output_shape_value = tf.get_static_value(output_shape)
                if output_shape_value is not None:
                    output_shape = output_shape_value

        output_shape = tf.convert_to_tensor(
            output_shape, tf.int32, name="output_shape"
        )

        if not output_shape.get_shape().is_compatible_with([2]):
            raise ValueError(
                "output_shape must be a 1-D Tensor of 2 elements: "
                "new_height, new_width, instead got "
                f"output_shape={output_shape}"
            )

        fill_value = tf.convert_to_tensor(
            fill_value, tf.float32, name="fill_value"
        )

        return tf.raw_ops.ImageProjectiveTransformV3(
            images=images,
            output_shape=output_shape,
            fill_value=fill_value,
            transforms=transforms,
            fill_mode=fill_mode.upper(),
            interpolation=interpolation.upper(),
        )


def get_rotation_matrix(angles, image_height, image_width, name=None):
    """Returns projective transform(s) for the given angle(s).

    Args:
        angles: A scalar angle to rotate all images by,
            or (for batches of images) a vector with an angle to
            rotate each image in the batch. The rank must be
            statically known (the shape is not `TensorShape(None)`).
        image_height: Height of the image(s) to be transformed.
        image_width: Width of the image(s) to be transformed.
        name: The name of the op.

    Returns:
        A tensor of shape (num_images, 8).
            Projective transforms which can be given
            to operation `image_projective_transform_v2`.
            If one row of transforms is
            [a0, a1, a2, b0, b1, b2, c0, c1], then it maps the *output* point
            `(x, y)` to a transformed *input* point
            `(x', y') = ((a0 x + a1 y + a2) / k, (b0 x + b1 y + b2) / k)`,
            where `k = c0 x + c1 y + 1`.
    """
    with tf.name_scope(name or "rotation_matrix"):
        x_offset = (
            (image_width - 1)
            - (
                tf.cos(angles) * (image_width - 1)
                - tf.sin(angles) * (image_height - 1)
            )
        ) / 2.0
        y_offset = (
            (image_height - 1)
            - (
                tf.sin(angles) * (image_width - 1)
                + tf.cos(angles) * (image_height - 1)
            )
        ) / 2.0
        num_angles = tf.shape(angles)[0]
        return tf.concat(
            values=[
                tf.cos(angles)[:, None],
                -tf.sin(angles)[:, None],
                x_offset[:, None],
                tf.sin(angles)[:, None],
                tf.cos(angles)[:, None],
                y_offset[:, None],
                tf.zeros((num_angles, 2), tf.float32),
            ],
            axis=1,
        )



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

        if labels is not None:
            original_label_shape = labels.shape

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
        angles = tf.random.uniform(
            shape=[batch_size], minval=min_angle, maxval=max_angle
        )

        rotation_matrix = get_rotation_matrix(angles, img_hd, img_wd)

        output_images = transform(
            images,
            rotation_matrix,
            fill_mode="constant",
            fill_value=-1.0,
            interpolation="bilinear",
        )

        fill_constant_color = tf.reshape(
            tf.constant(self.fill_constant_color, dtype=output_images.dtype), 
            [1, 1, 1, 3]
        )

        output_images = tf.where(
            tf.less(output_images, 0.0 - 1e-6),
            tf.constant(fill_constant_color, dtype=output_images.dtype),
            output_images,
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
            output_labels.set_shape(original_label_shape)

            return output_images, output_labels

        return output_images