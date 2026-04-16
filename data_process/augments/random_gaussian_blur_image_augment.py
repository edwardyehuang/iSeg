# ================================================================
# MIT License
# Copyright (c) 2026 edwardyehuang (https://github.com/edwardyehuang)
# ================================================================

import tensorflow as tf

from iseg.data_process.augments.data_augment_base import DataAugmentationBase, random_execute_helper


class RandomGaussianBlurImageAugment(DataAugmentationBase):
    def __init__(
        self,
        kernel_size=23,
        sigma_min=0.1,
        sigma_max=2.0,
        execute_prob=0.0,
        name=None,
    ):
        super().__init__(name=name)

        if kernel_size <= 0 or kernel_size % 2 == 0:
            raise ValueError("kernel_size must be positive and odd")

        if sigma_min <= 0 or sigma_max <= 0 or sigma_min > sigma_max:
            raise ValueError("sigma_min and sigma_max must be > 0 and sigma_min <= sigma_max")

        self.kernel_size = int(kernel_size)
        self.sigma_min = float(sigma_min)
        self.sigma_max = float(sigma_max)
        self.execute_prob = execute_prob

    def _gaussian_kernel_2d(self, sigma, dtype):
        radius = self.kernel_size // 2

        x = tf.range(-radius, radius + 1, dtype=tf.float32)
        kernel_1d = tf.exp(-0.5 * tf.square(x / sigma))
        kernel_1d /= tf.reduce_sum(kernel_1d)

        kernel_2d = kernel_1d[:, None] * kernel_1d[None, :]

        return tf.cast(kernel_2d, dtype)

    def _apply_blur(self, image):
        sigma = tf.random.uniform([], minval=self.sigma_min, maxval=self.sigma_max, dtype=tf.float32)

        kernel_2d = self._gaussian_kernel_2d(sigma=sigma, dtype=image.dtype)
        kernel_2d = kernel_2d[:, :, None, None]

        image_shape = tf.shape(image)
        channels = image_shape[-1]

        kernel = tf.tile(kernel_2d, [1, 1, channels, 1])

        image = tf.expand_dims(image, axis=0)
        image = tf.nn.depthwise_conv2d(image, kernel, strides=[1, 1, 1, 1], padding="SAME")
        image = tf.squeeze(image, axis=0)

        return image

    def call(self, image, label):

        image = random_execute_helper(
            self.execute_prob,
            lambda: self._apply_blur(image),
            lambda: image,
        )

        image.set_shape([None, None, image.shape[-1]])

        return image, label
