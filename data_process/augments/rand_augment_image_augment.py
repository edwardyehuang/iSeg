# ================================================================
# MIT License
# Copyright (c) 2026 edwardyehuang (https://github.com/edwardyehuang)
# ================================================================

import tensorflow as tf

from iseg.data_process.augments.data_augment_base import DataAugmentationBase, random_execute_helper
from iseg.data_process.augments.random_rotate_augment import get_rotation_matrix, transform


class RandAugmentImageAugment(DataAugmentationBase):
    def __init__(
        self,
        num_layers=2,
        magnitude=9.0,
        magnitude_max=10.0,
        magnitude_std=0.0,
        op_execute_prob=1.0,
        max_rotate_degree=30.0,
        max_shear_ratio=0.3,
        max_translate_ratio=0.45,
        max_enhance_delta=0.9,
        name=None,
    ):
        super().__init__(name=name)

        self.num_layers = int(num_layers)
        self.magnitude = float(magnitude)
        self.magnitude_max = float(magnitude_max)
        self.magnitude_std = float(magnitude_std)
        self.op_execute_prob = float(op_execute_prob)

        self.max_rotate_degree = float(max_rotate_degree)
        self.max_shear_ratio = float(max_shear_ratio)
        self.max_translate_ratio = float(max_translate_ratio)
        self.max_enhance_delta = float(max_enhance_delta)

        self._ops = [
            self._rotate,
            self._shear_x,
            self._shear_y,
            self._translate_x,
            self._translate_y,
            self._color,
            self._contrast,
            self._brightness,
            self._sharpness,
            self._solarize,
            self._posterize,
            self._autocontrast,
            self._equalize,
        ]

    def call(self, image, label):
        if self.num_layers <= 0:
            return image, label

        original_shape = image.shape

        _, image = tf.while_loop(
            lambda i, img: i < self.num_layers,
            lambda i, img: [i + 1, self._apply_one_layer(img)],
            [tf.constant(0, dtype=tf.int32), image],
            maximum_iterations=self.num_layers,
        )

        image = tf.clip_by_value(image, 0.0, 255.0)
        image.set_shape(original_shape)

        return image, label

    def _apply_one_layer(self, image):
        return random_execute_helper(
            self.op_execute_prob,
            lambda: self._execute_random_op(image),
            lambda: image,
        )

    def _execute_random_op(self, image):
        op_index = tf.random.uniform([], minval=0, maxval=len(self._ops), dtype=tf.int32)
        level = self._sample_level()
        sign = self._sample_sign()

        branch_fns = [
            (lambda fn=fn: fn(image, level, sign)) for fn in self._ops
        ]

        return tf.switch_case(op_index, branch_fns=branch_fns, default=lambda: image)

    def _sample_level(self):
        level = tf.cast(self.magnitude, tf.float32)

        if self.magnitude_std > 1e-6:
            level = level + tf.random.normal([], stddev=self.magnitude_std)

        level = tf.clip_by_value(level, 0.0, self.magnitude_max)

        return level / self.magnitude_max

    def _sample_sign(self):
        return tf.where(tf.random.uniform([]) < 0.5, -1.0, 1.0)

    def _apply_projective_transform(self, image, matrix):
        image = tf.expand_dims(image, axis=0)
        image = transform(
            image,
            matrix,
            fill_mode="constant",
            fill_value=0.0,
            interpolation="bilinear",
        )
        image = tf.squeeze(image, axis=0)

        return image

    def _rotate(self, image, level, sign):
        shape = tf.shape(image)
        height = tf.cast(shape[0], tf.float32)
        width = tf.cast(shape[1], tf.float32)

        angle_radian = sign * level * self.max_rotate_degree * (3.14159265 / 180.0)
        angle_radian = tf.reshape(angle_radian, [1])

        matrix = get_rotation_matrix(angle_radian, height, width)

        return self._apply_projective_transform(image, matrix)

    def _shear_x(self, image, level, sign):
        shear = sign * level * self.max_shear_ratio
        matrix = tf.reshape(
            tf.stack([
                tf.constant(1.0, dtype=tf.float32),
                shear,
                tf.constant(0.0, dtype=tf.float32),
                tf.constant(0.0, dtype=tf.float32),
                tf.constant(1.0, dtype=tf.float32),
                tf.constant(0.0, dtype=tf.float32),
                tf.constant(0.0, dtype=tf.float32),
                tf.constant(0.0, dtype=tf.float32),
            ]),
            [1, 8],
        )

        return self._apply_projective_transform(image, matrix)

    def _shear_y(self, image, level, sign):
        shear = sign * level * self.max_shear_ratio
        matrix = tf.reshape(
            tf.stack([
                tf.constant(1.0, dtype=tf.float32),
                tf.constant(0.0, dtype=tf.float32),
                tf.constant(0.0, dtype=tf.float32),
                shear,
                tf.constant(1.0, dtype=tf.float32),
                tf.constant(0.0, dtype=tf.float32),
                tf.constant(0.0, dtype=tf.float32),
                tf.constant(0.0, dtype=tf.float32),
            ]),
            [1, 8],
        )

        return self._apply_projective_transform(image, matrix)

    def _translate_x(self, image, level, sign):
        shape = tf.shape(image)
        width = tf.cast(shape[1], tf.float32)
        shift = sign * level * self.max_translate_ratio * width

        matrix = tf.reshape(
            tf.stack([
                tf.constant(1.0, dtype=tf.float32),
                tf.constant(0.0, dtype=tf.float32),
                -shift,
                tf.constant(0.0, dtype=tf.float32),
                tf.constant(1.0, dtype=tf.float32),
                tf.constant(0.0, dtype=tf.float32),
                tf.constant(0.0, dtype=tf.float32),
                tf.constant(0.0, dtype=tf.float32),
            ]),
            [1, 8],
        )

        return self._apply_projective_transform(image, matrix)

    def _translate_y(self, image, level, sign):
        shape = tf.shape(image)
        height = tf.cast(shape[0], tf.float32)
        shift = sign * level * self.max_translate_ratio * height

        matrix = tf.reshape(
            tf.stack([
                tf.constant(1.0, dtype=tf.float32),
                tf.constant(0.0, dtype=tf.float32),
                tf.constant(0.0, dtype=tf.float32),
                tf.constant(0.0, dtype=tf.float32),
                tf.constant(1.0, dtype=tf.float32),
                -shift,
                tf.constant(0.0, dtype=tf.float32),
                tf.constant(0.0, dtype=tf.float32),
            ]),
            [1, 8],
        )

        return self._apply_projective_transform(image, matrix)

    def _color(self, image, level, sign):
        factor = 1.0 + sign * level * self.max_enhance_delta
        gray = tf.image.grayscale_to_rgb(tf.image.rgb_to_grayscale(image))

        return self._blend(gray, image, factor)

    def _contrast(self, image, level, sign):
        factor = 1.0 + sign * level * self.max_enhance_delta

        return tf.image.adjust_contrast(image, factor)

    def _brightness(self, image, level, sign):
        delta = sign * level * self.max_enhance_delta * 255.0

        return image + delta

    def _sharpness(self, image, level, sign):
        factor = 1.0 + sign * level * self.max_enhance_delta

        kernel = tf.constant(
            [
                [1.0, 1.0, 1.0],
                [1.0, 5.0, 1.0],
                [1.0, 1.0, 1.0],
            ],
            dtype=tf.float32,
        ) / 13.0

        kernel = tf.reshape(kernel, [3, 3, 1, 1])
        channels = tf.shape(image)[-1]
        kernel = tf.tile(kernel, [1, 1, channels, 1])

        image_4d = tf.expand_dims(image, axis=0)
        smooth_4d = tf.nn.depthwise_conv2d(image_4d, kernel, strides=[1, 1, 1, 1], padding="SAME")
        smooth = tf.squeeze(smooth_4d, axis=0)

        return self._blend(smooth, image, factor)

    def _solarize(self, image, level, sign):
        del sign

        threshold = (1.0 - level) * 255.0

        return tf.where(image < threshold, image, 255.0 - image)

    def _posterize(self, image, level, sign):
        del sign

        bits = tf.cast(tf.round(8.0 - level * 4.0), tf.int32)
        bits = tf.clip_by_value(bits, 1, 8)
        shift = 8 - bits
        shift = tf.cast(shift, tf.uint8)

        image_uint8 = tf.cast(tf.clip_by_value(image, 0.0, 255.0), tf.uint8)
        image_uint8 = tf.bitwise.left_shift(tf.bitwise.right_shift(image_uint8, shift), shift)

        return tf.cast(image_uint8, image.dtype)

    def _autocontrast(self, image, level, sign):
        del level
        del sign

        lo = tf.reduce_min(image, axis=[0, 1], keepdims=True)
        hi = tf.reduce_max(image, axis=[0, 1], keepdims=True)

        scale = tf.math.divide_no_nan(255.0, hi - lo)
        projected = (image - lo) * scale

        use_projected = tf.cast(hi > lo, image.dtype)

        return projected * use_projected + image * (1.0 - use_projected)

    def _equalize(self, image, level, sign):
        del level
        del sign

        image_uint8 = tf.cast(tf.clip_by_value(image, 0.0, 255.0), tf.uint8)

        channels = tf.unstack(image_uint8, axis=-1)
        output_channels = [self._equalize_channel(channel) for channel in channels]

        return tf.cast(tf.stack(output_channels, axis=-1), image.dtype)

    def _equalize_channel(self, channel):
        channel_int = tf.cast(channel, tf.int32)
        hist = tf.histogram_fixed_width(channel_int, [0, 255], nbins=256)

        nonzero = tf.where(hist > 0)
        nonzero_hist = tf.gather_nd(hist, nonzero)

        step = (tf.reduce_sum(nonzero_hist) - nonzero_hist[-1]) // 255

        def _build_lut():
            lut = (tf.cumsum(hist) + (step // 2)) // step
            lut = tf.concat([tf.zeros([1], dtype=lut.dtype), lut[:-1]], axis=0)
            lut = tf.clip_by_value(lut, 0, 255)

            return tf.gather(lut, channel_int)

        return tf.cond(
            tf.equal(step, 0),
            lambda: channel_int,
            _build_lut,
        )

    def _blend(self, image0, image1, factor):
        image = image0 + factor * (image1 - image0)

        return tf.clip_by_value(image, 0.0, 255.0)
