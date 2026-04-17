# ================================================================
# MIT License
# Copyright (c) 2026 edwardyehuang (https://github.com/edwardyehuang)
# ================================================================

import tensorflow as tf

from iseg.data_process.augments.data_augment_base import DataAugmentationBase, random_execute_helper


class ClassificationBatchMixAugment(DataAugmentationBase):
    def __init__(
        self,
        num_class,
        mixup_alpha=0.8,
        mixup_prob=1.0,
        cutmix_alpha=1.0,
        cutmix_prob=1.0,
        switch_prob=0.5,
        label_smoothing=0.0,
        name=None,
    ):
        super().__init__(name=name)

        self.num_class = int(num_class)

        self.mixup_alpha = float(mixup_alpha)
        self.mixup_prob = float(mixup_prob)

        self.cutmix_alpha = float(cutmix_alpha)
        self.cutmix_prob = float(cutmix_prob)

        self.switch_prob = float(switch_prob)
        self.label_smoothing = float(label_smoothing)

    def call(self, image, label):
        label = self._to_soft_labels(label)

        use_mixup = self.mixup_alpha > 0 + 1e-6 and self.mixup_prob > 0 + 1e-6
        use_cutmix = self.cutmix_alpha > 0 + 1e-6 and self.cutmix_prob > 0 + 1e-6

        if use_mixup and use_cutmix:
            image, label = tf.cond(
                tf.random.uniform([]) < self.switch_prob,
                lambda: self._apply_mixup_with_prob(image, label),
                lambda: self._apply_cutmix_with_prob(image, label),
            )
        elif use_mixup:
            image, label = self._apply_mixup_with_prob(image, label)
        elif use_cutmix:
            image, label = self._apply_cutmix_with_prob(image, label)

        if self.label_smoothing > 0 + 1e-6:
            label = self._apply_label_smoothing(label)

        image = tf.cast(image, tf.float32)
        label = tf.cast(label, tf.float32)

        return image, label

    def _apply_mixup_with_prob(self, image, label):
        return random_execute_helper(
            self.mixup_prob,
            lambda: self._apply_mixup(image, label),
            lambda: (image, label),
        )

    def _apply_cutmix_with_prob(self, image, label):
        return random_execute_helper(
            self.cutmix_prob,
            lambda: self._apply_cutmix(image, label),
            lambda: (image, label),
        )

    def _apply_mixup(self, image, label):
        batch_size = tf.shape(image)[0]
        shuffled_indices = tf.random.shuffle(tf.range(batch_size))

        image_b = tf.gather(image, shuffled_indices)
        label_b = tf.gather(label, shuffled_indices)

        lam = self._sample_beta(self.mixup_alpha)
        lam = tf.maximum(lam, 1.0 - lam)

        lam_image = tf.reshape(lam, [1, 1, 1, 1])
        lam_label = tf.reshape(lam, [1, 1])

        image = lam_image * image + (1.0 - lam_image) * image_b
        label = lam_label * label + (1.0 - lam_label) * label_b

        return image, label

    def _apply_cutmix(self, image, label):
        batch_size = tf.shape(image)[0]
        image_h = tf.shape(image)[1]
        image_w = tf.shape(image)[2]

        shuffled_indices = tf.random.shuffle(tf.range(batch_size))
        image_b = tf.gather(image, shuffled_indices)
        label_b = tf.gather(label, shuffled_indices)

        lam = self._sample_beta(self.cutmix_alpha)

        cut_ratio = tf.sqrt(1.0 - lam)

        cut_w = tf.cast(cut_ratio * tf.cast(image_w, tf.float32), tf.int32)
        cut_h = tf.cast(cut_ratio * tf.cast(image_h, tf.float32), tf.int32)

        center_x = tf.random.uniform([], minval=0, maxval=image_w, dtype=tf.int32)
        center_y = tf.random.uniform([], minval=0, maxval=image_h, dtype=tf.int32)

        x1 = tf.clip_by_value(center_x - cut_w // 2, 0, image_w)
        x2 = tf.clip_by_value(center_x + cut_w // 2, 0, image_w)

        y1 = tf.clip_by_value(center_y - cut_h // 2, 0, image_h)
        y2 = tf.clip_by_value(center_y + cut_h // 2, 0, image_h)

        cutout = tf.ones([y2 - y1, x2 - x1, 1], dtype=image.dtype)
        cutout = tf.pad(cutout, [[y1, image_h - y2], [x1, image_w - x2], [0, 0]])
        cutout = tf.expand_dims(cutout, axis=0)

        image = image * (1.0 - cutout) + image_b * cutout

        cut_area = tf.cast((y2 - y1) * (x2 - x1), tf.float32)
        total_area = tf.cast(image_h * image_w, tf.float32)

        lam = 1.0 - tf.math.divide_no_nan(cut_area, total_area)
        lam_label = tf.reshape(lam, [1, 1])

        label = lam_label * label + (1.0 - lam_label) * label_b

        return image, label

    def _sample_beta(self, alpha):
        alpha = tf.cast(alpha, tf.float32)

        x = tf.random.gamma(shape=[], alpha=alpha)
        y = tf.random.gamma(shape=[], alpha=alpha)

        return tf.math.divide_no_nan(x, x + y)

    def _to_soft_labels(self, label):
        if label.dtype.is_floating:
            label = tf.cast(label, tf.float32)

            if label.shape.rank == 1:
                label = tf.cast(label, tf.int32)
                label = tf.one_hot(label, self.num_class, dtype=tf.float32)
            else:
                label = tf.reshape(label, [-1, self.num_class])

            return label

        if label.shape.rank is not None and label.shape.rank > 1 and label.shape[-1] == 1:
            label = tf.squeeze(label, axis=-1)

        label = tf.cast(label, tf.int32)
        label = tf.reshape(label, [-1])

        return tf.one_hot(label, self.num_class, dtype=tf.float32)

    def _apply_label_smoothing(self, label):
        smooth = self.label_smoothing / tf.cast(self.num_class, tf.float32)

        return label * (1.0 - self.label_smoothing) + smooth
