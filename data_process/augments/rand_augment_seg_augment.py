# ================================================================
# MIT License
# Copyright (c) 2026 edwardyehuang (https://github.com/edwardyehuang)
# ================================================================

# Label-aware RandAugment for segmentation.
#
# Rationale: RandAugment (Cubuk et al., 2020) matches AutoAugment in practice
# while removing the search phase. The official deeplab2 kMaX COCO recipe
# enables autoaugment with "simple_classification_policy" (see
# deeplab2/configs/coco/kmax_deeplab/*.textproto); deeplab2's classification
# policy applies geometric ops to the label with nearest interpolation and
# photometric ops to the image only. This class is our equivalent: it reuses
# the exact op set and magnitude semantics of RandAugmentImageAugment
# (image-only, used by ImageNet classification) and makes the geometric ops
# label-aware.
#
# Op semantics:
# - GEOMETRIC ops (rotate / shear x / shear y / translate x / translate y):
#   the SAME sampled affine transform is applied to the label with NEAREST
#   interpolation; out-of-frame label pixels are filled with ignore_label
#   (255), matching how PadAugment fills padded label regions. The image
#   keeps the existing image-only behavior (bilinear, constant 0 fill).
# - PHOTOMETRIC ops (color / contrast / brightness / sharpness / solarize /
#   posterize / autocontrast / equalize): applied to the image ONLY. The
#   label passes through untouched — this is deliberate, since class ids and
#   instance ids are invariant to photometric changes.
# - Cutout / random-erasing-type ops are NOT part of the RandAugment op set
#   (mirroring RandAugmentImageAugment); erasing is handled by the existing
#   RandomErasingAugment, which already sets erased label pixels to
#   ignore_label.
#
# Supported label layouts: [H, W] (semantic) and [H, W, C] (e.g. the
# panoptic [H, W, 2] (class, instance id) layout of ids/panoptic_dataset.py).
# All label channels are transformed with the same nearest+ignore_label rule.

import tensorflow as tf

from iseg.data_process.augments.data_augment_base import random_execute_helper
from iseg.data_process.augments.rand_augment_image_augment import RandAugmentImageAugment
from iseg.data_process.augments.random_rotate_augment import transform


class RandAugmentSegAugment(RandAugmentImageAugment):
    def __init__(self, *args, ignore_label=255, **kwargs):
        super().__init__(*args, **kwargs)

        self.ignore_label = ignore_label

        # Geometric ops are dispatched via their projective-matrix builders so
        # the identical transform can be replayed on the label.
        self._geometric_matrix_fns = [
            self._rotate_matrix,
            self._shear_x_matrix,
            self._shear_y_matrix,
            self._translate_x_matrix,
            self._translate_y_matrix,
        ]

        # Photometric ops are inherited unchanged from the image-only version.
        self._photometric_fns = [
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
        # image: [H, W, 3] float; label: [H, W] or [H, W, C] int, or None.
        if self.num_layers <= 0:
            return image, label

        original_image_shape = image.shape

        # Fall back to image-only behavior when there is no spatial label
        # (e.g. classification-style scalar labels). The parent's call() can
        # NOT be reused: its while_loop resolves _apply_one_layer and
        # _execute_random_op on this subclass (both require a label), so the
        # parent's versions are bound explicitly here.
        if label is None or (label.shape.rank is not None and label.shape.rank < 2):

            def _image_only_layer(img):
                return random_execute_helper(
                    self.op_execute_prob,
                    lambda: RandAugmentImageAugment._execute_random_op(self, img),
                    lambda: img,
                )

            _, image = tf.while_loop(
                lambda i, img: i < self.num_layers,
                lambda i, img: [i + 1, _image_only_layer(img)],
                [tf.constant(0, dtype=tf.int32), image],
                maximum_iterations=self.num_layers,
            )
            image = tf.clip_by_value(image, 0.0, 255.0)
            image.set_shape(original_image_shape)
            return image, label
        original_label_shape = label.shape

        _, image, label = tf.while_loop(
            lambda i, img, lbl: i < self.num_layers,
            lambda i, img, lbl: [i + 1, *self._apply_one_layer(img, lbl)],
            [tf.constant(0, dtype=tf.int32), image, label],
            maximum_iterations=self.num_layers,
        )

        image = tf.clip_by_value(image, 0.0, 255.0)
        image.set_shape(original_image_shape)
        label.set_shape(original_label_shape)

        return image, label

    def _apply_one_layer(self, image, label):
        # image: [H, W, 3] float; label: [H, W] or [H, W, C] int
        return random_execute_helper(
            self.op_execute_prob,
            lambda: self._execute_random_op(image, label),
            lambda: (image, label),
        )

    def _execute_random_op(self, image, label):
        # image: [H, W, 3] float; label: [H, W] or [H, W, C] int
        op_index = tf.random.uniform(
            [],
            minval=0,
            maxval=len(self._geometric_matrix_fns) + len(self._photometric_fns),
            dtype=tf.int32,
        )
        level = self._sample_level()  # scalar float in [0, 1]
        sign = self._sample_sign()  # scalar float in {-1.0, 1.0}

        branch_fns = [
            (lambda matrix_fn=matrix_fn: self._apply_geometric_op(matrix_fn, image, label, level, sign))
            for matrix_fn in self._geometric_matrix_fns
        ]

        # Photometric ops touch the image only; the label passes through
        # untouched by design (see module docstring).
        branch_fns += [
            (lambda photo_fn=photo_fn: (photo_fn(image, level, sign), label))
            for photo_fn in self._photometric_fns
        ]

        return tf.switch_case(op_index, branch_fns=branch_fns, default=lambda: (image, label))

    def _apply_geometric_op(self, matrix_fn, image, label, level, sign):
        # matrix: [1, 8] projective transform (output -> input mapping)
        matrix = matrix_fn(image, level, sign)

        # Image keeps the existing image-only fill behavior (bilinear, 0 fill).
        image = self._apply_projective_transform(image, matrix)  # [H, W, 3] float
        # Label is transformed with the SAME matrix, nearest interpolation and
        # ignore_label fill for out-of-frame pixels.
        label = self._apply_projective_transform_to_label(label, matrix)  # same shape/dtype as input label

        return image, label

    def _apply_projective_transform_to_label(self, label, matrix):
        # label: [H, W] or [H, W, C] integer label map; matrix: [1, 8]
        label_dtype = label.dtype
        squeeze_channel = label.shape.rank == 2

        if squeeze_channel:
            label = tf.expand_dims(label, axis=-1)  # [H, W] -> [H, W, 1]

        # ImageProjectiveTransformV3 takes float fill_value; label values are
        # small ints so float32 round-trips them exactly under nearest interp.
        label = tf.cast(label, tf.float32)  # [H, W, C] float
        label = tf.expand_dims(label, axis=0)  # [1, H, W, C]

        label = transform(
            label,
            matrix,
            fill_mode="constant",
            fill_value=float(self.ignore_label),
            interpolation="nearest",
        )  # [1, H, W, C]

        label = tf.squeeze(label, axis=0)  # [H, W, C]
        label = tf.cast(label, label_dtype)  # nearest interp keeps values integral

        if squeeze_channel:
            label = tf.squeeze(label, axis=-1)  # [H, W, 1] -> [H, W]

        return label
