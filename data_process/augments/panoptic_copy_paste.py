# ================================================================
# MIT License
# Copyright (c) 2026 edwardyehuang (https://github.com/edwardyehuang)
# ================================================================

# Panoptic copy-paste augmentation (kMaX, ECCV 2022; deeplab2 reference:
# data/sample_generator.py::_get_panoptic_copy_paste and
# data/dataloader/input_reader.py).
#
# Semantics mirrored from the official implementation:
# - The donor sample is drawn by zipping the dataset with a reshuffled copy of
#   itself (same pattern as deeplab2's input_reader).
# - The donor label's unique (class, instance) pairs are randomly subsampled
#   (keep k ~ Uniform{0..N} pairs, shuffled), then adjusted per option:
#     thing_option="all"  -> every donor thing instance is always pasted;
#     stuff_option="random" -> donor stuff regions follow the random subset;
#     void_option="none"  -> donor void (ignore_label) pixels are never pasted.
#   These are the exact options of the official kMaX COCO recipe.
# - Host pixels under pasted regions are overwritten; everything else is
#   untouched.
#
# Adaptations from the official version (documented per the repo's 2-channel
# panoptic label contract, see ids/panoptic_dataset.py):
# - Labels are [H, W, 2] (channel 0 = class, channel 1 = instance id) instead
#   of deeplab2's single panoptic id (class * label_divisor + instance).
# - Instead of inverting donor instance ids into the upper id range (official
#   trick requiring max instances < label_divisor // 2), pasted thing
#   instances receive FRESH ids offset by the host's max instance id, so they
#   can never collide with host ids. Donor stuff pastes with instance id 0.
# - Official applies copy-paste after both host and donor are fully
#   preprocessed (random scale with the donor's scale range multiplied by
#   panoptic_copy_paste_scale=0.5, then pad/crop to crop size). Here it is
#   applied on the raw decoded tensors at the DATASET level, before the
#   standard augment pipeline; the donor keeps its NATIVE scale (center-crop
#   or pad to the host canvas, never stretched — an earlier version resized
#   the donor to the host's exact (H, W), distorting every pasted instance
#   whenever the aspect ratios differed) and the shared subsequent random
#   scale jitter provides the scale diversity the official donor rescaling
#   provides.
# - `prob` adds a per-sample execution probability (panoptic_copy_paste_prob
#   flag); the official version applies copy-paste unconditionally once
#   enabled.
#
# Everything below is pure TF ops and runs inside tf.data map functions
# (graph mode), no tf.py_function / numpy_function.

import tensorflow as tf

from iseg.data_process.augments.data_augment_base import DataAugmentationBase, random_execute_helper

# Option semantics mirrored from deeplab2 sample_generator._get_panoptic_copy_paste.
OPTION_NONE = "none"
OPTION_RANDOM = "random"
OPTION_ALL = "all"

_VALID_OPTIONS = (OPTION_NONE, OPTION_RANDOM, OPTION_ALL)

# Multiplier for packing a (class, instance) pair into a single int64 key for
# tf.unique. Per-image instance counts are far below 2**20 in practice.
_INSTANCE_KEY_MULTIPLIER = 1 << 20


def panoptic_copy_paste(
    image,
    label,
    donor_image,
    donor_label,
    prob=1.0,
    num_thing_classes=80,
    ignore_label=255,
    thing_option=OPTION_ALL,
    stuff_option=OPTION_RANDOM,
    void_option=OPTION_NONE,
):
    """Merges donor segments into a host sample. See module docstring.

    Args:
        image: host image, float tensor of shape [H, W, 3].
        label: host panoptic label, int tensor of shape [H, W, 2]
            (channel 0 = class, channel 1 = instance id).
        donor_image: donor image, float tensor of shape [Hd, Wd, 3].
        donor_label: donor panoptic label, int tensor of shape [Hd, Wd, 2].
        prob: per-sample probability of applying the paste.
        num_thing_classes: classes [0, num_thing_classes) are "thing"
            (have instances), [num_thing_classes, ignore_label) are "stuff".
        ignore_label: void/ignore class id (never pasted).
        thing_option/stuff_option/void_option: one of "all", "random", "none".

    Returns:
        (merged_image, merged_label) with the same shapes/dtypes as
        (image, label).
    """

    if thing_option not in _VALID_OPTIONS:
        raise ValueError(f"Invalid thing_option = {thing_option}, must be one of {_VALID_OPTIONS}")
    if stuff_option not in _VALID_OPTIONS:
        raise ValueError(f"Invalid stuff_option = {stuff_option}, must be one of {_VALID_OPTIONS}")
    if void_option not in _VALID_OPTIONS:
        raise ValueError(f"Invalid void_option = {void_option}, must be one of {_VALID_OPTIONS}")

    if label.shape.rank is not None and (label.shape.rank != 3 or label.shape[-1] != 2):
        raise ValueError(
            "panoptic_copy_paste requires panoptic labels of shape [H, W, 2] "
            f"(class, instance id), got shape {label.shape}"
        )

    original_image_shape = image.shape
    original_label_shape = label.shape

    image = tf.convert_to_tensor(image, dtype=tf.float32)  # [H, W, 3]
    label = tf.convert_to_tensor(label, dtype=tf.int32)  # [H, W, 2]
    donor_image = tf.convert_to_tensor(donor_image, dtype=tf.float32)  # [Hd, Wd, 3]
    donor_label = tf.convert_to_tensor(donor_label, dtype=tf.int32)  # [Hd, Wd, 2]

    image, label = random_execute_helper(
        prob,
        lambda: _panoptic_copy_paste_internal(
            image, label, donor_image, donor_label,
            num_thing_classes=num_thing_classes,
            ignore_label=ignore_label,
            thing_option=thing_option,
            stuff_option=stuff_option,
            void_option=void_option,
        ),
        lambda: (image, label),
    )

    image.set_shape(original_image_shape)
    label.set_shape(original_label_shape)

    return image, label


def _fit_donor_to_canvas(donor_image, donor_label, host_hw, ignore_label):
    """Center-crop/pad the donor to the host canvas WITHOUT rescaling.

    Pasted segments keep their native pixel scale (official deeplab2 behavior:
    the donor goes through its own aspect-preserving preprocessing and is
    pasted as-is). Overflow is center-cropped, underflow is padded with void
    labels; the padded region holds no segments and never contributes.

    Args:
        donor_image: `[Hd, Wd, 3]` float tensor.
        donor_label: `[Hd, Wd, 2]` int32 tensor.
        host_hw: `[2]` int tensor, the host's (H, W).
        ignore_label: Void class id used to pad the label (both channels).

    Returns:
        `(donor_image, donor_label)` at `[H, W, 3]` / `[H, W, 2]`.
    """

    donor_hw = tf.shape(donor_image)[:2]  # [2] = (Hd, Wd)

    # Crop to at most the host size.
    crop_hw = tf.minimum(donor_hw, host_hw)  # [2]
    top_left = (donor_hw - crop_hw) // 2  # [2]
    donor_image = donor_image[
        top_left[0] : top_left[0] + crop_hw[0],
        top_left[1] : top_left[1] + crop_hw[1],
    ]  # [h, w, 3]
    donor_label = donor_label[
        top_left[0] : top_left[0] + crop_hw[0],
        top_left[1] : top_left[1] + crop_hw[1],
    ]  # [h, w, 2]

    # Pad to the host size; the padded region is void (class 255) in the
    # label and black in the image — it holds no segments by construction.
    pad_hw = host_hw - crop_hw  # [2]
    pad_top = pad_hw[0] // 2
    pad_left = pad_hw[1] // 2
    donor_image = tf.pad(
        donor_image,
        [[pad_top, pad_hw[0] - pad_top], [pad_left, pad_hw[1] - pad_left], [0, 0]],
    )  # [H, W, 3]
    donor_label = tf.pad(
        donor_label,
        [[pad_top, pad_hw[0] - pad_top], [pad_left, pad_hw[1] - pad_left], [0, 0]],
        constant_values=ignore_label,
    )  # [H, W, 2]

    return donor_image, donor_label


def _panoptic_copy_paste_internal(
    image,
    label,
    donor_image,
    donor_label,
    num_thing_classes,
    ignore_label,
    thing_option,
    stuff_option,
    void_option,
):
    # image: [H, W, 3] float; label: [H, W, 2] int32
    # donor_image: [Hd, Wd, 3] float; donor_label: [Hd, Wd, 2] int32

    host_hw = tf.shape(image)[:2]  # [2]

    # Keep the donor at its NATIVE scale: the previous version resized it to
    # the exact host (H, W), which anisotropically stretches every pasted
    # instance whenever the donor's aspect ratio differs from the host's —
    # teaching distorted shape priors to the model. Official deeplab2 runs the
    # donor through its own (aspect-preserving) preprocessing and pastes at
    # native scale. Overflow is center-cropped, underflow is padded with void
    # (those pixels hold no segments, so they never contribute to the paste).
    donor_image, donor_label = _fit_donor_to_canvas(
        donor_image, donor_label, host_hw, ignore_label
    )  # [H, W, 3], [H, W, 2]
    donor_image = tf.cast(donor_image, image.dtype)  # [H, W, 3]

    donor_class = donor_label[..., 0]  # [H, W]
    donor_instance = donor_label[..., 1]  # [H, W]

    # Unique (class, instance) pairs of the donor, packed into int64 keys.
    flat_pair_key = tf.reshape(
        tf.cast(donor_class, tf.int64) * _INSTANCE_KEY_MULTIPLIER + tf.cast(donor_instance, tf.int64),
        [-1],
    )  # [H*W]
    unique_pair_key, pixel_pair_idx = tf.unique(flat_pair_key)  # [U], [H*W]

    unique_class = tf.cast(unique_pair_key // _INSTANCE_KEY_MULTIPLIER, tf.int32)  # [U]

    is_void = tf.equal(unique_class, ignore_label)  # [U]
    is_thing = tf.logical_and(unique_class < num_thing_classes, tf.logical_not(is_void))  # [U]
    is_stuff = tf.logical_and(unique_class >= num_thing_classes, tf.logical_not(is_void))  # [U]

    # Randomly keep k ~ Uniform{0..U} unique pairs (official "random" subset).
    num_unique = tf.shape(unique_pair_key)[0]
    num_keep = tf.random.uniform([], maxval=num_unique + 1, dtype=tf.int32)
    keep_mask = tf.concat(
        [tf.ones([num_keep], tf.int32), tf.zeros([num_unique - num_keep], tf.int32)],
        axis=0,
    )  # [U]
    keep_mask = tf.random.shuffle(keep_mask)  # [U]

    keep_mask = _update_keep_mask(keep_mask, tf.cast(is_thing, tf.int32), thing_option)  # [U]
    keep_mask = _update_keep_mask(keep_mask, tf.cast(is_stuff, tf.int32), stuff_option)  # [U]
    keep_mask = _update_keep_mask(keep_mask, tf.cast(is_void, tf.int32), void_option)  # [U]

    # Per-pixel paste mask: True where a donor pixel belongs to a kept pair.
    per_pixel_keep = tf.gather(keep_mask, pixel_pair_idx)  # [H*W]
    paste_mask = tf.reshape(per_pixel_keep > 0, host_hw)  # [H, W] bool

    # Fresh instance ids for kept donor thing pairs: host_max + 1, +2, ...
    # in unique-pair order, so pasted ids can never collide with host ids.
    # Kept donor stuff pairs map to instance id 0.
    host_max_instance = tf.reduce_max(label[..., 1])  # scalar int32
    keep_thing = keep_mask * tf.cast(is_thing, tf.int32)  # [U], 1 for kept thing pairs
    fresh_id_per_pair = (host_max_instance + tf.cumsum(keep_thing)) * keep_thing  # [U]
    donor_fresh_instance = tf.reshape(tf.gather(fresh_id_per_pair, pixel_pair_idx), host_hw)  # [H, W]

    # Composite: donor overwrites host on pasted pixels; host untouched elsewhere.
    merged_image = tf.where(paste_mask[..., tf.newaxis], donor_image, image)  # [H, W, 3]
    merged_class = tf.where(paste_mask, donor_class, label[..., 0])  # [H, W]
    merged_instance = tf.where(paste_mask, donor_fresh_instance, label[..., 1])  # [H, W]
    merged_label = tf.stack([merged_class, merged_instance], axis=-1)  # [H, W, 2]

    return merged_image, merged_label


def _update_keep_mask(keep_mask, class_list_mask, option):
    """Mirrors deeplab2's update_label_id_mask. All args are int32 [U]."""

    if option == OPTION_NONE:
        # Drop every pair whose class is in the list.
        return keep_mask * (1 - class_list_mask)
    elif option == OPTION_ALL:
        # Force-keep every pair whose class is in the list.
        return tf.maximum(keep_mask, class_list_mask)
    # OPTION_RANDOM: pairs follow the random subset unchanged.
    return keep_mask


class PanopticCopyPasteAugment(DataAugmentationBase):
    """4-input variant of DataAugmentationBase; used at dataset level where a
    donor sample is available (see apply_panoptic_copy_paste_to_dataset)."""

    def __init__(
        self,
        prob=0.5,
        num_thing_classes=80,
        ignore_label=255,
        thing_option=OPTION_ALL,
        stuff_option=OPTION_RANDOM,
        void_option=OPTION_NONE,
        name=None,
    ):
        super().__init__(name=name)

        self.prob = float(prob)
        self.num_thing_classes = int(num_thing_classes)
        self.ignore_label = int(ignore_label)
        self.thing_option = thing_option
        self.stuff_option = stuff_option
        self.void_option = void_option

    def call(self, image, label, donor_image, donor_label):
        return panoptic_copy_paste(
            image,
            label,
            donor_image,
            donor_label,
            prob=self.prob,
            num_thing_classes=self.num_thing_classes,
            ignore_label=self.ignore_label,
            thing_option=self.thing_option,
            stuff_option=self.stuff_option,
            void_option=self.void_option,
        )


def apply_panoptic_copy_paste_to_dataset(
    ds,
    prob=0.5,
    num_thing_classes=80,
    ignore_label=255,
    shuffle_buffer_size=1024,
    thing_option=OPTION_ALL,
    stuff_option=OPTION_RANDOM,
    void_option=OPTION_NONE,
):
    """Zips a per-sample (image, label) dataset with a reshuffled copy of
    itself and pastes donor segments onto each sample.

    This mirrors deeplab2's input_reader, which duplicates the dataset with
    another random order and zips them together. Must be called on the raw
    decoded stream, BEFORE the standard augment pipeline.
    """

    augment = PanopticCopyPasteAugment(
        prob=prob,
        num_thing_classes=num_thing_classes,
        ignore_label=ignore_label,
        thing_option=thing_option,
        stuff_option=stuff_option,
        void_option=void_option,
    )

    donor_ds = ds.shuffle(shuffle_buffer_size, reshuffle_each_iteration=True)
    zipped_ds = tf.data.Dataset.zip((ds, donor_ds))

    print(f"Applied panoptic copy-paste augment (prob = {prob}, donor shuffle buffer = {shuffle_buffer_size})")

    return zipped_ds.map(
        _copy_paste_map_fn(augment),
        num_parallel_calls=tf.data.experimental.AUTOTUNE,
    )


def _copy_paste_map_fn(augment):

    @tf.autograph.experimental.do_not_convert
    def map_fn(host, donor):
        # host/donor: (image [H, W, 3] float, label [H, W, 2] int32)
        return augment(host[0], host[1], donor[0], donor[1])

    return map_fn
