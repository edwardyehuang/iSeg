# ================================================================
# MIT License
# Copyright (c) 2021 edwardyehuang (https://github.com/edwardyehuang)
# ================================================================

import iseg.static_strings as ss
import tensorflow as tf
import keras
from tensorflow.python.framework import ops

from iseg.utils import resize_image


def _crop(image, offset_height, offset_width, crop_height, crop_width):
    """Crops the given image using the provided offsets and sizes.

    Note that the method doesn't assume we know the input image size but it does
    assume we know the input image rank.

    Args:
        image: an image of shape [height, width, channels].
        offset_height: a scalar tensor indicating the height offset.
        offset_width: a scalar tensor indicating the width offset.
        crop_height: the height of the cropped image.
        crop_width: the width of the cropped image.

    Returns:
        The cropped (and resized) image.

    Raises:
        ValueError: if `image` doesn't have rank of 3.
        InvalidArgumentError: if the rank is not 3 or if the image dimensions are
        less than the crop size.
    """
    original_shape = tf.shape(image)

    if len(image.get_shape().as_list()) != 3:
        raise ValueError("input must have rank of 3")

    original_channels = image.get_shape().as_list()[2]

    rank_assertion = tf.Assert(tf.equal(tf.rank(image), 3), ["Rank of image must be equal to 3."])
    with tf.control_dependencies([rank_assertion]):
        cropped_shape = tf.stack([crop_height, crop_width, original_shape[2]])

    size_assertion = tf.Assert(
        tf.logical_and(
            tf.greater_equal(original_shape[0], crop_height), tf.greater_equal(original_shape[1], crop_width)
        ),
        ["Crop size greater than the image size."],
    )

    offsets = tf.cast(tf.stack([offset_height, offset_width, 0]), tf.dtypes.int32)

    # Use tf.slice instead of crop_to_bounding box as it accepts tensors to
    # define the crop size.
    with tf.control_dependencies([size_assertion]):
        image = tf.slice(image, offsets, cropped_shape)
    image = tf.reshape(image, cropped_shape)
    image.set_shape([crop_height, crop_width, original_channels])
    return image


def random_crop(image_list, crop_height, crop_width):
    """Crops the given list of images.

    The function applies the same crop to each image in the list. This can be
    effectively applied when there are multiple image inputs of the same
    dimension such as:

        image, depths, normals = random_crop([image, depths, normals], 120, 150)

    Args:
        image_list: a list of image tensors of the same dimension but possibly
        varying channel.
        crop_height: the new height.
        crop_width: the new width.

    Returns:
        the image_list with cropped images.

    Raises:
        ValueError: if there are multiple image inputs provided with different size
        or the images are smaller than the crop dimensions.
    """
    if not image_list:
        raise ValueError("Empty image_list.")

    # Compute the rank assertions.
    rank_assertions = []

    for i in range(len(image_list)):
        image_rank = tf.rank(image_list[i])
        rank_assert = tf.Assert(tf.equal(image_rank, 3), ["Wrong rank for tensor [expected] [actual]", 3, image_rank])
        rank_assertions.append(rank_assert)

    with tf.control_dependencies([rank_assertions[0]]):
        image_shape = tf.shape(image_list[0])

    image_height = image_shape[0]
    image_width = image_shape[1]
    crop_size_assert = tf.Assert(
        tf.logical_and(tf.greater_equal(image_height, crop_height), tf.greater_equal(image_width, crop_width)),
        ["Crop size greater than the image size."],
    )

    asserts = [rank_assertions[0], crop_size_assert]

    for i in range(1, len(image_list)):
        image = image_list[i]
        asserts.append(rank_assertions[i])

        with tf.control_dependencies([rank_assertions[i]]):
            shape = tf.shape(image)

        height = shape[0]
        width = shape[1]

        height_assert = tf.Assert(
            tf.equal(height, image_height), ["Wrong height for tensor [expected][actual]", height, image_height]
        )
        width_assert = tf.Assert(
            tf.equal(width, image_width), ["Wrong width for tensor [expected][actual]", width, image_width]
        )
        asserts.extend([height_assert, width_assert])

    # Create a random bounding box.
    #
    # Use tf.random_uniform and not numpy.random.rand as doing the former would
    # generate random numbers at graph eval time, unlike the latter which
    # generates random numbers at graph definition time.
    with tf.control_dependencies(asserts):
        max_offset_height = tf.reshape(image_height - crop_height + 1, [])
        max_offset_width = tf.reshape(image_width - crop_width + 1, [])
    offset_height = tf.random.uniform([], maxval=max_offset_height, dtype=tf.int32)
    offset_width = tf.random.uniform([], maxval=max_offset_width, dtype=tf.int32)

    return [_crop(image, offset_height, offset_width, crop_height, crop_width) for image in image_list]


def _image_dimensions(image, rank):
    """Returns the dimensions of an image tensor.

    Args:
        image: A rank-D Tensor. For 3-D  of shape: `[height, width, channels]`.
        rank: The expected rank of the image

    Returns:
        A list of corresponding to the dimensions of the input image. Dimensions
        that are statically known are python integers, otherwise they are integer
        scalar tensors.
    """
    if image.get_shape().is_fully_defined():
        return image.get_shape().as_list()
    else:
        static_shape = image.get_shape().with_rank(rank).as_list()
        dynamic_shape = tf.unstack(tf.shape(image), rank)
        return [s if s is not None else d for s, d in zip(static_shape, dynamic_shape)]


def pad_to_bounding_box(image, offset_height, offset_width, target_height, target_width, pad_value):
    """Pads the given image with the given pad_value.

    Works like tf.image.pad_to_bounding_box, except it can pad the image
    with any given arbitrary pad value and also handle images whose sizes are not
    known during graph construction.

    Args:
        image: 3-D tensor with shape [height, width, channels]
        offset_height: Number of rows of zeros to add on top.
        offset_width: Number of columns of zeros to add on the left.
        target_height: Height of output image.
        target_width: Width of output image.
        pad_value: Value to pad the image tensor with.

    Returns:
        3-D tensor of shape [target_height, target_width, channels].

    Raises:
        ValueError: If the shape of image is incompatible with the offset_* or
        target_* arguments.
    """

    with ops.name_scope(None, "pad_to_bounding_box", [image]):
        image = tf.convert_to_tensor(image, name="image")
        original_dtype = image.dtype
        if original_dtype != tf.float32 and original_dtype != tf.float64 and original_dtype != tf.bfloat16 and original_dtype != tf.float16:
            # If image dtype is not float, we convert it to int32 to avoid overflow.
            image = tf.cast(image, tf.int32)
        image_rank_assert = tf.Assert(
            tf.logical_or(tf.equal(tf.rank(image), 3), tf.equal(tf.rank(image), 4)), ["Wrong image tensor rank."]
        )

        with tf.control_dependencies([image_rank_assert]):
            image -= pad_value

        image_shape = image.get_shape()
        is_batch = True

        if image_shape.ndims == 3:
            is_batch = False
            image = tf.expand_dims(image, 0)
        elif image_shape.ndims is None:
            is_batch = False
            image = tf.expand_dims(image, 0)
            image.set_shape([None] * 4)
        elif image.get_shape().ndims != 4:
            raise ValueError("Input image must have either 3 or 4 dimensions.")

        _, height, width, _ = _image_dimensions(image, rank=4)

        target_width_assert = tf.Assert(tf.greater_equal(target_width, width), ["target_width must be >= width"])
        target_height_assert = tf.Assert(tf.greater_equal(target_height, height), ["target_height must be >= height"])

        with tf.control_dependencies([target_width_assert]):
            after_padding_width = target_width - offset_width - width
        with tf.control_dependencies([target_height_assert]):
            after_padding_height = target_height - offset_height - height

        '''
        offset_assert = tf.Assert(
            tf.logical_and(tf.greater_equal(after_padding_width, 0), tf.greater_equal(after_padding_height, 0)),
            ["target size not possible with the given target offsets"],
        )

        '''
        batch_params = tf.stack([0, 0])
        height_params = tf.stack([offset_height, after_padding_height])
        width_params = tf.stack([offset_width, after_padding_width])
        channel_params = tf.stack([0, 0])

        # with tf.control_dependencies([offset_assert]):
        paddings = tf.stack([batch_params, height_params, width_params, channel_params])

        padded = tf.pad(image, paddings)

        if not is_batch:
            padded = tf.squeeze(padded, axis=[0])

        outputs = padded + pad_value

        if outputs.dtype != original_dtype:
            outputs = tf.cast(outputs, original_dtype)

        return outputs


def flip_dim(tensor_list, prob=0.5, dim=1):

    """Randomly flips a dimension of the given tensor.
    The decision to randomly flip the `Tensors` is made together. In other words,
    all or none of the images pass in are flipped.
    Note that tf.random_flip_left_right and tf.random_flip_up_down isn't used so
    that we can control for the probability as well as ensure the same decision
    is applied across the images.
    Args:
        tensor_list: A list of `Tensors` with the same number of dimensions.
        prob: The probability of a left-right flip.
        dim: The dimension to flip, 0, 1, ..
    Returns:
        outputs: A list of the possibly flipped `Tensors` as well as an indicator
        `Tensor` at the end whose value is `True` if the inputs were flipped and
        `False` otherwise.
    Raises:
        ValueError: If dim is negative or greater than the dimension of a `Tensor`.
    """
    random_value = tf.random.uniform([])

    def flip():
        flipped = []
        for tensor in tensor_list:
            if dim < 0 or dim >= len(tensor.get_shape().as_list()):
                raise ValueError("dim must represent a valid dimension.")
            flipped.append(tf.reverse(tensor, [dim]))

        return flipped

    is_flipped = tf.less_equal(random_value, prob)
    outputs = tf.cond(is_flipped, flip, lambda: tensor_list)

    if not isinstance(outputs, (list, tuple)):
        outputs = [outputs]

    outputs.append(is_flipped)

    return outputs


def random_grayscale(tensor, prob=0.5):

    random_value = tf.random.uniform([])
    is_grayscale = tf.less(random_value, prob)

    def to_grayscale():
        grayscale = tf.image.rgb_to_grayscale(tensor)
        return tf.image.grayscale_to_rgb(grayscale)

    outputs = tf.cond(is_grayscale, to_grayscale, lambda: tensor)

    return outputs


def get_random_scale(min_scale_factor, max_scale_factor, step_size):
    """Gets a random scale value.
    Args:
        min_scale_factor: Minimum scale value.
        max_scale_factor: Maximum scale value.
        step_size: The step size from minimum to maximum value.
    Returns:
        A random scale value selected between minimum and maximum value.
    Raises:
        ValueError: min_scale_factor has unexpected value.
    """
    if min_scale_factor < 0 or min_scale_factor > max_scale_factor:
        raise ValueError("Unexpected value of min_scale_factor.")

    if min_scale_factor == max_scale_factor:
        return tf.cast(min_scale_factor, tf.float32)

    # When step_size = 0, we sample the value uniformly from [min, max).
    if step_size == 0:
        return tf.random_uniform([1], minval=min_scale_factor, maxval=max_scale_factor)

    # When step_size != 0, we randomly select one discrete value from [min, max].
    num_steps = int((max_scale_factor - min_scale_factor) / step_size + 1)
    scale_factors = tf.linspace(min_scale_factor, max_scale_factor, num_steps)
    shuffled_scale_factors = tf.random.shuffle(scale_factors)
    return shuffled_scale_factors[0]


def randomly_scale_image_and_label(image, label=None, scale_h=1.0, scale_w=1.0):
    """Randomly scales image and label.
    Args:
        image: Image with shape [height, width, 3].
        label: Label with shape [height, width, 1].
        scale_h: The height value to scale image and label.
        scale_w: The width value to scale image and label.
    Returns:
        Scaled image and label.
    """

    return tf.cond(
        tf.math.logical_and(tf.math.equal(scale_h, 1.0), tf.math.equal(scale_w, 1.0)),
        lambda: (image, label),
        lambda: _randomly_scale_image_and_label_internal(
            image, 
            label,
            scale_h=scale_h, 
            scale_w=scale_w
        ),
    )


def _randomly_scale_image_and_label_internal(image, label=None, scale_h=1.0, scale_w=1.0):

    image_shape = tf.shape(image)

    new_height = tf.cast(tf.cast(image_shape[0], tf.float32) * scale_h, tf.int32)
    new_width = tf.cast(tf.cast(image_shape[1], tf.float32) * scale_w, tf.int32)

    new_dim = (new_height, new_width)

    # Need squeeze and expand_dims because image interpolation takes
    # 4D tensors as input.
    image = resize_image(image, new_dim)

    if label is not None:
        label = resize_image(label, new_dim, method="nearest")

    return image, label



def resize_to_range(
    image,
    label=None,
    min_size=None,
    max_size=None,
    factor=None,
    label_layout_is_chw=False,
    scope=None,
    method=tf.image.ResizeMethod.BILINEAR,
):
    """Resizes image or label so their sides are within the provided range.
    The output size can be described by two cases:
    1. If the image can be rescaled so its minimum size is equal to min_size
        without the other side exceeding max_size, then do so.
    2. Otherwise, resize so the largest side is equal to max_size.
    An integer in `range(factor)` is added to the computed sides so that the
    final dimensions are multiples of `factor` plus one.
    Args:
        image: A 3D tensor of shape [height, width, channels].
        label: (optional) A 3D tensor of shape [height, width, channels] (default)
        or [channels, height, width] when label_layout_is_chw = True.
        min_size: (scalar) desired size of the smaller image side.
        max_size: (scalar) maximum allowed size of the larger image side. Note
        that the output dimension is no larger than max_size and may be slightly
        smaller than min_size when factor is not None.
        factor: Make output size multiple of factor plus one.
        label_layout_is_chw: If true, the label has shape [channel, height, width].
        We support this case because for some instance segmentation dataset, the
        instance segmentation is saved as [num_instances, height, width].
        scope: Optional name scope.
        method: Image resize method. Defaults to tf.image.ResizeMethod.BILINEAR.
    Returns:
        A 3-D tensor of shape [new_height, new_width, channels], where the image
        has been resized (with the specified method) so that
        min(new_height, new_width) == ceil(min_size) or
        max(new_height, new_width) == ceil(max_size).
    Raises:
        ValueError: If the image is not a 3D tensor.
    """
    with ops.name_scope(None, "resize_to_range", [image]):
        new_tensor_list = []
        min_size = tf.cast(min_size, tf.float32)
        if max_size is not None:
            max_size = tf.cast(max_size, tf.float32)
            # Modify the max_size to be a multiple of factor plus 1 and make sure the
            # max dimension after resizing is no larger than max_size.
            if factor is not None:
                max_size = max_size + (factor - (max_size - 1) % factor) % factor - factor

        [orig_height, orig_width, _] = resolve_shape(image, rank=3)
        orig_height = tf.cast(orig_height, tf.float32)
        orig_width = tf.cast(orig_width, tf.float32)
        orig_min_size = tf.minimum(orig_height, orig_width)

        # Calculate the larger of the possible sizes
        large_scale_factor = min_size / orig_min_size
        large_height = tf.cast(tf.math.ceil(orig_height * large_scale_factor), tf.int32)
        large_width = tf.cast(tf.math.ceil(orig_width * large_scale_factor), tf.int32)

        large_size = tf.stack([large_height, large_width])

        new_size = large_size
        if max_size is not None:
            # Calculate the smaller of the possible sizes, use that if the larger
            # is too big.
            orig_max_size = tf.maximum(orig_height, orig_width)
            small_scale_factor = max_size / orig_max_size
            small_height = tf.cast(tf.math.ceil(orig_height * small_scale_factor), tf.int32)
            small_width = tf.cast(tf.math.ceil(orig_width * small_scale_factor), tf.int32)
            small_size = tf.stack([small_height, small_width])
            new_size = tf.cond(
                tf.cast(tf.reduce_max(large_size), tf.float32) > max_size, lambda: small_size, lambda: large_size
            )
        # Ensure that both output sides are multiples of factor plus one.
        if factor is not None:
            new_size += (factor - (new_size - 1) % factor) % factor
        new_tensor_list.append(resize_image(image, new_size, method=method))
        if label is not None:
            if label_layout_is_chw:
                # Input label has shape [channel, height, width].
                resized_label = tf.expand_dims(label, 3)
                resized_label = resize_image(
                    resized_label, new_size, method="nearest"
                )

                resized_label = tf.squeeze(resized_label, 3)
            else:
                # Input label has shape [height, width, channel].
                resized_label = resize_image(
                    label, new_size, method="nearest"
                )

            new_tensor_list.append(resized_label)
        else:
            new_tensor_list.append(None)

        return new_tensor_list


@tf.function
def resize_to_max(size, image, label=None):

    image_dtype = image.dtype
    label_dtype = label.dtype

    image_shape = tf.shape(image)
    image_height = image_shape[0]
    image_width = image_shape[1]

    if image_height > size or image_width > size:
        image = tf.image.resize(
            image, size=(size, size), preserve_aspect_ratio=True, method=tf.image.ResizeMethod.BILINEAR
        )
        label = tf.compat.v1.image.resize_bilinear()

    return (image, label)


def random_pad_single_pixel(tensor, ignore_label=255):
    pass
    # tensor = tf.pad(tensor, paddings = )


def resolve_shape(tensor, rank=None, scope=None):
    """Fully resolves the shape of a Tensor.
    Use as much as possible the shape components already known during graph
    creation and resolve the remaining ones during runtime.
    Args:
        tensor: Input tensor whose shape we query.
        rank: The rank of the tensor, provided that we know it.
        scope: Optional name scope.
    Returns:
        shape: The full shape of the tensor.
    """
    with ops.name_scope(None, "resolve_shape", [tensor]):
        if rank is not None:
            shape = tensor.get_shape().with_rank(rank).as_list()
        else:
            shape = tensor.get_shape().as_list()

        if None in shape:
            shape_dynamic = tf.shape(tensor)
            for i in range(len(shape)):
                if shape[i] is None:
                    shape[i] = shape_dynamic[i]

        return shape


def is_resnet_beta(backbone_name):
    return backbone_name == ss.RESNET52 or backbone_name == ss.RESNET103