import tensorflow as tf 

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

    image = tf.identity(image, name="pad_to_bounding_box/images")
    original_dtype = image.dtype
    if original_dtype != tf.float32 and original_dtype != tf.float64:
        # If image dtype is not float, we convert it to int32 to avoid overflow.
        image = tf.cast(image, tf.int32)

    tf.debugging.assert_rank_in(image, [3, 4], "Wrong image tensor rank.")
    image -= pad_value
    image_shape = image.shape
    is_batch = True

    if len(image_shape) == 3:
        is_batch = False
        image = tf.expand_dims(image, 0)

    height = tf.shape(height)[1]
    width = tf.shape(width)[2]

    tf.debugging.assert_greater_equal(target_height, height, "target_height must be >= height")
    tf.debugging.assert_greater_equal(target_width, width, "target_width must be >= width")

    after_padding_width = target_width - offset_width - width
    after_padding_height = target_height - offset_height - height

    batch_params = tf.stack([0, 0])
    height_params = tf.stack([offset_height, after_padding_height])
    width_params = tf.stack([offset_width, after_padding_width])
    channel_params = tf.stack([0, 0])

    paddings = tf.stack([batch_params, height_params, width_params, channel_params])

    padded = tf.pad(image, paddings)

    if not is_batch:
        padded = tf.squeeze(padded, axis=[0])

    outputs = padded + pad_value

    if outputs.dtype != original_dtype:
        outputs = tf.cast(outputs, original_dtype)

    return outputs
