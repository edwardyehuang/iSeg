import tensorflow as tf

from iseg.utils.common import get_tensor_shape

def dynamic_padding_2d(x, paddings, constant_values=0):
    """
    Args:
        x: A tensor of shape [batch, height, width, channels].
        paddings: A tensor of shape [4] where paddings = [top, bottom, left, right].
        constant_values: A scalar value to pad with.
    Returns:
        A tensor of shape [batch, height + top + bottom, width + left + right, channels].
    """

    if not isinstance(paddings, tf.Tensor):
        paddings = tf.convert_to_tensor(paddings, dtype=tf.int32)
    
    top, bottom, left, right = tf.unstack(paddings)

    batch_size, height, width, channels = get_tensor_shape(x)

    output_width = left + width + right

    constant_values = tf.cast(constant_values, x.dtype)

    base_one = tf.ones([batch_size, 1, 1, channels], dtype=x.dtype, name="base_one") * constant_values

    top_pad = tf.image.resize(base_one, [top, output_width], method="nearest", name="top_pad")
    bottom_pad = tf.image.resize(base_one, [bottom, output_width], method="nearest", name="bottom_pad")
    
    left_pad = tf.image.resize(base_one, [height, left], method="nearest", name="left_pad")
    right_pad = tf.image.resize(base_one, [height, right], method="nearest", name="right_pad")

    x = tf.concat([left_pad, x, right_pad], axis=2) # [batch, height, width + left + right, channels]
    x = tf.concat([top_pad, x, bottom_pad], axis=1) # [batch, height + top + bottom, width + left + right, channels]

    return x

