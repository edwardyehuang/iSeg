import tensorflow as tf

def _large_compatible_negative(tensor_type):
    """Large negative number as Tensor.

    This function is necessary because the standard value for epsilon
    in this module (-1e9) cannot be represented using tf.float16

    Args:
        tensor_type: a dtype to determine the type.

    Returns:
        a large negative number.
    """
    # In case of dtype=float16 (e.g., for mixed-precision), the largest
    # negative number (dtypes.float16.min) is divided by 2, in order to
    # avoid overflows when summing negative inputs.
    if tensor_type == tf.float16:
        return tf.float16.min / 2.0
    return -1e9


def safed_softmax (x, mask=None):
    t = x.dtype

    if mask is not None:
        mask_add = (1.0 - tf.cast(mask, x.dtype)) * _large_compatible_negative(t)
        x += mask_add

    if t == tf.float16:
        x = tf.cast(x, tf.float32)
        x = tf.nn.softmax(x)
        x = tf.cast(x, t)
    else:
        x = tf.nn.softmax(x)

    return x