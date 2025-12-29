import tensorflow as tf

from iseg.utils.common import rep

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



def replace_nan(x, value=0.0):
    return tf.where(tf.math.is_nan(x), tf.ones_like(x) * value, x)


def replace_inf(x):

    inf_to_zero = tf.where(tf.math.is_inf(x), tf.zeros_like(x), x)
    max_value = tf.reduce_max(inf_to_zero)
    min_value = tf.reduce_min(inf_to_zero)

    return tf.clip_by_value(x, min_value, max_value)


def replace_nan_or_inf(x, nan_value=0.0):
    
    with tf.name_scope("replace_nan_or_inf"):
        return replace_inf(replace_nan(x, nan_value))



def guard_grads (x):
    @tf.custom_gradient
    def guard_grads_func (x):

        def grad (upstream):

            return replace_nan_or_inf(upstream)
        
        return x, grad
    
    return guard_grads_func(x)