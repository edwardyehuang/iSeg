import tensorflow as tf

from iseg.utils.common import isinstance_all

def get_sliding_start_indexs(length, crop_length):

    stride_rate = 2.0 / 3.0

    if isinstance_all([length, crop_length], int):
        return _get_sliding_start_indexs_py(length, crop_length, stride_rate)
    
    return _get_sliding_start_indexs_graph(length, crop_length, stride_rate)


def _get_sliding_start_indexs_py (length, crop_length, stride_rate):

    stride = int(stride_rate * crop_length)

    times = (length - crop_length) // stride + 1

    cond = length - (times - 1) * stride > crop_length

    cropped_indexs = []

    for i in range(times):
        cropped_indexs.append(stride * i)

    if cond:
        cropped_indexs.append(length - crop_length)

    return cropped_indexs


@tf.function
def _get_sliding_start_indexs_graph (length, crop_length, stride_rate):

    stride = tf.cast(stride_rate * tf.cast(crop_length, tf.float32), tf.int32)

    times = (length - crop_length) // stride + 1

    cond = length - (times - 1) * stride > crop_length

    array_len = times + tf.cast(cond, tf.int32)
    cropped_indexs = tf.TensorArray(tf.int32, size=array_len, dynamic_size=False, clear_after_read=False)

    for i in range(times):
        cropped_indexs = cropped_indexs.write(i, stride * i)

    if cond:
        cropped_indexs = cropped_indexs.write(times, length - crop_length)

    results = cropped_indexs.stack()
    cropped_indexs.close()

    return results