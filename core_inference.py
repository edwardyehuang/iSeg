# ================================================================
# MIT License
# Copyright (c) 2021 edwardyehuang (https://github.com/edwardyehuang)
# ================================================================

import tensorflow as tf


@tf.function
def internel_inference(inputs, model, training=None):

    return model(inputs, training=training)


def inference_fn(inputs, model, num_class=21, training=False, sliding_window_crop_size=None):

    if sliding_window_crop_size is None:
        model_results = internel_inference(inputs, model, training=training)
    else:
        model_results = inference_with_sliding_window(
            inputs, num_class=num_class, model=model, training=training, windows_size=sliding_window_crop_size
        )
    return model_results


def get_sliding_start_indexs_v2(length, crop_length):

    stride_rate = 2.0 / 3.0

    stride = tf.cast(stride_rate * tf.cast(crop_length, tf.float32), tf.int32)

    times = (length - crop_length) // stride + 1

    cond = length - (times - 1) * stride > crop_length

    array_len = times + 1 if cond else times
    cropped_indexs = tf.TensorArray(tf.int32, size=array_len, dynamic_size=False, clear_after_read=False)

    for i in range(times):
        cropped_indexs = cropped_indexs.write(i, stride * i)

    if cond:
        cropped_indexs = cropped_indexs.write(times, length - crop_length)

    results = cropped_indexs.stack()
    cropped_indexs.close()

    return results


def check_if_tuple_or_list(inputs):

    return isinstance(inputs, list) or isinstance(inputs, tuple)


def convert_to_list_if_single(inputs):

    if not check_if_tuple_or_list(inputs):
        return [inputs]

    return inputs


def free_from_list_if_single(inputs):

    if not check_if_tuple_or_list(inputs):
        raise ValueError("Inputs already single")

    if len(inputs) == 1:
        return inputs[0]

    return inputs


def __check_if_seg_map(tensor):

    return len(tensor.shape) >= 3

    """
    tensor_shape = tf.shape(tensor)
    
    return (tf.rank(tensor) >= 3 and
        (crop_size is None or (crop_size[0] == tensor_shape[1] and crop_size[1] == tensor_shape[2])))
    """


def result_handler(inputs, seg_map_handler, others_handler=None):

    if __check_if_seg_map(inputs):
        return seg_map_handler(inputs)
    elif others_handler is not None:
        return others_handler(inputs)
    else:
        return inputs


def multi_results_handler(multi_inputs, seg_map_handler, others_handler=None):

    results = []

    for x in multi_inputs:
        y = result_handler(x, seg_map_handler, others_handler)
        results.append(y)

    return results


def multi_results_add(v0, v1):
    return [a + b for a, b in zip(v0, v1)]


def create_base_tensor_for_cropped_result(tensor, full_size):
    def seg_map_handler(x):
        tensor_shape = tf.shape(x)
        return tf.zeros(tf.stack([tensor_shape[0], full_size[0], full_size[1], tensor_shape[-1]]), dtype=x.dtype)

    return multi_results_handler(tensor, seg_map_handler, lambda x: tf.zeros_like(x))


def get_sliding_window_slices_paddings_list(stride_h, stride_w, inputs_height, inputs_width):

    sliding_indexs_h = get_sliding_start_indexs_v2(inputs_height, stride_h)  # [None]
    sliding_indexs_w = get_sliding_start_indexs_v2(inputs_width, stride_w)  # [None]

    inference_count_map = tf.zeros(tf.stack([1, inputs_height, inputs_width, 1]), tf.int32)
    cropped_onces = tf.ones(tf.stack([1, stride_h, stride_w, 1]), tf.int32)  # [1, window_h, window_w, 1]

    sliding_indexs_h_len = tf.shape(sliding_indexs_h)[0]
    sliding_indexs_w_len = tf.shape(sliding_indexs_w)[0]

    total_sliding_indexs_len = sliding_indexs_h_len * sliding_indexs_w_len

    slices_list = tf.TensorArray(
        tf.int32, size=total_sliding_indexs_len, dynamic_size=False, clear_after_read=False, name="slices_list"
    )
    paddings_list = tf.TensorArray(
        tf.int32, size=total_sliding_indexs_len, dynamic_size=False, clear_after_read=False, name="paddings_list"
    )

    for i in range(total_sliding_indexs_len):

        j = i // sliding_indexs_w_len
        k = i % sliding_indexs_w_len

        top = sliding_indexs_h[j]
        bottom = top + stride_h
        left = sliding_indexs_w[k]
        right = left + stride_w

        pad_bottom = inputs_height - bottom
        pad_right = inputs_width - right

        paddings = [[0, 0], [top, pad_bottom], [left, pad_right], [0, 0]]
        inference_count_map += tf.pad(cropped_onces, paddings)

        slice_indexs = [top, bottom, left, right]

        slices_list = slices_list.write(i, slice_indexs)
        paddings_list = paddings_list.write(i, paddings)

    slices_list_result = slices_list.stack()
    paddings_list_result = paddings_list.stack()

    slices_list.close()
    paddings_list.close()

    return (slices_list_result, paddings_list_result, inference_count_map)


def inference_with_sliding_window(inputs, model, num_class=21, training=False, windows_size=(769, 769)):

    if windows_size is None:
        raise ValueError("Window size must not be None !!!!!!!!")

    inputs_shape = tf.shape(inputs)
    inputs_height = inputs_shape[1]
    inputs_width = inputs_shape[2]

    # stride_h = windows_size[0] if inputs_height > windows_size[0] else inputs_height
    # stride_w = windows_size[1] if inputs_width > windows_size[1] else inputs_width

    stride_h = tf.where(inputs_height > windows_size[0], windows_size[0], inputs_height)
    stride_w = tf.where(inputs_width > windows_size[1], windows_size[1], inputs_width)

    slices_list, paddings_list, inference_count_map = get_sliding_window_slices_paddings_list(
        stride_h, stride_w, inputs_height, inputs_width
    )

    results = None

    total_sliding_indexs_len = tf.shape(slices_list)[0]

    def loop_body(i, results=None):

        slices_indexs = slices_list[i]
        paddings = paddings_list[i]

        cropped_inputs = inputs[:, slices_indexs[0] : slices_indexs[1], slices_indexs[2] : slices_indexs[3], :]

        cropped_results = internel_inference(cropped_inputs, model, training=training)

        cropped_results = convert_to_list_if_single(cropped_results)

        if results is None:
            results = create_base_tensor_for_cropped_result(cropped_results, (inputs_height, inputs_width))

        cropped_results = multi_results_handler(cropped_results, seg_map_handler=lambda x: tf.pad(x, paddings))

        results = multi_results_add(results, cropped_results)

        return i, results

    _, results = loop_body(tf.constant(0))

    for i in range(1, total_sliding_indexs_len):
        _, results = loop_body(i, results)

    inference_count_map = tf.cast(inference_count_map, dtype=results[0].dtype)

    results = multi_results_handler(results, lambda r: r / inference_count_map)

    # sliding_indexs_h.close()
    # sliding_indexs_w.close()

    results = free_from_list_if_single(results)

    return results
