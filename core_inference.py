# ================================================================
# MIT License
# Copyright (c) 2021 edwardyehuang (https://github.com/edwardyehuang)
# ================================================================

import tensorflow as tf

from iseg.utils.common import get_tensor_shape, smart_where, isinstance_all
from iseg.utils.sliding_window_inference_utils import get_sliding_start_indexs
from iseg.utils.tensor_utils import dynamic_padding_2d

def extract_seq_input_signatures (inputs):

    if isinstance(inputs, tuple):
        inputs = list(inputs)
    
    if not isinstance(inputs, list):
        inputs = [inputs]

    input_signatures = []

    for x in inputs:

        shapes = get_tensor_shape(x, return_list=True)

        if len(shapes) >= 3:
            shapes[0] = None
            shapes[1] = None
            shapes[2] = None

        input_signatures.append(tf.TensorSpec(shape=shapes, dtype=x.dtype))


    return input_signatures


def internel_inference(inputs, model, training=None):

    # input_signatures = extract_seq_input_signatures(inputs)

    return model(inputs, training=training)

@tf.autograph.experimental.do_not_convert
def inference_fn(inputs, model, num_class=21, training=False, sliding_window_crop_size=None):

    if sliding_window_crop_size is None:
        model_results = internel_inference(inputs, model, training=training)
    else:
        model_results = inference_with_sliding_window(
            inputs, model=model, training=training, windows_size=sliding_window_crop_size
        )

        # print(inference_with_sliding_window.pretty_printed_concrete_signatures())

    return model_results


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

@tf.function(autograph=True)
def create_base_tensor_for_cropped_result(tensor, full_size):
    def seg_map_handler(x):
        tensor_shape = tf.shape(x)
        return tf.zeros(tf.stack([tensor_shape[0], full_size[0], full_size[1], tensor_shape[-1]]), dtype=x.dtype)

    return multi_results_handler(tensor, seg_map_handler, lambda x: tf.zeros_like(x))

@tf.function(autograph=False)
def get_sliding_window_slices_paddings_list(stride_h, stride_w, inputs_height, inputs_width):

    # print("trace: get_sliding_window_slices_paddings_list")

    sliding_indexs_h = get_sliding_start_indexs(inputs_height, stride_h)  # [None]
    sliding_indexs_w = get_sliding_start_indexs(inputs_width, stride_w)  # [None]

    inference_count_map = tf.zeros(tf.stack([1, inputs_height, inputs_width, 1]), tf.int32)
    cropped_onces = tf.ones(tf.stack([1, stride_h, stride_w, 1]), tf.int32)  # [1, window_h, window_w, 1]

    if isinstance_all([sliding_indexs_h, sliding_indexs_w], list):
        sliding_indexs_h_len = len(sliding_indexs_h)
        sliding_indexs_w_len = len(sliding_indexs_w)
    else:
        sliding_indexs_h_len = tf.shape(sliding_indexs_h)[0]
        sliding_indexs_w_len = tf.shape(sliding_indexs_w)[0]

    total_sliding_indexs_len = sliding_indexs_h_len * sliding_indexs_w_len

    slices_list = tf.TensorArray(
        tf.int32, size=total_sliding_indexs_len, dynamic_size=False, clear_after_read=False, name="slices_list"
    )
    paddings_list = tf.TensorArray(
        tf.int32, size=total_sliding_indexs_len, dynamic_size=False, clear_after_read=False, name="paddings_list"
    )

    # for i in tf.range(total_sliding_indexs_len):

    def loop_body(i, _slices_list, _paddings_list, _inference_count_map):
        j = i // sliding_indexs_w_len
        k = i % sliding_indexs_w_len

        top = sliding_indexs_h[j]
        bottom = top + stride_h
        left = sliding_indexs_w[k]
        right = left + stride_w

        pad_bottom = inputs_height - bottom
        pad_right = inputs_width - right

        paddings = [top, pad_bottom, left, pad_right]
        # paddings = [[0, 0], [top, pad_bottom], [left, pad_right], [0, 0]]
        _inference_count_map += dynamic_padding_2d(cropped_onces, paddings, constant_values=0)

        slice_indexs = [top, bottom, left, right]

        _slices_list = _slices_list.write(i, slice_indexs)
        _paddings_list = _paddings_list.write(i, paddings)

        return tf.add(i, 1), _slices_list, _paddings_list, _inference_count_map
    
    _, slices_list, paddings_list, inference_count_map = tf.while_loop(
        lambda i, *_: i < total_sliding_indexs_len,
        loop_body,
        [0, slices_list, paddings_list, inference_count_map],
    )

    slices_list_result = slices_list.stack()
    paddings_list_result = paddings_list.stack()

    slices_list.close()
    paddings_list.close()

    return (slices_list_result, paddings_list_result, inference_count_map)



@tf.function(autograph=False, reduce_retracing=True)
def sliding_window_body(i, inputs, model, slices_list, training=False):

    # print("trace: inference_with_sliding_window window_body")

    slices_indexs = slices_list[i]

    cropped_inputs = inputs[:, slices_indexs[0] : slices_indexs[1], slices_indexs[2] : slices_indexs[3], :]

    cropped_results = internel_inference(cropped_inputs, model, training=training)

    cropped_results = convert_to_list_if_single(cropped_results)

    return cropped_results



@tf.autograph.experimental.do_not_convert
def inference_with_sliding_window(inputs, model, training=False, windows_size=(769, 769)):

    if windows_size is None:
        raise ValueError("Window size must not be None !!!!!!!!")

    _, inputs_height, inputs_width, _ = get_tensor_shape(inputs)

    stride_h = smart_where(inputs_height > windows_size[0], windows_size[0], inputs_height)
    stride_w = smart_where(inputs_width > windows_size[1], windows_size[1], inputs_width)

    slices_list, paddings_list, inference_count_map = get_sliding_window_slices_paddings_list(
        stride_h, stride_w, inputs_height, inputs_width
    )

    total_sliding_indexs_len = tf.shape(slices_list)[0]

    
    @tf.function(autograph=False, reduce_retracing=True)
    def loop_body(i, results=None):

        paddings = paddings_list[i]
        cropped_results = sliding_window_body(i, inputs, model, slices_list, training=training)

        if results is None:
            results = create_base_tensor_for_cropped_result(
                cropped_results, 
                (inputs_height, inputs_width)
            )

        cropped_results = multi_results_handler(
            cropped_results, 
            seg_map_handler=lambda x: dynamic_padding_2d(x, paddings, constant_values=0),
        )

        results = multi_results_add(results, cropped_results)

        return tf.add(i, 1), results

    _, results = loop_body(tf.constant(0))

    _, results = tf.while_loop(
        lambda i, _: i < total_sliding_indexs_len, 
        loop_body, 
        [1, results]
    )

    inference_count_map = tf.cast(inference_count_map, dtype=results[0].dtype)

    results = multi_results_handler(results, lambda r: r / inference_count_map)
    results = free_from_list_if_single(results)

    return results
